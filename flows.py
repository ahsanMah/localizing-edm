import os
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from scorer import EDMScorer, VEScorer

import pickle
import dnnlib
from torch.utils.data import Subset
from functools import partial
import click
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import json
from torchinfo import summary
from torch.nn.functional import interpolate
import math
from torch_utils import misc


_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
def logprob(z, ldj):
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=1) + ldj

def subnet_fc(c_in, c_out, ndim=128):
    return nn.Sequential(
        nn.Linear(c_in, ndim),
        nn.GELU(),
        nn.Linear(ndim, ndim),
        nn.GELU(),
        nn.Linear(ndim, c_out),
    )


def subnet_conv(c_in, c_out, ndim=256):
    return nn.Sequential(
        nn.Conv2d(c_in, ndim, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(ndim, c_out, 3, padding=1),
    )


def subnet_conv_1x1(c_in, c_out, ndim=256):
    return nn.Sequential(nn.Conv2d(c_in, ndim, 1), nn.GELU(), nn.Conv2d(ndim, c_out, 1))


# Taken from CFlow-AD repo
# https://github.com/gudovskiy/cflow-ad/blob/b2ebf9e673a0aa46992a3b18367ec066a57bba89/model.py
def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(
                D
            )
        )
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    )
    P[1:D:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    )
    P[D::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    )
    P[D + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    )
    return P


class PatchFlow(torch.nn.Module):
    """
    Contructs a conditional flow model that operates on patches of an image.
    Each patch is fed into a separate flow model.
    The flow models are conditioned on a positional encoding of the patch location.
    The resulting patch-densities can then be recombined into a full image density.
    """

    def __init__(self, input_size, patch_size=3, spatial_cond_dim=32, num_blocks=2):
        super().__init__()

        channels = input_size[0]

        with torch.no_grad():
            self.pooler = torch.nn.AvgPool2d(
                patch_size, stride=2, padding=1
            ).requires_grad_(False)
            _, c, h, w = self.pooler(torch.empty(1, *input_size)).shape

            self.spatial_res = (h, w)
            self.num_patches = h * w
            self.channels = channels
            pos = positionalencoding2d(spatial_cond_dim, h, w)
            pos = pos.reshape(spatial_cond_dim, self.num_patches)
            pos = pos.permute(1, 0)  # N x C_d
            self.register_buffer("positional_encoding", pos)
            print(
                f"Generating {patch_size}x{patch_size} patches from input size: {input_size}"
            )
            print(f"Pooled spatial resolution: {self.spatial_res}")
            print(f"Number of flows / patches: {self.num_patches}")

        # # Each patch gets fed into a flow model
        # self.flows = {f"patch_{i}":
        #               self.build_cflow_head(cond_dim, channels) for i in range(self.num_patches)}
        # self.flows = nn.ModuleDict(self.flows)

        self.highres_flow = build_conv_model(input_size, levels=2, num_blocks=2)
        # Output of high res flow is downsampled to match the spatial resolution of the patches
        # Downsampling redistributes the spatial dims across the channels
        # 2x downsampling -> 4x channels
        global_cond_dim = channels * 4

        cond_dims = spatial_cond_dim  # + global_cond_dim
        self.flow = self.build_cflow_head(
            cond_dims, channels + global_cond_dim, num_blocks
        )

        # for m in self.modules():
        #     print(m)
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight.data)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias.data)

    def build_cflow_head(self, n_cond, n_feat, num_blocks=2):
        coder = Ff.SequenceINN(n_feat)
        for k in range(num_blocks):
            # idx = int(k % 2 == 0)
            coder.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(n_cond,),
                subnet_constructor=subnet_fc,
                global_affine_type="SOFTPLUS",
                permute_soft=True,
            )

        return coder

    def patchify(self, x):
        b, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)  # Patches x batch x channels
        x = x.reshape(self.num_patches, b, c)
        return x

    def forward(self, x):
        B = x.shape[0]

        hres_zs, hres_log_jac_dets = self.highres_flow(x)
        global_context_patches = self.patchify(hres_zs)
        xs = self.pooler(x)
        local_patches = self.patchify(xs)  # Patches x batch x channels

        zs = [hres_zs.reshape(B, -1)]
        log_jac_dets = [hres_log_jac_dets]
        pos = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)
        for patch_feature, glb_ctx, spatial_cond in zip(
            local_patches, global_context_patches, pos
        ):
            # print(patch_feature.shape, glb_ctx.shape, spatial_cond.shape)
            z, ldj = self.flow(
                # patch_feature,
                # c=[torch.cat([glb_ctx, spatial_cond], dim=-1)],
                torch.cat([patch_feature, glb_ctx], dim=-1),
                c=[spatial_cond],
            )
            zs.append(z)
            log_jac_dets.append(ldj)

        return zs, log_jac_dets

    def stochastic_train_step(self, x, n_patches=1):
        B = x.shape[0]

        hres_zs, hres_log_jac_dets = self.highres_flow(x)
        global_context_patches = self.patchify(hres_zs)
        xs = self.pooler(x)
        local_patches = self.patchify(xs)  # Patches x batch x channels
        # local_patches = self.patchify(hres_zs)

        zs = []
        log_jac_dets = []
        pos = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)

        rand_idx = torch.randperm(self.num_patches)[:n_patches]
        for idx in rand_idx:
            patch_feature, glb_ctx, spatial_cond = (
                local_patches[idx],
                global_context_patches[idx],
                pos[idx],
            )
            z, ldj = self.flow(
                # patch_feature,
                # c=[torch.cat([glb_ctx, spatial_cond], dim=-1)],
                torch.cat([patch_feature, glb_ctx], dim=-1),
                c=[spatial_cond],
            )
            zs.append(z)
            log_jac_dets.append(ldj)

        return zs, log_jac_dets, hres_log_jac_dets

    def reverse(self, zs):
        B = zs[0].shape[0]
        pos = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)
        xs = []

        for i, (patch_feature, spatial_cond) in enumerate(zip(zs, pos)):
            x, ldj = self.flow(
                patch_feature,
                c=[
                    spatial_cond,
                ],
                rev=True,
            )
            xs.append(x)
        return xs

    def reconstruct(self, zs):
        """
        Reconstructs image from a list of patch densities
        """
        hres_logpx = zs.pop(0)[:, None, None]
        # x is num_patches x batch
        x = torch.stack(zs)
        # Swap batch and patch dim
        x = x.permute(1, 0)
        x = x.reshape(x.shape[0], *self.spatial_res)
        return x  # + hres_logpx

    def log_density(self, x):
        self.eval()
        with torch.no_grad():
            zs, jacs = self(x)
            logpzs = [-0.5 * torch.sum(z**2, dim=1) + ldj for z, ldj in zip(zs, jacs)]
            logpx = self.reconstruct(logpzs)
        return logpx


def patchflow_stochastic_step(flownet, x, opt, n_samples=128):
    n_samples = min(n_samples, flownet.flow.num_patches)
    scores = flownet.scorenet(x)
    # Optimizing the highres flow
    # Note this could be combined with the patch loss below
    # But dual optimizer step is more stable
    opt.zero_grad(set_to_none=True)
    hres_zs, hres_log_jac_dets = flownet.flow.highres_flow(scores)
    hres_loss = nll(hres_zs.reshape(x.shape[0], -1), hres_log_jac_dets)
    hres_loss = hres_loss / n_samples
    hres_loss.backward()
    opt.step()

    total_loss = hres_loss.item()
    # Optimizing the patch flow
    zs, log_jac_dets, _ = flownet.flow.stochastic_train_step(
        scores, n_patches=n_samples
    )
    patch_losses = [nll(z, ldj) for z, ldj in zip(zs, log_jac_dets)]
    loss = 0.0
    opt.zero_grad(set_to_none=True)
    for l in patch_losses:
        loss = loss + l
    loss.backward()
    opt.step()

    total_loss += loss.item()

    return {
        "loss": total_loss,
        "global_loss": hres_loss.item(),
        "local_loss": loss.item(),
    }


class ScoreFlow(torch.nn.Module):
    def __init__(
        self,
        flow,
        scorenet,
        vectorize=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **scorenet_kwargs,
    ):
        super().__init__()
        self.flow = flow.to(device)
        self.scorenet = VEScorer(
            scorenet, device=device, **scorenet_kwargs
        ).requires_grad_(False)

        if vectorize:
            self.fastscore = self.scorenet.build_vectorized_forward()
            # self.fastscore = torch.jit.script(self.fastscore)

        # Initialize weights with Xavier
        for m in self.flow.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x, vectorize=False, **score_kwargs):
        with torch.no_grad():
            if vectorize:
                x_scores = self.fastscore(x)
            else:
                x_scores = self.scorenet(x, **score_kwargs)

        return self.flow(x_scores.contiguous())

    @torch.inference_mode()
    def log_density(self, x) -> torch.Tensor:
        z, log_jac_det = self.forward(x)
        logpz = -0.5 * torch.sum(z**2, dim=1, keepdim=True) + log_jac_det.reshape(
            z.shape[0], *[1] * len(z.shape[1:])
        )
        return logpz

    def score_image(self, x):
        return -self.log_density(x)

    @torch.inference_mode()
    def score_patches(self, x):
        return self.flow.log_density(self.scorenet(x))


def load_pretrained_model(network_pkl):
    with dnnlib.util.open_url(network_pkl) as f:
        with torch.no_grad():
            model = pickle.load(f)

    config = dnnlib.EasyDict(
        dataset_kwargs=dnnlib.EasyDict(**model["dataset_kwargs"]),
    )
    if model["augment_pipe"] is not None:
        config.augment_pipe = dnnlib.EasyDict(**model["augment_pipe"].init_kwargs)

    net = model["ema"]
    return net, config


# TODO: Make a U-Net style model
def build_conv_model(
    input_dims, num_filters=256, levels=2, num_blocks=2, num_fc_blocks=0, unet=False
):
    nodes = [Ff.InputNode(*input_dims, name="input")]

    # Higher resolution convolutional part
    for k in range(num_blocks):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.AllInOneBlock,
                {"subnet_constructor": subnet_conv},
                name=f"conv_high_res_{k}",
            )
        )
        # nodes.append(
        #     Ff.Node(
        #         nodes[-1],
        #         Fm.AllInOneBlock,
        #         {"subnet_constructor": subnet_conv, "gin_block": False},
        #     )
        # )

    # Lower resolution convolutional part
    levels -= 1
    for i in range(levels):
        nodes.append(
            Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name=f"downsample_{i}")
        )

        for k in range(num_blocks):
            if k % 2 == 0:
                subnet = partial(subnet_conv_1x1, ndim=num_filters)
            else:
                subnet = partial(subnet_conv, ndim=num_filters)

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet_conv},
                    name=f"conv_high_res_{k}",
                )
            )

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet_conv_1x1},
                )
            )

    if unet:
        # Upsampling part
        for i in range(levels):
            nodes.append(Ff.Node(nodes[-1], Fm.HaarUpsampling, {}))

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet_conv},
                    name=f"conv_high_res_{k}",
                )
            )
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet_conv, "gin_block": True},
                )
            )

    if num_fc_blocks > 0:
        ndim_x = np.prod(input_dims)
        # Make the outputs into a vector, then split off 1/4 of the outputs for the
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name="flatten"))
        split_node = Ff.Node(
            nodes[-1],
            Fm.Split,
            {"section_sizes": (ndim_x // 4, 3 * ndim_x // 4), "dim": 0},
            name="split",
        )
        nodes.append(split_node)

        # Fully connected part
        for k in range(num_fc_blocks):
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {"subnet_constructor": subnet_fc, "clamp": 2.0},
                    name=f"fully_connected_{k}",
                )
            )
            nodes.append(
                Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_{k}")
            )

        # Concatenate the fully connected part and the skip connection to get a single output
        nodes.append(
            Ff.Node(
                [nodes[-1].out0, split_node.out1],
                Fm.Concat1d,
                {"dim": 0},
                name="concat",
            )
        )

        nodes.append(
            Ff.Node(nodes[-1], Fm.Reshape, {"output_dims": input_dims}, name="reshape")
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    conv_inn = Ff.GraphINN(nodes)

    return conv_inn


# @torch.inference_mode()
# def log_density(inn, x) -> torch.Tensor:
#     z, log_jac_det = inn(x)
#     logpz = -0.5 * torch.sum(z**2, dim=1, keepdim=True) - log_jac_det.reshape(
#         z.shape[0], *[1] * len(z.shape[1:])
#     )
#     return logpz


@torch.inference_mode()
def per_dim_ll(inn, x):
    inn.eval()
    z, log_jac_det = inn(x)
    z = z.cpu().numpy()
    log_jac_det = log_jac_det.cpu().numpy()
    logpz = -0.5 * z**2 + log_jac_det.reshape(-1, 1)
    return logpz


# @torch.jit.script
def logloss(z, log_jac_det):
    bsz = z.shape[0]
    z = z.reshape(bsz, -1)
    log_jac_det = log_jac_det.reshape(bsz, -1)
    ll = logprob(z, log_jac_det)
    nll = -torch.mean(ll)
    return nll


def nll(z, ldj):
    # logpz = -0.5 * torch.sum(z**2, dim=1) + ldj
    return -torch.mean(logprob(z, ldj))


def build_nd_flow(input_dim, num_blocks=4, ndim=128):
    inn = Ff.SequenceINN(input_dim)
    for k in range(num_blocks):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

    return inn


def train_flow(
    inn, dataset_iterator, num_epochs=100, lr=3e-4, device=torch.device("cuda")
):
    losses = []
    opt = torch.optim.Adam(inn.parameters(), lr=lr)
    progbar = tqdm(range(num_epochs))

    inn.train()
    for epoch in progbar:
        niter = 0
        for x, _ in dataset_iterator:
            opt.zero_grad()
            x = x.float().to(device)
            z, log_jac_det = inn(x)
            loss = logloss(z, log_jac_det)
            loss.backward()
            opt.step()

            if niter % 5 == 0:
                progbar.set_description(f"Loss: {loss.item():.4f}")
                losses.append(loss.item())

            niter += 1

        progbar.set_description(f"Loss: {loss.item():.4f}")

    return losses


def patchflow_step(flownet, x, opt):
    # scores = flownet.scorenet(x)
    zs, log_jac_dets = flownet(x)

    total_loss = 0.0
    # for z, ljd in zip(zs, log_jac_dets):
    #     opt.zero_grad(set_to_none=True)
    #     loss = -0.5*torch.sum(z**2, dim=1) + ljd
    #     loss = torch.mean(-loss)
    #     print(loss)
    #     loss.backward()
    #     total_loss += loss.item()
    #     opt.step()

    opt.zero_grad(set_to_none=True)
    patch_losses = [nll(z, ldj) for z, ldj in zip(zs, log_jac_dets)]
    loss = 0.0
    for l in patch_losses:
        loss += l
    loss.backward()
    opt.step()

    return loss.item()


def fastflow_step(flownet, x, opt):
    opt.zero_grad()
    z, log_jac_det = flownet(x)
    loss = logloss(z, log_jac_det)
    loss.backward()
    opt.step()
    return {"loss": loss.item()}


def train_msma_flow(
    flownet,
    train_ds,
    val_ds,
    augment_pipe=None,
    kimg=10,
    lr=3e-4,
    device=torch.device("cuda"),
    run_dir="runs/",
    log_interval=5,
    checkpoint_interval=20,
    log_tensorboard=False,
    patchflow=False,
):
    losses = []
    opt = torch.optim.Adam(flownet.flow.parameters(), lr=lr)
    batch_sz = train_ds.batch_size
    total_iters = kimg * 1000 // batch_sz + 1
    progbar = tqdm(range(total_iters))
    train_step = patchflow_stochastic_step if patchflow else fastflow_step
    checkpoint_path = f"{run_dir}/checkpoint.pth"

    if log_tensorboard:
        writer = SummaryWriter(log_dir=run_dir)

    flownet = flownet.to(device)
    flownet.train()

    niter = 0
    imgcount = 0
    train_iter = iter(train_ds)
    val_iter = iter(val_ds)

    for niter in progbar:
        x, _ = next(train_iter)
        x = x.to(device).to(torch.float32) / 127.5 - 1
        if augment_pipe:
            x, _ = augment_pipe(x)

        loss_dict = train_step(flownet, x, opt)

        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(f"train_loss/{loss_type}", loss_dict[loss_type], niter)

        if niter % log_interval == 0:
            flownet.eval()

            with torch.no_grad():
                val_loss = 0.0
                x, _ = next(val_iter)
                x = x.to(device).to(torch.float32) / 127.5 - 1
                z, log_jac_det = flownet(x)
                if patchflow:
                    for z, ldj in zip(z, log_jac_det):
                        val_loss += nll(z, ldj).item()
                else:
                    val_loss += logloss(z, log_jac_det).item()

            flownet.train()

            progbar.set_description(f"Val Loss: {val_loss:.4f}")
            if log_tensorboard:
                writer.add_scalar("val_loss", val_loss, niter)
            losses.append(val_loss)

        imgcount += x.shape[0]
        progbar.set_postfix(batch=f"{imgcount}/{kimg}K")

    if niter % checkpoint_interval == 0:
        torch.save(
            {
                "epoch": -1,
                "kimg": niter,
                "model_state_dict": flownet.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        # progbar.set_description(f"Loss: {loss:.4f}")

    torch.save(
        {
            "epoch": -1,
            "kimg": niter,
            "model_state_dict": flownet.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    if log_tensorboard:
        writer.close()

    return losses


def build_dataset(dataset_kwargs, augment_prob=0.1, val_ratio=0.1):
    # Build dataset
    dataset_kwargs.update(xflip=False)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    train_len = int((1 - val_ratio) * dataset_kwargs.max_size)
    val_len = dataset_kwargs.max_size - train_len

    augment_kwargs = dnnlib.EasyDict(
        class_name="training.augment.AugmentPipe", p=augment_prob
    )
    augment_kwargs.update(
        xflip=1e8, yflip=0, scale=1, rotate_frac=1, aniso=1, translate_frac=1
    )

    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_prob > 0
        else None
    )

    # Subset of training dataset
    train_ds = Subset(dataset_obj, range(train_len))
    val_ds = Subset(dataset_obj, range(train_len, train_len + val_len))

    return train_ds, val_ds, augment_pipe


@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--run_dir",
    help="Directory to save runs.",
    metavar="PATH|URL",
    type=str,
    required=True,
)
# Data options
@click.option(
    "--resolution", default=None, type=int, help="Resolution of images to train on."
)
@click.option("--augment", default=0.1, help="Probability of augmentation.")
# Flow options
@click.option("--sigma_min", default=0.1, help="Minimum sigma for the score norms.")
@click.option("--num_sigmas", default=10, help="Number of sigmas for the score norms.")
@click.option("--levels", default=2, type=int, help="Number of blocks for each level.")
@click.option(
    "--num_blocks", default=2, type=int, help="Number of blocks for each level."
)
@click.option(
    "--unet",
    default=False,
    help="Use U-Net-like architecture (for the lower levels only).",
)
# Optimization options
@click.option("--num_epochs", default=10, help="Number of epochs to train.")
@click.option("--kimg", default=10, help="Number of images to train for in 1000s.")
@click.option("--batch_size", default=128, help="Batch size.")
@click.option("--lr", default=3e-4, help="Learning rate.")
@click.option("--fp16", default=False, help="Use fp16 (applies to scorenet only).")
@click.option("--workers", default=4, help="Number of workers.")
@click.option("--device", default="cuda", help="Device to use.")
@click.option("--seed", default=42, type=int, help="Device to use.")
def main(network_pkl, **kwargs):
    # Load network alongside with its training config.
    print('Loading network from "%s"' % network_pkl)
    scorenet, model_config = load_pretrained_model(network_pkl=network_pkl)
    print("Loaded network")

    config = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict(**model_config)
    device = torch.device(config.device)

    if config.resolution is not None:
        c.dataset_kwargs.resolution = int(config.resolution)

    # Build datasets
    train_ds, val_ds, augment_pipe = build_dataset(c.dataset_kwargs, config.augment)

    # Build data loader
    c.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True,
        num_workers=config.workers,
        prefetch_factor=2,
        batch_size=config["batch_size"],
    )

    train_ds_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        sampler=misc.InfiniteSampler(dataset=train_ds, seed=config.seed),
        **c.data_loader_kwargs,
    )
    val_ds_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        sampler=misc.InfiniteSampler(dataset=val_ds, seed=config.seed),
        **c.data_loader_kwargs,
    )

    # Dump config
    os.makedirs(config["run_dir"], exist_ok=True)
    with open(f"{config['run_dir']}/config.json", "w") as f:
        conf = {"config": config, **c}
        json.dump(conf, f)

    # # Build flow model
    # inn = build_model(
    #     (
    #         config["num_sigmas"],
    #         c.dataset_kwargs.resolution,
    #         c.dataset_kwargs.resolution,
    #     ),
    #     levels=config["levels"],
    #     num_blocks=config["num_blocks"],
    # )

    inn = PatchFlow(
        input_size=(
            config["num_sigmas"],
            c.dataset_kwargs.resolution // 2,
            c.dataset_kwargs.resolution // 2,
        ),
        num_blocks=config["num_blocks"],
    )

    # exit()

    flownet = ScoreFlow(
        inn,
        scorenet=scorenet,
        vectorize=False,
        use_fp16=config["fp16"],
        num_sigmas=config["num_sigmas"],
        sigma_min=config["sigma_min"],
        post_downsample=True,
    )

    model_stats = summary(
        flownet,
        input_size=(1, *train_ds.dataset.image_shape),
        verbose=1,
        depth=1,
    )

    # Train
    losses = train_msma_flow(
        flownet,
        train_ds_loader,
        val_ds_loader,
        augment_pipe=augment_pipe,
        # num_epochs=config["num_epochs"],
        kimg=config["kimg"],
        device=device,
        run_dir=config["run_dir"],
        log_tensorboard=True,
        patchflow=True,
    )

    df = pd.DataFrame(losses, columns=["train_loss"])
    df["ema"] = df.train_loss.ewm(alpha=0.5).mean()
    df.to_csv(f"{config['run_dir']}/losses.csv", index=False)


if __name__ == "__main__":
    main()
