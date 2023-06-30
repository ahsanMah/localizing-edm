import json
import math
import os
import pickle
from functools import partial

import click
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm.auto import tqdm

import dnnlib
from flow_utils import (
    build_conv_model,
    positionalencoding2d,
    subnet_fc,
    SpatialNorm2D,
    ScoreAttentionBlock,
)
from scorer import EDMScorer, VEScorer
from torch_utils import misc
import pdb

_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


def logprob(z, ldj):
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


class PatchFlow(torch.nn.Module):
    """
    Contructs a conditional flow model that operates on patches of an image.
    Each patch is fed into the same flow model i.e. parameters are shared across patches.
    The flow models are conditioned on a positional encoding of the patch location.
    The resulting patch-densities can then be recombined into a full image density.
    """

    # Opinionated parameters for computing the spatial norms of the input scores
    patch_config = {
        3: {
            "local": {
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
            },
            "global": {"kernel_size": 3, "padding": 1, "stride": 1},  # no downsampling
        },
        7: {
            "local": {
                "kernel_size": 7,
                "padding": 2,
                "stride": 4,
            },  # factor 4 downsampling
            "global": {"kernel_size": 5, "padding": 2, "stride": 2},  # factor 2
        },
        # # factor 4 downsampling
        # "local": {"kernel_size": 7, "padding": 3, "stride": 2},
        # 11: {"kernel_size": 11, "padding": 3, "stride": 2},
        # # factor 8 downsampling
        # 15: {"kernel_size": 15, "padding": 2, "stride": 7},  # factor 8 downsampling
    }

    def __init__(
        self,
        input_size,
        patch_size=3,
        spatial_cond_dim=128,
        num_blocks=2,
        global_flow=False,
        patch_batch_size=128,
    ):
        super().__init__()
        assert (
            patch_size in PatchFlow.patch_config
        ), f"PatchFlow only support certain patch sizes: {PatchFlow.patch_config.keys()}"
        channels = input_size[0]
        
        # Used to chunk the input into in fast_forward (vectorized) 
        self.patch_batch_size = patch_batch_size

        self.use_global_context = global_flow


        with torch.no_grad():
            # Pooling for global "high resolution" flow
            # self.global_pooler = SpatialNorm2D(
            #     channels,  # **PatchFlow.patch_config[3]["global"]
            # ).requires_grad_(False)

            # Pooling for local "patch" flow
            # Each patch-norm is input to the shared conditional flow model
            self.local_pooler = SpatialNorm2D(
                channels, **PatchFlow.patch_config[7]["local"]
            ).requires_grad_(False)

            # Compute the spatial resolution of the patches
            _, c, h, w = self.local_pooler(torch.empty(1, *input_size)).shape
            self.spatial_res = (h, w)
            self.num_patches = h * w
            low_res_channels = self.channels = c
            pos = positionalencoding2d(spatial_cond_dim, h, w)
            pos = pos.reshape(spatial_cond_dim, self.num_patches)
            pos = pos.permute(1, 0)  # N x C_d
            self.register_buffer("positional_encoding", pos)
            print(
                f"Generating {patch_size}x{patch_size} patches from input size: {input_size}"
            )
            print(f"Pooled spatial resolution: {self.spatial_res}")
            print(f"Number of flows / patches: {self.num_patches}")

        # _b, c, h, w = self.global_pooler(torch.empty(1, *input_size)).shape
        # self.global_flow = nn.Sequential(
        #     global_pooler,
        #     build_conv_model((c, h, w), levels=2, num_blocks=num_blocks // 2, unet=False),
        # )

        # # Output of high res flow is downsampled to match the spatial resolution of the patches
        # # Downsampling redistributes the spatial dims across the channels
        # # 2x downsampling -> 4x channels
        # global_cond_dim = channels * 4

        cond_dims = spatial_cond_dim

        if self.use_global_context:
            # self.global_flow = build_conv_model(
            #     input_size, levels=2, num_blocks=[num_blocks//2, num_blocks], unet=False
            # )
            # self.global_flow = torch.nn.Sequential(
            #     nn.AdaptiveAvgPool2d((h*2, w*2)),
            #     self.global_flow,
            # )
            # global_cond_dim = channels
            # self.global_pooler = nn.AdaptiveAvgPool2d((h//2, w//2))
            self.global_attention = ScoreAttentionBlock(
                input_size=(c, h, w), embed_dim=cond_dims
            )
            cond_dims += cond_dims

        num_features = low_res_channels  # + global_cond_dim
        self.flow = self.build_cflow_head(cond_dims, num_features, num_blocks)

    def init_weights(self):
        # Initialize weights with Xavier
        for m in self.flow.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        if self.use_global_context:
            for m in self.global_attention.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # print(m)
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

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
                affine_clamping=1.9,
            )

        return coder

    def patchify(self, x):
        # print(x.shape)
        b, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)  # Patches x batch x channels
        x = x.reshape(self.num_patches, b, c)
        return x

    def forward(self, x, return_attn=False, fast=True):
        B = x.shape[0]
        x = self.local_pooler(x)
        local_patches = self.patchify(x)  # Patches x batch x channels
        context = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)

        mask = torch.eye(
            self.num_patches, dtype=torch.bool, device=x.device, requires_grad=False
        )

        global_context, attn_maps = self.global_attention(local_patches, mask)
        context = torch.cat([context, global_context], dim=-1)

        if fast:
            zs, log_jac_dets = self.fast_forward(local_patches, context)

        else:
            zs = []
            log_jac_dets = []

            for patch_feature, context_vector in zip(local_patches, context):
                # print(patch_feature.shape, glb_ctx.shape, spatial_cond.shape)
                z, ldj = self.flow(
                    patch_feature,
                    c=[context_vector],
                    # torch.cat([patch_feature, glb_ctx], dim=-1),
                    # c=[spatial_cond],
                )
                zs.append(z)
                log_jac_dets.append(ldj)

        if return_attn:
            return zs, log_jac_dets, attn_maps

        return zs, log_jac_dets

    def fast_forward(self, x, ctx):

        assert self.num_patches % self.patch_batch_size == 0, "Need patch batch size to be disible by total number of patches"
        nchunks = self.num_patches // self.patch_batch_size
        P,B,C = x.shape
        _,_,D = ctx.shape
        x, ctx = x.reshape(P*B, C), ctx.reshape(P*B, D)
        x, ctx = x.chunk(nchunks, dim=0), ctx.chunk(nchunks, dim=0)
        zs, jacs, = [], []
        
        for p, c in zip(x, ctx):
            # assert torch.isclose(c[0, :32], c[B-1, :32]).all() # Check that patch context is same for all batch elements
            # assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            z, ldj = self.flow(
                p,
                c=[c],
            )
            zs.append(z)
            jacs.append(ldj)
        
        zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        jacs = torch.cat(jacs, dim=0).reshape(self.num_patches, B)

        return zs, jacs

    def stochastic_train_step(self, x, opt, n_patches=1):
        B = x.shape[0]
        x = self.local_pooler(x)
        local_patches = self.patchify(x)  # Patches x batch x channels
        context = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)

        G = self.global_attention
        # global_context =
        # global_context, attn_maps = self.global_attention(local_patches)
        # context = torch.cat([context, global_context], dim=-1)
        mask = torch.eye(
            self.num_patches, dtype=torch.bool, device=x.device, requires_grad=False
        )

        rand_idx = torch.randperm(self.num_patches)[:n_patches]
        local_loss = 0.0
        global_loss = 0.0
        for idx in rand_idx:
            patch_feature, context_vector = (
                local_patches[idx],
                context[idx],
            )

            # Need separate loss for each patch
            glb_ctx = G.proj(local_patches.permute(1, 0, 2))
            glb_ctx = G.norm(glb_ctx.add_(G.positional_encoding))
            # print("GLOBAL CTX:",glb_ctx.shape)
            p = glb_ctx[:, [idx], :]
            p_ctx, attn_maps = G.attention(p, glb_ctx, glb_ctx, attn_mask=mask[idx].unsqueeze(0))
            p_ctx = G.normout(G.ffn(p + p_ctx))
            p_ctx = p_ctx.squeeze(1)
            # print(p_ctx.shape, attn_maps.shape)
            context_vector = torch.cat([context_vector, p_ctx], dim=-1)

            z, ldj = self.flow(
                patch_feature,
                c=[context_vector],
            )

            opt.zero_grad(set_to_none=True)
            loss = nll(z, ldj)
            loss.backward()
            # name, p = G.named_parameters().__next__()
            # print(name, p.grad.mean())
            opt.step()
            local_loss += loss.item()
            global_loss += attn_maps.mean().item()

        return global_loss, local_loss

    # def forward(self, x):
    #     B = x.shape[0]

    #     # x = self.local_pooler(x)
    #     local_patches = self.patchify(self.local_pooler(x))  # Patches x batch x channels

    #     zs = []
    #     log_jac_dets = []
    #     context = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)

    #     if self.use_global_context:
    #         hres_zs, hres_log_jac_dets = self.global_flow(x)
    #         global_context_patches = self.patchify(hres_zs)
    #         context = torch.cat([context, global_context_patches], dim=-1)
    #         zs.append(hres_zs.reshape(B, -1))
    #         log_jac_dets.append(hres_log_jac_dets)

    #     for patch_feature, context_vector in zip(local_patches,context,):
    #         z, ldj = self.flow(
    #             patch_feature,
    #             c=[context_vector],
    #         )
    #         zs.append(z)
    #         log_jac_dets.append(ldj)

    #     return zs, log_jac_dets

    # def stochastic_train_step(self, x, opt, n_patches=1):
    #     B = x.shape[0]

    #     # x = self.local_pooler(x)
    #     # print(x)
    #     # print("hres:", self.global_pooler(x))
    #     local_patches = self.patchify(self.local_pooler(x))  # Patches x batch x channels

    #     # zs = []
    #     # log_jac_dets = []
    #     context = self.positional_encoding.unsqueeze(1).repeat(1, B, 1)

    #     global_loss = 0.0
    #     if self.use_global_context:
    #         # print("using global context - before:", context.shape)
    #         # Likeilihood step for global flow
    #         hres_zs, hres_log_jac_dets = self.global_flow(x)
    #         opt.zero_grad(set_to_none=True)
    #         loss = nll(hres_zs.reshape(B,-1), hres_log_jac_dets) / n_patches
    #         loss.backward()
    #         name,p = self.global_flow.named_parameters().__next__()
    #         print(name, p.grad.mean())
    #         opt.step()
    #         global_loss += loss.item()

    #         # Global flow will be used in context loss
    #         hres_zs, hres_log_jac_dets = self.global_flow(x)
    #         global_context_patches = self.patchify(hres_zs)
    #         context = torch.cat([context, global_context_patches], dim=-1)
    #         # print("using global context:", context.shape, global_context_patches.shape)
    #         # zs.append(hres_zs.reshape(B, -1))
    #         # log_jac_dets.append(hres_log_jac_dets)
    #         # print(context.shape)

    #     local_loss = 0.0
    #     rand_idx = torch.randperm(self.num_patches)[:n_patches]
    #     for idx in rand_idx:
    #         patch_feature, context_vector = (
    #             local_patches[idx],
    #             context[idx],
    #         )
    #         z, ldj = self.flow(
    #             patch_feature,
    #             c=[context_vector],
    #         )
    #         # zs.append(z)
    #         # log_jac_dets.append(ldj)

    #         opt.zero_grad(set_to_none=True)
    #         loss = nll(z, ldj)
    #         loss.backward(retain_graph=True)
    #         name,p = self.global_flow.named_parameters().__next__()
    #         print(name, p.grad.mean())
    #         opt.step()
    #         local_loss += loss.item()

    #     return global_loss, local_loss

    def reconstruct(self, logpxs):
        """
        Reconstructs image from a list of patch densities
        """
        # if self.use_global_context:
        #     hres_logpx = zs.pop(0)[:, None, None]

        # x is num_patches x batch
        if isinstance(logpxs, list):
            x = torch.stack(logpxs)
        else:
            x = logpxs
        # Swap batch and patch dim
        x = x.permute(1, 0)
        x = x.reshape(x.shape[0], *self.spatial_res)

        # if self.use_global_context:
        #     x = x + hres_logpx / self.num_patches

        return x

    @torch.no_grad()
    def log_density(self, x, fast=True):
        self.eval()
        zs, jacs = self.forward(x, fast=fast)

        if isinstance(zs, list):
            zs = torch.stack(zs)
            jacs = torch.stack(jacs)
        assert zs.dim() == 3
        # if self.use_global_context:
        #     hres_logpx = zs.pop(0)
        #     hres_jac = jacs.pop(0)[:, None, None]
        #     hres_logpx = hres_logpx.reshape(
        #         x.shape[0], self.channels, *self.spatial_res
        #     )
        #     hres_logpx = hres_logpx.sum(dim=1)
        #     hres_logpx = hres_logpx + hres_jac / self.num_patches

        # logpzs = [-0.5 * torch.sum(z**2, dim=1) + ldj for z, ldj in zip(zs, jacs)]
        logpx = logprob(zs, jacs)

        # x is num_patches x batch
        # logpx = torch.stack(logpzs)
        # Swap batch and patch dim
        logpx = logpx.permute(1, 0)
        logpx = logpx.reshape(x.shape[0], *self.spatial_res)

        # if self.use_global_context:
        #     logpx = logpx + hres_logpx  # / self.num_patches

        return logpx


class ScoreFlow(torch.nn.Module):
    def __init__(
        self,
        flow,
        scorenet,
        vectorize=False,
        fastflow=False,
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

        if fastflow:
            # Initialize weights with Xavier
            for m in self.flow.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # print(m)
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)
        else:
            self.flow.init_weights()

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
    def score_patches(self,x, fast=True):
        return -self.flow.log_density(self.fastscore(x), fast=fast)


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


def patchflow_stochastic_step(flownet, x, opt, n_samples=128):
    n_samples = min(n_samples, flownet.flow.num_patches)
    with torch.no_grad():
        scores = flownet.fastscore(x, chunks=10)
    # print("SCORES", scores)
    # Optimizing the highres flow
    # Note this could be combined with the patch loss below
    # But dual optimizer step is more stable
    # opt.zero_grad(set_to_none=True)
    # hres_zs, hres_log_jac_dets = flownet.flow.global_flow(scores)
    # hres_loss = nll(hres_zs.reshape(x.shape[0], -1), hres_log_jac_dets)
    # hres_loss = hres_loss / n_samples
    # hres_loss.backward()
    # opt.step()

    # hres_loss = 0.0
    # total_loss = 0.0  # hres_loss.item()
    # # Optimizing the patch flow
    # zs, log_jac_dets = flownet.flow.stochastic_train_step(scores, n_patches=n_samples)
    # patch_losses = [nll(z, ldj) for z, ldj in zip(zs, log_jac_dets)]
    # pdb.set_trace()
    # hres_loss = patch_losses[0]
    # losses = 0.0
    # opt.zero_grad(set_to_none=True)
    # for loss in patch_losses:
    #     losses = losses + loss
    # loss.backward()
    # opt.step()

    global_loss, local_loss = flownet.flow.stochastic_train_step(
        scores, opt, n_patches=n_samples
    )
    total_loss = global_loss + local_loss

    return {
        "loss": total_loss,
        "global_loss": global_loss,
        "local_loss": local_loss,
    }


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
    losses = 0.0
    for loss in patch_losses:
        losses += loss
    losses.backward()
    opt.step()

    return losses.item()


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
    fastflow=False,
    patch_batch_size=32,
):
    losses = []
    opt = torch.optim.AdamW(flownet.flow.parameters(), lr=lr, weight_decay=1e-5)
    batch_sz = train_ds.batch_size
    total_iters = kimg * 1000 // batch_sz + 1
    progbar = tqdm(range(total_iters))
    train_step = (
        fastflow_step
        if fastflow
        else partial(patchflow_stochastic_step, n_samples=patch_batch_size)
    )
    checkpoint_path = f"{run_dir}/checkpoint.pth"
    checkpoint_meta_path = f"{run_dir}/checkpoint-meta.pth"

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
                writer.add_scalar(
                    f"train_loss/{loss_type}", loss_dict[loss_type], niter
                )

        if niter % log_interval == 0:
            flownet.eval()

            with torch.no_grad():
                val_loss = 0.0
                x, _ = next(val_iter)
                x = x.to(device).to(torch.float32) / 127.5 - 1
                z, log_jac_det = flownet(x, vectorize=True)
                if fastflow:
                    val_loss += logloss(z, log_jac_det).item()
                else:
                    # print(z.shape, log_jac_det.shape)
                    # pdb.set_trace()
                    val_loss = -logprob(z, log_jac_det).mean(-1).sum().item()
                    # for z, ldj in zip(z, log_jac_det):
                    #     val_loss += nll(z, ldj).item()

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
                checkpoint_meta_path,
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
@click.option("--augment", default=0.05, help="Probability of augmentation.")
# Flow options
@click.option(
    "--fastflow",
    default=False,
    type=bool,
    help="Whether to use a fully convolutional FastFlow-like model.",
)
@click.option(
    "--global_flow",
    default=False,
    type=bool,
    help="Additionaly condition using a convolutional flow on full image.",
)
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
@click.option(
    "--patch_batch_size",
    default=32,
    help="Number of patches included in one training step of patch-based model.",
)
@click.option("--lr", default=3e-4, help="Learning rate.")
@click.option("--fp16", default=False, help="Use fp16 (applies to scorenet only).")
@click.option("--workers", default=4, help="Number of workers.")
@click.option("--device", default="cuda", help="Device to use.")
@click.option("--seed", default=42, type=int, help="Device to use.")
def main(network_pkl, **kwargs):
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # Load network alongside with its training config.
    print('Loading network from "%s"' % network_pkl)
    scorenet, model_config = load_pretrained_model(network_pkl=network_pkl)
    print("Loaded network")

    config = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict(**model_config)
    device = torch.device(config.device)

    hparams = f"nb{config.num_blocks}-lr{config.lr}-bs{config.batch_size}-pbs{config.patch_batch_size}-kimg{config.kimg}-augp{config.augment}"
    config["run_dir"] = f"{config.run_dir}/{hparams}"

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

    if config["fastflow"]:
        inn = build_conv_model(
            (
                config["num_sigmas"],
                c.dataset_kwargs.resolution,
                c.dataset_kwargs.resolution,
            ),
            num_blocks=[config["num_blocks"] // 2, config["num_blocks"]],
        )
    else:
        inn = PatchFlow(
            input_size=(
                config["num_sigmas"],
                c.dataset_kwargs.resolution,
                c.dataset_kwargs.resolution,
            ),
            num_blocks=config["num_blocks"],
            patch_size=7,
            global_flow=config["global_flow"],
            patch_batch_size=config["patch_batch_size"],
        )

    # exit()

    flownet = ScoreFlow(
        inn,
        scorenet=scorenet,
        vectorize=True,
        use_fp16=config["fp16"],
        num_sigmas=config["num_sigmas"],
        sigma_min=config["sigma_min"],
        fastflow=config["fastflow"],
    )

    model_stats = summary(
        flownet,
        input_size=(5, *train_ds.dataset.image_shape),
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
        fastflow=config["fastflow"],
        patch_batch_size=config["patch_batch_size"],
    )

    df = pd.DataFrame(losses, columns=["val_loss"])
    df["ema"] = df.val_loss.ewm(alpha=0.5).mean()
    df.to_csv(f"{config['run_dir']}/losses.csv", index=False)


if __name__ == "__main__":
    main()
