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
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x, vectorize=False, **score_kwargs):
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
    
    def score_anomaly(self, x):
        return -self.log_density(x)

def load_pretrained_model(network_pkl):
    with dnnlib.util.open_url(network_pkl) as f:
        with torch.no_grad():
            model = pickle.load(f)

    config = dnnlib.EasyDict(
        augment_pipe=dnnlib.EasyDict(**model["augment_pipe"].init_kwargs),
        dataset_kwargs=dnnlib.EasyDict(**model["dataset_kwargs"]),
    )
    net = model["ema"]
    return net, config


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
        nn.ReLU(),
        nn.Conv2d(ndim, c_out, 3, padding=1),
    )


def subnet_conv_1x1(c_in, c_out, ndim=256):
    return nn.Sequential(nn.Conv2d(c_in, ndim, 1), nn.GELU(), nn.Conv2d(ndim, c_out, 1))


def build_model(input_dims, num_filters=256, levels=2, num_blocks=2, num_fc_blocks=0):
    nodes = [Ff.InputNode(*input_dims, name="input")]

    # Higher resolution convolutional part
    for k in range(num_blocks):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GLOWCouplingBlock,
                {"subnet_constructor": subnet_conv, "clamp": 1.2},
                name=f"conv_high_res_{k}",
            )
        )
        nodes.append(
            Ff.Node(
                nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_high_res_{k}"
            )
        )

    # Lower resolution convolutional part
    levels -= 1
    for i in range(levels):
        nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

        for k in range(num_blocks):
            if k % 2 == 0:
                subnet = partial(subnet_conv_1x1, ndim=num_filters)
            else:
                subnet = partial(subnet_conv, ndim=num_filters)

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {"subnet_constructor": subnet, "clamp": 1.2},
                    name=f"conv_level_{i+1}_res_{k}",
                )
            )
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.PermuteRandom,
                    {"seed": k},
                    name=f"permute_level_{i+1}_res_{k}",
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
    ll = -0.5 * torch.sum(z**2, dim=1) + log_jac_det
    nll = -torch.mean(ll)
    return nll


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


def train_msma_flow(
    flownet,
    train_ds,
    val_ds,
    augment_pipe=None,
    num_epochs=100,
    lr=3e-4,
    device=torch.device("cuda"),
    run_dir="runs/",
    log_interval=5,
    checkpoint_interval=20,
    log_tensorboard=False,
):
    losses = []
    opt = torch.optim.Adam(flownet.flow.parameters(), lr=lr)
    progbar = tqdm(range(num_epochs))
    num_batches = len(train_ds)

    checkpoint_path = f"{run_dir}/checkpoint.pth"

    if log_tensorboard:
        writer = SummaryWriter(log_dir=run_dir)

    flownet = flownet.to(device)
    flownet.train()

    niter = 0
    for epoch in progbar:
        for x, _ in train_ds:
            opt.zero_grad(set_to_none=True)

            x = x.to(device).to(torch.float32) / 127.5 - 1
            if augment_pipe:
                x, _ = augment_pipe(x)
            z, log_jac_det = flownet(x)
            loss = torch.mean(logloss(z, log_jac_det))
            loss.backward()
            opt.step()

            if niter % log_interval == 0:
                flownet.eval()

                with torch.no_grad():
                    val_loss = 0.0
                    for i, (x, _) in enumerate(val_ds):
                        x = x.to(device).to(torch.float32) / 127.5 - 1
                        z, log_jac_det = flownet(x)
                        val_loss += torch.mean(logloss(z, log_jac_det))

                val_loss /= i + 1
                progbar.set_description(f"Val Loss: {val_loss.item():.4f}")
                # progbar.set_postfix(val_loss=f"{loss.item():.4f}")

                flownet.train()

                if log_tensorboard:
                    writer.add_scalar("train_loss", loss.item(), niter)
                    writer.add_scalar("val_loss", val_loss.item(), niter)

                losses.append(val_loss.item())

            niter += 1
            progbar.set_postfix(batch=f"{niter}/{num_batches}")

        if niter % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": flownet.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss.item(),
                },
                checkpoint_path,
            )

        progbar.set_description(f"Loss: {loss.item():.4f}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": flownet.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": val_loss.item(),
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
        xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
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
# Optimization options
@click.option("--num_epochs", default=10, help="Number of epochs to train.")
@click.option("--batch_size", default=128, help="Batch size.")
@click.option("--lr", default=3e-4, help="Learning rate.")
@click.option("--fp16", default=False, help="Use fp16 (applies to scorenet only).")
@click.option("--workers", default=4, help="Number of workers.")
@click.option("--device", default="cuda", help="Device to use.")
def main(network_pkl, **kwargs):
    # Load network alongside with its training config.
    print('Loading network from "%s"' % network_pkl)
    scorenet, model_config = load_pretrained_model(network_pkl=network_pkl)
    print("Loaded network")

    config = dnnlib.EasyDict(kwargs)

    # opts = dnnlib.EasyDict(**config.dataset_kwargs)
    device = torch.device(config.device)

    c = dnnlib.EasyDict(**model_config)

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
        dataset=train_ds, **c.data_loader_kwargs
    )
    val_ds_loader = torch.utils.data.DataLoader(dataset=val_ds, **c.data_loader_kwargs)

    # Dump config
    os.makedirs(config["run_dir"], exist_ok=True)
    with open(f"{config['run_dir']}/config.json", "w") as f:
        conf = {"config": config, **c}
        json.dump(conf, f)

    # Build flow model
    conv_inn = build_model(
        (
            config["num_sigmas"],
            c.dataset_kwargs.resolution,
            c.dataset_kwargs.resolution,
        ),
        levels=config["levels"],
        num_blocks=config["num_blocks"],
    )

    flownet = ScoreFlow(
        conv_inn,
        scorenet=scorenet,
        vectorize=False,
        use_fp16=config["fp16"],
        num_sigmas=config["num_sigmas"],
        sigma_min=config["sigma_min"],
    )

    model_stats = summary(
        flownet, input_size=(1, *train_ds.dataset.image_shape), verbose=1
    )

    # Train
    losses = train_msma_flow(
        flownet,
        train_ds_loader,
        val_ds_loader,
        augment_pipe=augment_pipe,
        num_epochs=config["num_epochs"],
        device=device,
        run_dir=config["run_dir"],
        log_tensorboard=True,
    )

    df = pd.DataFrame(losses, columns=["train_loss"])
    df["ema"] = df.train_loss.ewm(alpha=0.5).mean()
    df.to_csv(f"{config['run_dir']}/losses.csv", index=False)


if __name__ == "__main__":
    main()
