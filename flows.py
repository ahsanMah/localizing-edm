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


class ScoreFlow(torch.nn.Module):
    def __init__(self, flow, scorenet, num_steps, vectorize=False, use_fp16=False):
        super().__init__()
        self.flow = flow
        self.scorenet = VEScorer(
            scorenet, num_steps=num_steps, use_fp16=use_fp16
        ).requires_grad_(False)

        if vectorize:
            self.fastscore = self.scorenet.build_vectorized_forward()
            # self.fastscore = torch.jit.script(self.fastscore)

    def forward(self, x, vectorize=False, **score_kwargs):
        if vectorize:
            x_scores = self.fastscore(x)
        else:
            x_scores = self.scorenet(x, **score_kwargs)

        return self.flow(x_scores.contiguous())


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
        nn.GELU(),
        nn.Conv2d(ndim, c_out, 3, padding=1),
    )


def subnet_conv_1x1(c_in, c_out, ndim=256):
    return nn.Sequential(nn.Conv2d(c_in, ndim, 1), nn.GELU(), nn.Conv2d(ndim, c_out, 1))


def build_model(
    input_dims, num_filters=256, num_hres_blocks=4, num_lres_blocks=4, num_fc_blocks=0
):
    nodes = [Ff.InputNode(*input_dims, name="input")]

    # Higher resolution convolutional part
    for k in range(num_hres_blocks):
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

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    # Lower resolution convolutional part
    for k in range(num_lres_blocks):
        if k % 2 == 0:
            subnet = partial(subnet_conv_1x1, ndim=num_filters)
        else:
            subnet = partial(subnet_conv, ndim=num_filters)

        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GLOWCouplingBlock,
                {"subnet_constructor": subnet, "clamp": 1.2},
                name=f"conv_low_res_{k}",
            )
        )
        nodes.append(
            Ff.Node(
                nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_low_res_{k}"
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

    # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetUpsampling, {}))

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    conv_inn = Ff.GraphINN(nodes)

    return conv_inn


@torch.inference_mode()
def log_density(inn, x) -> torch.Tensor:
    z, log_jac_det = inn(x)
    logpz = -0.5 * torch.sum(z**2, dim=1, keepdim=True) - log_jac_det.reshape(
        z.shape[0], *[1] * len(z.shape[1:])
    )
    return logpz


@torch.inference_mode()
def per_dim_ll(inn, x):
    inn.eval()
    z, log_jac_det = inn(x)
    z = z.cpu().numpy()
    log_jac_det = log_jac_det.cpu().numpy()
    logpz = -0.5 * z**2 - log_jac_det.reshape(-1, 1)
    return logpz


# @torch.jit.script
def logloss(z, log_jac_det):
    bsz = z.shape[0]
    z = z.reshape(bsz, -1)
    log_jac_det = log_jac_det.reshape(bsz, -1)
    return 0.5 * torch.mean(z**2, dim=1) - log_jac_det


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
            loss = torch.mean(logloss(z, log_jac_det))
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
    num_epochs=100,
    lr=3e-4,
    device=torch.device("cuda"),
    run_dir="runs/",
    log_interval=5,
    checkpoint_interval=100,
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
            z, log_jac_det = flownet(x)
            loss = torch.mean(logloss(z, log_jac_det))
            loss.backward()
            opt.step()

            if niter % log_interval == 0:
                progbar.set_description(f"Loss: {loss.item():.4f}")

                val_loss = 0.0
                for x, _ in val_ds:
                    x = x.to(device).to(torch.float32) / 127.5 - 1
                    z, log_jac_det = flownet(x)
                    val_loss += torch.mean(logloss(z, log_jac_det))
                    progbar.set_postfix(val_loss=f"{loss.item():.4f}")

                if log_tensorboard:
                    writer.add_scalar("train_loss", loss.item(), niter)
                    writer.add_scalar("val_loss", val_loss.item(), niter)

                losses.append(val_loss.item())

            niter += 1
            # progbar.set_postfix(batch=f"{niter}/{num_batches}")

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

        if log_tensorboard:
            writer.close()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": flownet.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": val_loss.item(),
        },
        checkpoint_path,
    )

    return losses


def build_dataset(dataset_kwargs, val_ratio=0.1):
    # Build dataset
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    train_len = int((1 - val_ratio) * dataset_kwargs.max_size)
    val_len = dataset_kwargs.max_size - train_len
    train_ds = Subset(dataset_obj, range(train_len))

    # Subset of training dataset
    val_ds = Subset(dataset_obj, range(train_len, train_len + val_len))

    return train_ds, val_ds


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
@click.option("--resolution", default=None, type=int, help="Resolution of images to train on.")
# Flow options
@click.option("--num_sigmas", default=10, help="Number of sigmas in the flow.")
@click.option(
    "--num_hres_block", default=4, help="Number of high resolution blocks in the flow."
)
@click.option(
    "--num_lres_block", default=4, help="Number of low resolution blocks in the flow."
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
    train_ds, val_ds = build_dataset(c.dataset_kwargs)

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
        (config["num_sigmas"], c.dataset_kwargs.resolution, c.dataset_kwargs.resolution)
    )
    flownet = ScoreFlow(
        conv_inn,
        scorenet=scorenet,
        vectorize=False,
        use_fp16=config["fp16"],
        num_steps=config["num_sigmas"],
    )

    model_stats = summary(
        flownet, input_size=(1, *train_ds.dataset.image_shape), verbose=1
    )

    # Train
    losses = train_msma_flow(
        flownet,
        train_ds_loader,
        val_ds_loader,
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
