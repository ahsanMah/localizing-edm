import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


def subnet_fc(c_in, c_out, ndim=128):
    return nn.Sequential(
        nn.Linear(c_in, ndim),
        nn.GELU(),
        nn.Linear(ndim, ndim),
        nn.GELU(),
        nn.Linear(ndim, c_out),
    )


@torch.inference_mode()
def log_density(inn, x) -> torch.Tensor:
    inn.eval()
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
    for k in range(4):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)


def train_flow(inn, dataset_iterator, num_epochs=100, lr=3e-4, device="cuda"):
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
