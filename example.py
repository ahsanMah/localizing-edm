# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import click

# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_image_grid(
    network_pkl,
    dest_path,
    img_resolution=None,
    seed=0,
    grid=3,
    device=torch.device("cuda"),
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):  
    gridw = gridh = grid
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device)

    if img_resolution is not None:
        net.img_resolution = img_resolution
   
    # Pick latents and labels.
    print(f"Generating {batch_size} images of size {net.img_resolution}...")

    latents = torch.randn(
        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm.tqdm(
        list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit="step"
    ):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(
            x_cur
        )

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(
        gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels
    )
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, "RGB").save(dest_path)
    print("Done.")


# ----------------------------------------------------------------------------
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
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    default=None,
)
@click.option(
    "--res",
    help="Resoluton of images to generate",
    metavar="INT",
    type=int,
    default=None,
)
@click.option(
    "--steps",
    help="Number of diffusion steps",
    metavar="INT",
    type=int,
    default=28,
)
def main(network_pkl, outdir, res, steps):
    # model_root = '/workspace/localizing-edm/workdir/pretrained_models'
    # generate_image_grid(f'{model_root}/edm-cifar10-32x32-uncond-ve.pkl',   'cifar10-32x32.png',  num_steps=18) # FID = 1.79, NFE = 35

    # model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    # generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-vp.pkl',   'cifar10-32x32.png',  num_steps=18) # FID = 1.79, NFE = 35
    # generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=40) # FID = 2.39, NFE = 79
    # generate_image_grid(f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl',  'afhqv2-64x64.png',   num_steps=40) # FID = 1.96, NFE = 79
    # generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=256, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003) # FID = 1.36, NFE = 511

    # model_name = network_pkl.split("/")[-1][:-4]
    model_root = network_pkl.split("/")
    model_name = model_root[-1][:-4]
    
    if outdir is None:
        outdir = "/".join(model_root[:-1])

    generate_image_grid(
        network_pkl, f"{outdir}/{model_name}_steps={steps}.png", num_steps=steps, img_resolution=res,
#        S_churn=40, S_min=0.05, S_max=50, S_noise=1.0030
    )  # FID = 1.36, NFE = 511
    # generate_image_grid(network_pkl, f"{outdir}/{model_name}.png", num_steps=128, S_churn=40, S_min=0.05, S_max=50, S_noise=1.0030) # FID = 1.36, NFE = 511


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
