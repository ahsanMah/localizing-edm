import math
from functools import partial

import cv2 as cv
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils


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

def plot_batch_with_heatmaps(images_batch, anomaly_maps, clip_val=None, grid_size=(5, 5), blur_radius=5,
                             alpha=0.5, symmetric_norm=True, linthresh_q=0.9):
    '''

    Anomaly maps are expected to be anomaly scores i.e. higher is more anomalous
    Make sure if passing in likelihoods, the maps should be inverted appropriately
    '''
    
    # Not considering log probabilities higher than threshold
    if clip_val is not None:
        anomaly_maps = anomaly_maps.clip(min=clip_val)
    
    anomaly_maps = anomaly_maps.squeeze(1)
    
    # Putting batch as "channel" dimension
    # so kernel is run over all batches (independently)
    anomaly_maps = anomaly_maps.transpose(1,2,0)
    anomaly_maps = cv.blur(anomaly_maps, ksize=(blur_radius,blur_radius))
    anomaly_maps = anomaly_maps.transpose(2,0,1)
    
    # Min-Max Scaling
    amin = anomaly_maps.min(axis=(1,2), keepdims=True)
    amax = anomaly_maps.max(axis=(1,2), keepdims=True) 
    anomaly_maps = (anomaly_maps - amin) / (amax-amin)
    
    scaled_heatmaps = []
    for anomap in anomaly_maps:

        if symmetric_norm:
            norm=colors.SymLogNorm(linthresh=np.quantile(anomap,linthresh_q),
                                   vmin=np.quantile(anomap,0.01),
                                   vmax=np.quantile(anomap,0.99))
        shape = anomap.shape
        scaled_heatmaps.append(norm(anomap.ravel()).reshape(*shape))
    
    
#     if symmetric_norm:
#         norm=colors.SymLogNorm(linthresh=np.quantile(anomaly_maps,linthresh_q),
#                                vmin=np.quantile(anomaly_maps,0.01),
#                                vmax=np.quantile(anomaly_maps,0.99))
#     else:
#         norm=colors.LogNorm(vmin=np.quantile(anomaly_maps,0.01),
#                             vmax=np.quantile(anomaly_maps,0.99))
#     shape = anomaly_maps.shape
#     scaled_heatmaps  = norm(anomaly_maps.ravel()).reshape(*shape)
    
    # Expand anomaly_maps to 3 color channels and apply colormap
    anomaly_maps_rgb = plt.get_cmap('jet')(scaled_heatmaps)[:,:,:,:3]
    anomaly_maps_rgb = torch.from_numpy(anomaly_maps_rgb).float()
    anomaly_maps_rgb = anomaly_maps_rgb.permute(0,3,1,2)
    
    # Merge images with anomaly maps with given alpha
    overlaid_images = alpha * images_batch + (1-alpha) * anomaly_maps_rgb

    # Convert batch of tensors to grid
    grid = vutils.make_grid(overlaid_images, nrow=grid_size[1], padding=2, normalize=True)
    
    # Plot grid
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(np.transpose(grid.numpy(),(1,2,0)))
    
    return anomaly_maps_rgb