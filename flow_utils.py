import math
from functools import partial

import cv2 as cv
# import FrEIA.framework as Ff
# import FrEIA.modules as Fm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib as mpl
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

import normflows as nf
from normflows.distributions.base import BaseDistribution


class ConditionalGaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(self, n_modes, dim, context_encoder):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          scale: List of diagonals of the covariance matrices
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim
        self.context_encoder = context_encoder
        self.split_sizes = [
            self.n_modes,
            self.n_modes * self.dim,
            self.n_modes * self.dim,
        ]

    def forward(self, num_samples=1, context=None):
        encoder_output = self.context_encoder(context)
        w, loc, log_scale = torch.split(encoder_output, self.split_sizes, dim=1)
        loc = loc.reshape(loc.shape[0], self.n_modes, self.dim)
        log_scale = log_scale.reshape(loc.shape[0], self.n_modes, self.dim)

        # Get weights
        weights = torch.softmax(w, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(num_samples, self.dim, dtype=loc.dtype, device=loc.device)
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z, context=None):
        encoder_output = self.context_encoder(context)
        w, loc, log_scale = torch.split(encoder_output, self.split_sizes, dim=1)
        loc = loc.reshape(loc.shape[0], self.n_modes, self.dim)
        log_scale = log_scale.reshape(loc.shape[0], self.n_modes, self.dim)

        # Get weights
        weights = torch.softmax(w, 1)

        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return log_p


def build_flow_head(
    latent_size,
    context_size,
    hidden_units=256,
    hidden_layers=2,
    num_blocks=2,
    num_mixtures=5,
):
    K, D = num_mixtures, latent_size

    gmm_dims = K + (K * D) + (K * D)  # Diagonal Gaussians: weights, mu, scales
    gmm_context_encoder = torch.nn.Sequential(
        torch.nn.Linear(context_size, context_size * 2, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(context_size * 2, context_size * 2, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(context_size * 2, gmm_dims, bias=False),
    )

    flow_blocks = []

    for _ in range(num_blocks):
        flow_blocks += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size,
                hidden_layers,
                hidden_units,
                num_context_channels=context_size,
                activation=nn.GELU,
            )
        ]
        flow_blocks += [nf.flows.LULinearPermute(latent_size)]

    base = ConditionalGaussianMixture(K, D, gmm_context_encoder)

    flow = nf.ConditionalNormalizingFlow(base, flow_blocks)

    return flow


class ScoreAttentionBlock(nn.Module):
    def __init__(self, input_size, embed_dim, outdim=None, num_heads=4, dropout=0.1):
        super().__init__()
        num_sigmas, h, w = input_size
        outdim = outdim or embed_dim

        self.proj = nn.Linear(num_sigmas, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        pos = positionalencoding2d(embed_dim, h, w)
        pos = pos.reshape(1, embed_dim, h * w).permute(0, 2, 1)  # (B) N x C_d
        self.register_buffer("positional_encoding", pos)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, outdim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outdim, outdim, bias=False),
        )
        self.normout = nn.LayerNorm(outdim)

    def forward(self, x, attn_mask=None):
        # Making batch first to trigger flash attention
        x = x.permute(1, 0, 2)
        assert (
            x.shape[1] == self.positional_encoding.shape[1]
        ), "Can only be called on flattened image"

        x = self.proj(x)
        # x = x @ self.proj
        x = self.norm(x.add_(self.positional_encoding))

        h, attn = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
        )

        x = self.normout(self.ffn(x + h))

        x = x.permute(1, 0, 2)
        return x, attn


class SpatialNorm2D(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv.weight.data.fill_(1)  # all ones weights
        self.conv.weight.requires_grad = False  # freeze weights

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x.square()).pow_(0.5)


# Taken from https://github.com/y0ast/Glow-PyTorch/blob/13c7b013dde32600e732416f2ed15438d686636f/modules.py#LL217C1-L244C68
class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        # self.logscale_factor = logscale_factor
        # self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output  # * torch.exp(self.logs * self.logscale_factor)


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def subnet_fc(c_in, c_out, ndim=256):
    return nn.Sequential(
        nn.Linear(c_in, ndim),
        nn.ReLU(),
        nn.Linear(ndim, ndim),
        nn.ReLU(),
        nn.Linear(ndim, c_out),
    )


def subnet_attention(c_in, c_out, ndim=256):
    pass


def subnet_conv(c_in, c_out, ndim=256):
    return nn.Sequential(
        nn.Conv2d(c_in, ndim, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(ndim, c_out, 3, padding=1),
        nn.ReLU(),
        Conv2dZeros(c_out, c_out),
    )


def subnet_conv_1x1(c_in, c_out, ndim=256):
    return nn.Sequential(nn.Conv2d(c_in, ndim, 1), nn.ReLU(), nn.Conv2d(ndim, c_out, 1))


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

    if isinstance(num_blocks, int):
        highres_blocks = lowres_blocks = num_blocks
    else:
        highres_blocks, lowres_blocks = num_blocks

    # Higher resolution convolutional part
    for k in range(highres_blocks):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.AllInOneBlock,
                {"subnet_constructor": subnet_conv},
                name=f"conv_high_res_{k}",
            )
        )

    # Lower resolution convolutional part
    levels -= 1
    for i in range(levels):
        nodes.append(
            Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name=f"downsample_{i}")
        )

        for k in range(lowres_blocks):
            if k % 2 == 0:
                subnet = partial(subnet_conv_1x1, ndim=num_filters)
            else:
                subnet = partial(subnet_conv, ndim=num_filters)

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet},
                    name=f"conv_high_res_{k}",
                )
            )

    if unet:
        # Upsampling part
        for i in range(levels):
            nodes.append(Ff.Node(nodes[-1], Fm.HaarUpsampling, {}))

            for k in range(highres_blocks):
                nodes.append(
                    Ff.Node(
                        nodes[-1],
                        Fm.AllInOneBlock,
                        {"subnet_constructor": subnet_conv},
                        name=f"conv_high_res_{k}",
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


def plot_batch_with_heatmaps(
    images_batch,
    anomaly_maps,
    ground_truth_masks=None,
    clip_val=None,
    grid_size=(5, 5),
    blur_radius=5,
    alpha=0.5,
    symmetric_norm=True,
    linthresh_q=0.9,
):
    """

    Anomaly maps are expected to be anomaly scores i.e. higher is more anomalous
    Make sure if passing in likelihoods, the maps should be inverted appropriately
    """

    # Not considering log probabilities higher than threshold
    if clip_val is not None:
        assert len(clip_val) == 2
        anomaly_maps = anomaly_maps.clip(min=clip_val[0], max=clip_val[1])

    # anomaly_maps = anomaly_maps.squeeze(1)

    # Putting batch as "channel" dimension
    # so kernel is run over all batches (independently)
    anomaly_maps = anomaly_maps.transpose(1, 2, 0)
    anomaly_maps = cv.blur(anomaly_maps, ksize=(blur_radius, blur_radius))
    anomaly_maps = anomaly_maps.transpose(2, 0, 1)

    # # Min-Max Scaling
    amin = anomaly_maps.min(axis=(1, 2), keepdims=True)
    amax = anomaly_maps.max(axis=(1, 2), keepdims=True)
    anomaly_maps = (anomaly_maps - amin) / (amax - amin)

    scaled_heatmaps = []
    for anomap in anomaly_maps:
        if symmetric_norm:
            norm = colors.SymLogNorm(
                linthresh=np.quantile(anomap, linthresh_q),
                vmin=np.quantile(anomap, 0.01),
                vmax=np.quantile(anomap, 0.99),
            )
        else:
            norm = colors.LogNorm(
                vmin=np.quantile(anomap, 0.01), vmax=np.quantile(anomap, 0.99)
            )
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
    anomaly_maps_rgb = plt.get_cmap("jet")(scaled_heatmaps)[:, :, :, :3]
    anomaly_maps_rgb = torch.from_numpy(anomaly_maps_rgb).float()
    anomaly_maps_rgb = anomaly_maps_rgb.permute(0, 3, 1, 2)

    if ground_truth_masks is not None:
        # Use opencv to combine the images and anomaly maps
        mask = np.expand_dims(ground_truth_masks, 1).repeat(3, 1).astype(np.float32)
        # mask = cv.c
        images_batch = images_batch.numpy()
        # images_batch = cv.addWeighted(images_batch, 0.5, mask, 0.5, 0)
        images_batch = images_batch * 0.4 + 0.6 * mask
        images_batch = torch.from_numpy(images_batch)

    # Merge images with anomaly maps with given alpha
    overlaid_images = alpha * images_batch + (1 - alpha) * anomaly_maps_rgb

    # Convert batch of tensors to grid
    grid = vutils.make_grid(
        overlaid_images, nrow=grid_size[1], padding=2, normalize=True
    )

    # Plot grid
    plt.figure(figsize=(3 * grid_size[0], 3 * grid_size[1]))
    plt.axis("off")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))

    return anomaly_maps_rgb


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def metrics_evaluation(scores, masks, expect_fpr=0.3, max_step=5000):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score, average_precision_score
    from skimage import measure
    import pandas as pd
    import time

    print("Calculating AUC, IOU, PRO metrics on testing data...")
    time_start = time.time()

    scores = scores - scores.min()

    # binary masks
    masks[masks <= 0.5] = 0
    masks[masks > 0.5] = 1
    masks = masks.astype(bool)

    # auc score (image level) for detection
    labels = np.any(masks, axis=(1, 2))
    # #         preds = scores.mean(1).mean(1)
    preds = np.mean(scores, axis=(1, 2))  # for detection

    det_auc_score = roc_auc_score(labels, preds)
    det_pr_score = average_precision_score(labels, preds)
    print(f"Det AUC: {det_auc_score:.4f} Det PR: {det_pr_score:.4f}")

    # auc score (per pixel level) for segmentation
    seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
    seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
    # metrics over all data
    print(f"Seg AUC: {seg_auc_score:.4f} Seg PR: {seg_pr_score:.4f}")

    # per region overlap and per image iou
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map.squeeze())
            for prop in props:
                (
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ) = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image  # corrected!
                intersection = (
                    np.logical_and(cropped_pred_label, cropped_mask)
                    .astype(np.float32)
                    .sum()
                )
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = (
                np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            )
            union = (
                np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            )
            if masks[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)

    # save results
    data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
    df_metrics = pd.DataFrame(
        data=data.T,
        columns=["thred", "fpr", "pros_mean", "pros_std", "ious_mean", "ious_std"],
    )
    # save results
    #         df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_iou.csv'), sep=',', index=False)

    # best per image iou
    best_miou = ious_mean.max()
    print(f"Best IOU: {best_miou:.4f}")

    # default 30% fpr vs pro, pro_auc
    idx = (
        fprs <= expect_fpr
    )  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)

    # save results
    data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
    df_metrics_30fpr = pd.DataFrame(
        data=data.T, columns=["thred", "fpr", "pros_mean", "pros_std"]
    )
    #         df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_{}.csv'.format(expect_fpr)), sep=',', index=False)

    # save auc, pro as 30 fpr
    #         with open(os.path.join(self.eval_path, 'pr_auc_pro_iou_{}.csv'.format(expect_fpr)), mode='w') as f:
    #                 f.write("det_pr, det_auc, seg_pr, seg_auc, seg_pro, seg_iou\n")
    #                 f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}")
    return df_metrics, df_metrics_30fpr


def plot_batch_with_normalized_heatmaps(
    images_batch,
    anomaly_maps,
    quantiles,
    ground_truth_masks=None,
    clip_val=None,
    grid_size=(5, 5),
    blur_radius=5,
    alpha=0.5,
    power_gamma = 1.0
):
    """

    Anomaly maps are expected to be anomaly scores i.e. higher is more anomalous
    Make sure if passing in likelihoods, the maps should be inverted appropriately
    """
    
    qmin = np.min(quantiles)
    qmax = np.max(quantiles)
    pnorm = mpl.colors.PowerNorm(power_gamma, vmin=qmin, vmax=qmax, clip=False)
    
    mapped_quantiles = (quantiles - qmin) / (qmax - qmin)

    # Build colormap from quantiles
    quantile_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom', 
        [(mq, color) for mq, color in zip(
            mapped_quantiles, mpl.cm.coolwarm(np.linspace(0, 1, len(mapped_quantiles)))
        )],
        N=256
    )

    quantile_cmap.set_over(color="Yellow")
    # quantile_cmap.set_over(color="orangered")

    # anomaly_maps = anomaly_maps.squeeze(1)

    # Putting batch as "channel" dimension
    # so kernel is run over all batches (independently)
    anomaly_maps = anomaly_maps.transpose(1, 2, 0)
    anomaly_maps = cv.blur(anomaly_maps, ksize=(blur_radius, blur_radius))
    anomaly_maps = anomaly_maps.transpose(2, 0, 1)

    scaled_heatmaps = pnorm(anomaly_maps)
    
#     cmap = mpl.cm.RdGy_r
    cmap = quantile_cmap
    anomaly_maps_rgb = cmap(scaled_heatmaps)[:, :, :, :3]
    anomaly_maps_rgb = torch.from_numpy(anomaly_maps_rgb).float()
    anomaly_maps_rgb = anomaly_maps_rgb.permute(0, 3, 1, 2)

    if ground_truth_masks is not None:
        # Use opencv to combine the images and anomaly maps
        mask = np.expand_dims(ground_truth_masks, 1).repeat(3, 1).astype(np.float32)
        # mask = cv.c
        images_batch = images_batch.numpy()
        # images_batch = cv.addWeighted(images_batch, 0.5, mask, 0.5, 0)
        images_batch = images_batch * 0.4 + 0.6 * mask
        images_batch = torch.from_numpy(images_batch)

    # Merge images with anomaly maps with given alpha
    overlaid_images = alpha * images_batch + (1 - alpha) * anomaly_maps_rgb

    # Convert batch of tensors to grid
    grid = vutils.make_grid(
        overlaid_images, nrow=grid_size[1], padding=2, normalize=True
    )

    # Plot grid
    fig, ax = plt.subplots(figsize=(3 * grid_size[0], 3 * grid_size[1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='3%', pad=0.05)

#     fig = plt.figure(figsize=(3 * grid_size[0], 3 * grid_size[1]))
#     plt.axis("off")
#     plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    
    
    fig.colorbar(mpl.cm.ScalarMappable(cmap=quantile_cmap, norm=pnorm),
                 cax=cax, orientation='horizontal', label='Percentiles',extend="max")

    qs = np.linspace(0.01, 0.999, 10)
    indxs = [0,5,8,9]
    labels = [0, 50, 90, 99]
    # labels = qs[indxs]
    positions = quantiles[indxs]
    cax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    cax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    
    im = ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    ax.axis("off")

    return anomaly_maps_rgb