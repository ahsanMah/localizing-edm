import torch
import numpy as np
from tqdm import tqdm
import dnnlib
import matplotlib.pyplot as plt
import torch.nn as nn


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


class EDMScorer(torch.nn.Module):
    def __init__(
        self,
        net,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        rho=7,  # Time step discretization.
        num_steps=20,  # Number of nosie levels to evaluate.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = net.model.eval()

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        orig_t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        self.sigma_steps = net.round_sigma(orig_t_steps)

    def forward(
        self, x, sigma, class_labels=None, force_fp32=False, debug=False, **model_kwargs
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()

        batch_scores = []
        for sigma in self.sigma_steps:
            sigma = sigma.reshape(-1, 1, 1, 1)
            c_noise = sigma.log() / 4

            score = self.model(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )
            score *= c_out.flatten()
            score = score.mean(dim=1)
            batch_scores.append(score)

            if debug:
                print("c_in:", c_skip)
                print("c_noise:", c_noise)
                print("c_out:", c_out)

        batch_scores = torch.stack(batch_scores, axis=1)

        return batch_scores


class VEScorer(torch.nn.Module):
    def __init__(
        self,
        net,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.1,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        num_sigmas=20,  # Number of noise levels to evaluate.
        device=torch.device("cpu"),  # Device to use.
        post_downsample=False,  # Downsample the image after scoring.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.model = net.model.eval().to(device)
        self.num_steps = num_sigmas

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Compute the noise levels to evaluate.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=device)
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (self.num_steps - 1))
        )
        self.register_buffer(
            "sigma_steps", net.round_sigma(torch.sqrt(orig_t_steps)).to(torch.float64)
        )

        self.downsample = post_downsample
        if self.downsample:
            self.norm_pool = SpatialNorm2D(num_sigmas)

    @torch.no_grad()
    def forward(
        self,
        x,
        class_labels=None,
        force_fp32=False,
        outscale=False,
        debug=False,
        **model_kwargs,
    ):
        # x = x.to(torch.float32)
        class_labels = None
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_in = 1

        b,c,h,w = x.shape
        batch_scores = torch.zeros(
            size=(b, self.num_steps, h, w),
            device=x.device, dtype=torch.float32,
            requires_grad=False,
        )

        for idx, sigma in enumerate(self.sigma_steps):
            sigma = sigma.reshape(-1, 1, 1, 1)
            c_noise = (0.5 * sigma).log().to(torch.float32)

            score = self.model(
                x.to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            ).to(torch.float32)
            score = score.mean(dim=1)

            batch_scores[:, idx].copy_(score)

            if debug:
                print("c_in:", c_skip)
                print("c_noise:", c_noise)
                print("c_out:", c_out)

        if self.downsample:
            batch_scores = self.norm_pool(batch_scores)

        if outscale:
            c_out = self.sigma_steps.reshape(1, -1, 1, 1)
            batch_scores = c_out * batch_scores
            print("OUTSCALING")

        return batch_scores

    # @torch.no_grad
    # @torch.jit.script
    def build_vectorized_forward(
        self,
        class_labels=None,
        force_fp32=False,
        outscale=False,
        debug=False,
        **model_kwargs,
    ):
        dtype = torch.float16 if (self.use_fp16 and not force_fp32) else torch.float32

        if outscale:
            c_out = self.sigma_steps.reshape(1, -1, 1, 1).to(torch.float32)
        else:
            c_out = 1

        c_noise = (0.5 * self.sigma_steps).log().float()
        num_steps = self.num_steps

        def forward(
            x,
            class_labels=None,
            num_steps=num_steps,
            dtype=dtype,
        ):
            x = x.to(torch.float32)
            n, c, h, w = x.shape

            x_batch = x.repeat_interleave(num_steps, dim=0)

            batch_scores = self.model(
                x_batch.to(dtype),
                c_noise.repeat(x.shape[0]).flatten(),
                class_labels=class_labels,
            )

            batch_scores = batch_scores.reshape(n, self.num_steps, c, h, w)
            batch_scores = batch_scores.mean(dim=2).to(torch.float32)

            batch_scores = c_out * batch_scores

            return batch_scores

        return forward


def compute_scores(
    net,
    dataset_iterator,
    sigma_min=0.002,
    sigma_max=80,
    num_steps=20,
    rho=7,
    model_type="ve",
    device=torch.device("cuda"),
    return_norms=False,
    vectorized=False,
    outscale=False,
):
    if model_type == "edm":
        scorer = EDMScorer(
            net,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=net.sigma_data,  # for EDM cond
        )
    elif model_type == "ve":
        scorer = VEScorer(
            net,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        raise ValueError(f"Unknown schedule {model_type}")

    # Assign score function
    if vectorized:
        score_fn = scorer.vectorized
    else:
        score_fn = scorer

    scores = []
    for x, _ in tqdm(dataset_iterator, desc="Scoring", leave=False):
        x = x.to(device).to(torch.float32) / 127.5 - 1
        batch_scores = score_fn(x, outscale=outscale)

        if return_norms:
            batch_scores = torch.linalg.norm(
                batch_scores.reshape((batch_scores.shape[0], num_steps)),
                axis=-1,
            )

        scores.append(batch_scores.cpu().numpy())

        break

    scores = np.concatenate(scores, axis=0)

    return scores

def build_img_grid(imgs, rows):
    b,c,h,w = imgs.shape
    cols = b // rows
    # Channels last
    img = imgs.permute(0,2,3,1).cpu()
    # Make grid of images
    # (x,y), h,w,c
    img = img.reshape(rows,cols,h,w,c)
    # Swap half batch across cols
    # And half across rows
    # x,h y,w, c
    img = img.permute(0,2,1,3,4)
    # for each img_row,
    #    each pixel row will have y images concatted
    #  every subsewuent will
    img = img.reshape(h*rows, w*cols, c)
    
    return img

def plot_score_grid(scores, num_samples=9, plot_sigma_idxs=[0, 5, 10, 19]):
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    gridh = gridw = int(num_samples**0.5)
    n = gridh * gridw

    row = int(np.sqrt(len(plot_sigma_idxs)))
    col = len(plot_sigma_idxs) // row
    fig, axs = plt.subplots(row, col, figsize=(gridw * 2.5, gridh * 2.5), squeeze=False)

    for i, ax in zip(plot_sigma_idxs, axs.flatten()):
        image = np.abs(scores[:n, i])[:, None, ...]
        image = image.reshape(gridh, gridw, *image.shape[1:]).transpose(0, 3, 1, 4, 2)
        image = image.reshape(
            gridh * scores.shape[2],
            gridw * scores.shape[2],
        )
        ax.matshow(image)
        ax.axis("off")

    return
