import torch
import numpy as np
from tqdm import tqdm
import dnnlib


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
        self.model = model

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

        return score


class VEScorer(torch.nn.Module):
    def __init__(
        self,
        net,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        num_steps=20,  # Number of noise levels to evaluate.
        device=torch.device("cuda"),  # Device to use.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.model = net.model.eval().to(device)
        self.num_steps = num_steps

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Compute the noise levels to evaluate.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        self.register_buffer(
            "sigma_steps", net.round_sigma(torch.sqrt(orig_t_steps)).to(torch.float64)
        )

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
        x = x.to(torch.float32)
        class_labels = None
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_in = 1

        batch_scores = []
        for sigma in self.sigma_steps:
            sigma = sigma.reshape(-1, 1, 1, 1)
            c_noise = (0.5 * sigma).log().to(torch.float32)

            score = self.model(
                x.to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )
            score = score.mean(dim=1)
            # print(t, score.mean())
            batch_scores.append(score)

            if debug:
                print("c_in:", c_skip)
                print("c_noise:", c_noise)
                print("c_out:", c_out)

        batch_scores = torch.stack(batch_scores, axis=1).to(torch.float32)

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
        dtype = (
                torch.float16
                if (self.use_fp16 and not force_fp32)
                else torch.float32
        )

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
