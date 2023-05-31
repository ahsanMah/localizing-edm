import torch
import numpy as np
from tqdm import tqdm
import dnnlib


class EDMScorer(torch.nn.Module):
    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

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
        c_noise = sigma.log() / 4

        score = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        score *= c_out.flatten()

        if debug:
            print("c_in:", c_skip)
            print("c_noise:", c_noise)
            print("c_out:", c_out)

        return score


class VEScorer(torch.nn.Module):
    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = model

    @torch.inference_mode()
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

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        score = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        # score = c_out * score.to(torch.float32)

        if debug:
            print("c_in:", c_skip)
            print("c_noise:", c_noise)
            print("c_out:", c_out)

        return score


def compute_scores(
    net,
    dataset_kwargs,
    data_loader_kwargs,
    sigma_min=0.002,
    sigma_max=80,
    num_steps=20,
    rho=7,
    model_type="ve",
    device=torch.device("cuda"),
    return_norms=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if model_type == "edm":
        score_fn = EDMScorer(
            net.model,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=net.sigma_data,  # for EDM cond
        )
    elif model_type == "ve":
        score_fn = VEScorer(
            net.model,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    batch_size = data_loader_kwargs["batch_size"]
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            **data_loader_kwargs,
        )
    )

    if model_type == "edm":
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        orig_t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        # sigma_steps = torch.cat(
        #     [net.round_sigma(orig_t_steps), torch.zeros_like(orig_t_steps[:1])]
        # )  # t_N = 0
        sigma_steps = net.round_sigma(orig_t_steps)
        print(sigma_steps)

    elif model_type == "ve":
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = torch.sqrt(orig_t_steps)
    else:
        raise ValueError(f"Unknown schedule {schedule}")


    scores = []
    for x, _ in tqdm(dataset_iterator, desc="Scoring", leave=False):
        x = x.to(device)
        batch_scores = []

        for i, t in enumerate(sigma_steps):
            score = score_fn(x, t, debug=True)
            score = score.cpu().numpy()
            print(t, score.mean())

            if return_norms:
                score = np.linalg.norm(
                    score.reshape((score.shape[0], -1)),
                    axis=-1,
                )
            batch_scores.append(score)
        
        batch_scores = np.stack(batch_scores, axis=1)
        scores.append(batch_scores)

        break
    
    scores = np.concatenate(scores, axis=0)

    return scores
