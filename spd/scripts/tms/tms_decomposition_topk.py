"""TMS decomposition script.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from spd.log import logger
from spd.types import RootPath
from spd.utils import (
    calculate_closeness_to_identity,
    init_wandb,
    load_config,
    permute_to_identity,
    set_seed,
)

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    topk: int
    topk_coeff: float
    n_features: int
    n_hidden: int
    n_instances: int
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    k: int | None = None
    lr_scale: Literal["linear", "constant", "cosine"] = "constant"
    lr_warmup_pct: float = 0.0
    bias_val: float = 0.0
    train_bias: bool = False
    feature_probability: float = 0.05
    pretrained_model_path: RootPath | None = None


class Model(nn.Module):
    def __init__(self, config: Config, device: str = "cuda"):
        super().__init__()
        self.config = config

        k = config.k if config.k is not None else config.n_features

        self.A = nn.Parameter(
            torch.empty((config.n_instances, config.n_features, k), device=device)
        )
        self.B = nn.Parameter(torch.empty((config.n_instances, k, config.n_hidden), device=device))

        bias_data = (
            torch.zeros((config.n_instances, config.n_features), device=device) + config.bias_val
        )
        self.b_final = nn.Parameter(bias_data) if config.train_bias else bias_data

        nn.init.xavier_normal_(self.A)
        # Fix the first instance to the identity to compare losses
        assert (
            config.n_features == k
        ), "Currently only supports n_features == k if fixing first instance to identity"
        self.A.data[0] = torch.eye(config.n_features, device=device)
        nn.init.xavier_normal_(self.B)

        self.feature_probability = config.feature_probability
        self.importance = torch.ones((), device=device)

    def forward(
        self, features: Float[Tensor, "... i f"]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        h_0 = torch.einsum("...if,ifk->...ik", features, normed_A)
        hidden = torch.einsum("...ik,ikh->...ih", h_0, self.B)

        h_1 = torch.einsum("...ih,ikh->...ik", hidden, self.B)
        hidden_2 = torch.einsum("...ik,ifk->...if", h_1, normed_A)

        pre_relu = hidden_2 + self.b_final
        out = F.relu(pre_relu)
        return out, h_0, h_1, hidden, pre_relu, normed_A

    def forward_topk(
        self,
        features: Float[Tensor, "... i f"],
        topk: int,
        grads: list[Float[Tensor, "... i k"]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        h_0 = torch.einsum("...if,ifk->...ik", features, normed_A)

        if grads is not None:
            topk_indices_0 = (grads[0] * h_0).abs().topk(topk, dim=-1).indices
        else:
            topk_indices_0 = h_0.abs().topk(topk, dim=-1).indices
        topk_values_0 = h_0.gather(dim=-1, index=topk_indices_0)
        h_0_topk = torch.zeros_like(h_0)
        h_0_topk.scatter_(dim=-1, index=topk_indices_0, src=topk_values_0)

        hidden = torch.einsum("...ik,ikh->...ih", h_0_topk, self.B)
        h_1 = torch.einsum("...ih,ikh->...ik", hidden, self.B)

        if grads is not None:
            topk_indices_1 = (grads[1] * h_1).abs().topk(topk, dim=-1).indices
        else:
            topk_indices_1 = h_1.abs().topk(topk, dim=-1).indices
        topk_values_1 = h_1.gather(dim=-1, index=topk_indices_1)
        h_1_topk = torch.zeros_like(h_1)
        h_1_topk.scatter_(dim=-1, index=topk_indices_1, src=topk_values_1)

        hidden_2 = torch.einsum("...ik,ifk->...if", h_1_topk, normed_A)

        pre_relu = hidden_2 + self.b_final
        out = F.relu(pre_relu)
        return out, h_0, h_1, hidden, pre_relu, normed_A

    def generate_batch(self, n_batch: int) -> torch.Tensor:
        device = self.A.device

        # Generate random features
        feat = torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=device)

        # Generate mask for which features to keep
        mask = (
            torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=device)
            <= self.feature_probability
        )

        # Ensure at least one feature is nonzero for each instance
        zero_instances = torch.all(~mask, dim=2)
        if zero_instances.any():
            # Generate random feature indices for zero instances
            random_feature = torch.randint(
                0, self.config.n_features, (n_batch, self.config.n_instances), device=device
            )

            # Create indexing tensors
            batch_indices = (
                torch.arange(n_batch, device=device)
                .unsqueeze(1)
                .expand(-1, self.config.n_instances)
            )
            instance_indices = (
                torch.arange(self.config.n_instances, device=device)
                .unsqueeze(0)
                .expand(n_batch, -1)
            )

            # Filter indices for zero instances
            zero_batch_indices = batch_indices[zero_instances]
            zero_instance_indices = instance_indices[zero_instances]
            zero_feature_indices = random_feature[zero_instances]

            # Set mask to True for selected features of zero instances
            mask[zero_batch_indices, zero_instance_indices, zero_feature_indices] = True

        # Apply the mask to the features
        batch = torch.where(mask, feat, torch.zeros(1, device=device))

        return batch


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def get_current_pnorm(step: int, total_steps: int, pnorm_end: float | None = None) -> float:
    if pnorm_end is None:
        return 1.0
    progress = step / total_steps
    return 1 + (pnorm_end - 1) * progress


def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    n_instances = x.shape[0]

    fig, axs = plt.subplots(
        1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
    )

    cmap = "Blues" if pos_only else "RdBu"
    ims = []
    for i in range(n_instances):
        ax = axs[0, i]
        instance_data = x[i, :, :].detach().cpu().float().numpy()
        max_abs_val = np.abs(instance_data).max()
        vmin = 0 if pos_only else -max_abs_val
        vmax = max_abs_val
        im = ax.matshow(instance_data, vmin=vmin, vmax=vmax, cmap=cmap)
        ims.append(im)
        ax.xaxis.set_ticks_position("bottom")
        if i == 0:
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else:
            ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")

    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    fig.subplots_adjust(bottom=0.2)

    return fig


def get_sparsity_coeff_linear_warmup(
    step: int, steps: int, max_sparsity_coeff: float, sparsity_warmup_pct: float
) -> float:
    warmup_steps = int(steps * sparsity_warmup_pct)
    if step < warmup_steps:
        return max_sparsity_coeff * (step / warmup_steps)
    return max_sparsity_coeff


def get_lr_with_warmup(
    step: int, steps: int, lr: float, lr_scale_fn: Callable[[int, int], float], lr_warmup_pct: float
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_scale_fn(step - warmup_steps, steps - warmup_steps)


def optimize(
    model: Model,
    config: Config,
    out_dir: Path,
    device: str,
    pretrained_model_path: RootPath | None = None,
) -> None:
    pretrained_W = None
    if pretrained_model_path:
        pretrained_W = torch.load(pretrained_model_path, weights_only=True, map_location="cpu")[
            "W"
        ].to(device)
        # Set requires_grad to False for the pretrained W
        pretrained_W.requires_grad = False
    opt = torch.optim.AdamW(list(model.parameters()), lr=config.lr)

    lr_scale_fn: Callable[[int, int], float]
    if config.lr_scale == "linear":
        lr_scale_fn = linear_lr
    elif config.lr_scale == "constant":
        lr_scale_fn = constant_lr
    elif config.lr_scale == "cosine":
        lr_scale_fn = cosine_decay_lr
    else:
        lr_scale_fn = constant_lr

    total_samples = 0

    for step in tqdm(range(config.steps)):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_scale_fn=lr_scale_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = model.generate_batch(config.batch_size)

        total_samples += batch.shape[0]  # don't include the number of instances

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Stage 1: Do a full forward pass and get the gradients w.r.t h_0 and h_1
            out, h_0, h_1, _, _, normed_A = model(batch)
            all_grads = [torch.zeros_like(h_0), torch.zeros_like(h_1)]
            for feature_idx in range(out.shape[-1]):
                grads = torch.autograd.grad(
                    out[:, :, feature_idx].sum(), [h_0, h_1], retain_graph=True
                )
                all_grads[0] += grads[0]
                all_grads[1] += grads[1]

            # Stage 2: Do a forward pass with topk
            out_topk, _, _, _, _, _ = model.forward_topk(batch, config.topk, all_grads)

            param_match_loss = torch.zeros(model.config.n_instances, device=device)
            if pretrained_model_path:
                # If the user passed a pretrained model, then calculate the param_match_loss
                assert pretrained_W is not None
                W = torch.einsum("ifk,ikh->ifh", normed_A, model.B)
                param_match_loss = ((pretrained_W[: model.config.n_instances] - W) ** 2).sum(
                    dim=(-2, -1)
                )

            error = model.importance * (batch - out) ** 2
            recon_loss = einops.reduce(error, "b i f -> i", "mean")

            error_topk = model.importance * (batch - out_topk) ** 2
            recon_loss_topk = einops.reduce(error_topk, "b i f -> i", "mean")

            with torch.inference_mode():
                if step % config.print_freq == config.print_freq - 1 or step == 0:
                    recon_repr = [f"{x:.4f}" for x in recon_loss]
                    recon_repr_topk = [f"{x:.4f}" for x in recon_loss_topk]
                    tqdm.write(f"Step {step}")
                    tqdm.write(f"Reconstruction loss: \n{recon_repr}")
                    tqdm.write(f"Reconstruction loss (topk): \n{recon_repr_topk}")
                    if pretrained_model_path:
                        param_match_repr = [f"{x:.4f}" for x in param_match_loss]
                        tqdm.write(f"Param match loss: \n{param_match_repr}")

                    closeness_vals: list[float] = []
                    permuted_A_T_list: list[torch.Tensor] = []
                    for i in range(model.config.n_instances):
                        permuted_matrix = permute_to_identity(normed_A[i].T.abs())
                        closeness = calculate_closeness_to_identity(permuted_matrix)
                        closeness_vals.append(closeness)
                        permuted_A_T_list.append(permuted_matrix)
                    permuted_A_T = torch.stack(permuted_A_T_list, dim=0)

                    fig = plot_A_matrix(permuted_A_T, pos_only=True)

                    fig.savefig(out_dir / f"A_{step}.png")
                    plt.close(fig)
                    tqdm.write(f"Saved A matrix to {out_dir / f'A_{step}.png'}")
                    if config.wandb_project:
                        wandb.log(
                            {
                                "step": step,
                                "lr": step_lr,
                                "recon_loss": recon_loss[1:].mean().item(),
                                "recon_loss_topk": recon_loss_topk[1:].mean().item(),
                                "param_match_loss": param_match_loss[1:].mean().item(),
                                "closeness": sum(closeness_vals[1:])
                                / (model.config.n_instances - 1),
                                "A_matrix": wandb.Image(fig),
                            },
                            step=step,
                        )

            recon_loss = recon_loss.mean()
            recon_loss_topk = recon_loss_topk.mean()
            param_match_loss = param_match_loss.mean()

            if pretrained_model_path:
                loss = param_match_loss + config.topk_coeff * recon_loss_topk
            else:
                loss = recon_loss + config.topk_coeff * recon_loss_topk

        loss.backward()
        assert model.A.grad is not None
        # Don't update the gradient of the 0th instance (which we fixed to be the identity)
        model.A.grad[0] = torch.zeros_like(model.A.grad[0])
        opt.step()

    torch.save(model.state_dict(), out_dir / "model.pth")
    if config.wandb_project:
        wandb.save(str(out_dir / "model.pth"))


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"lr{config.lr}_"
            f"topk{config.topk}_"
            f"topk_coeff{config.topk_coeff}_"
            f"bs{config.batch_size}_"
            f"ft{config.n_features}_"
            f"hid{config.n_hidden}"
        )
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        # Save the config to wandb
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "final_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(mode="json"), f, indent=2)
            wandb.save(str(config_path), policy="now", base_path=tmp_dir)
            # Unfortunately wandb.save is async, so we need to wait for it to finish before
            # continuing, and wandb python api provides no way to do this.
            # TODO: Find a better way to do this.
            time.sleep(1)

    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(config=config, device=device)

    optimize(
        model=model,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model_path=config.pretrained_model_path,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
