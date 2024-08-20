"""Run SPD on a model."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.log import logger
from spd.models.base import Model, SPDModel
from spd.types import RootPath
from spd.utils import calc_attributions


class TMSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    n_features: int
    n_hidden: int
    n_instances: int
    k: int
    feature_probability: float
    train_bias: bool
    bias_val: float
    pretrained_model_path: RootPath


class BoolCircuitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["bool_circuit"] = "bool_circuit"
    k: int
    pretrained_model_path: RootPath


class DeepLinearConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["deep_linear"] = "deep_linear"
    n_features: int | None = None
    n_layers: int | None = None
    n_instances: int | None = None
    k: int | None = None
    pretrained_model_path: RootPath | None = None


class PiecewiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["piecewise"] = "piecewise"
    n_functions: int
    neurons_per_function: int
    n_layers: int
    feature_probability: float
    range_min: float
    range_max: float
    k: int
    handcoded_AB: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    topk: int | None = None
    batch_size: int
    steps: int
    print_freq: int
    save_freq: int | None = None
    lr: float
    max_sparsity_coeff: float
    pnorm: float | None = None
    pnorm_end: float | None = None
    lr_scale: Literal["linear", "constant", "cosine"] = "constant"
    lr_warmup_pct: float = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    loss_type: Literal["param_match", "behavioral"] = "param_match"
    sparsity_warmup_pct: float = 0.0
    batch_topk: bool = True
    topk_l2_coeff: float = 0.0
    task_config: DeepLinearConfig | BoolCircuitConfig | PiecewiseConfig | TMSConfig = Field(
        ..., discriminator="task_name"
    )


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def get_lr_scale_fn(
    lr_scale: Literal["linear", "constant", "cosine"],
) -> Callable[[int, int], float]:
    if lr_scale == "linear":
        return linear_lr
    elif lr_scale == "constant":
        return constant_lr
    elif lr_scale == "cosine":
        return cosine_decay_lr
    else:
        raise ValueError(f"Unknown lr_scale: {lr_scale}")


def get_current_pnorm(step: int, total_steps: int, pnorm_end: float | None = None) -> float:
    if pnorm_end is None:
        return 1.0
    progress = step / total_steps
    return 1 + (pnorm_end - 1) * progress


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


def calc_recon_mse(
    output: Float[Tensor, "... n_features"],
    labels: Float[Tensor, "... n_features"],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss


def calc_topk_l2(
    model: SPDModel,
    topk_indices: Int[Tensor, "... topk"],
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Args:
        model (SPDModel): The model to calculate the L2 penalty for.
        topk_indices (Int[Tensor, "batch ... topk"]): The topk indices to use for the L2 penalty.
            Will contain an n_instances dimension if the model has an n_instances dimension.
        device (str): The device to run computations on.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    batch_size = topk_indices.shape[0]
    n_instances = topk_indices.shape[1] if topk_indices.ndim == 3 else None
    accumulate_shape = (batch_size,) if n_instances is None else (batch_size, n_instances)

    topk_l2_penalty = torch.zeros(accumulate_shape, device=device)
    for A, B in zip(model.all_As(), model.all_Bs(), strict=True):
        n_features = A.shape[-2]
        n_hidden = B.shape[-1]
        # normed_A: [n_features, k] or [n_instances, n_features, k]
        # B: [k, n_hidden] or [n_instances, k, n_hidden]
        # topk_indices: [batch, topk] or [batch, n_instances, topk]
        expanded_topk_indices_A = einops.repeat(topk_indices, "b ... t -> b ... f t", f=n_features)
        expanded_A = einops.repeat(A, "... f k -> b ... f k", b=batch_size)
        A_topk: Float[Tensor, "batch ... n_features topk"] = expanded_A.gather(
            dim=-1, index=expanded_topk_indices_A
        )

        expanded_topk_indices_B = einops.repeat(topk_indices, "b ... t -> b ... h t", h=n_hidden)
        expanded_B = einops.repeat(B, "... k h -> b ... h k", b=batch_size)
        B_topk: Float[Tensor, "batch ... n_hidden topk"] = expanded_B.gather(
            dim=-1, index=expanded_topk_indices_B
        )

        AB_topk = torch.einsum("...ft,...ht->...fh", A_topk, B_topk)
        topk_l2_penalty = topk_l2_penalty + ((AB_topk) ** 2).mean(dim=(-2, -1))

    return topk_l2_penalty.mean(dim=0)  # Mean over batch dim


def optimize(
    model: SPDModel,
    config: Config,
    out_dir: Path,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model | None,
    plot_results_fn: Callable[..., plt.Figure | None] | None = None,
) -> None:
    assert (
        (config.pnorm is None and config.pnorm_end is not None)
        or (config.pnorm is not None and config.pnorm_end is None)
        or config.topk is not None
    ), "Exactly one of pnorm and pnorm_end must be set"

    has_instance_dim = hasattr(model, "n_instances")
    if config.loss_type == "param_match":
        assert pretrained_model is not None, "Need a pretrained model for param_match loss"
        pretrained_model.requires_grad_(False)
        pretrained_weights = pretrained_model.all_decomposable_params()
    else:
        pretrained_weights = None

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_scale_fn = get_lr_scale_fn(config.lr_scale)

    total_samples = 0

    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps)):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_scale_fn=lr_scale_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

        current_pnorm = (
            get_current_pnorm(step, config.steps, config.pnorm_end)
            if config.pnorm is None
            else config.pnorm
        )

        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        try:
            batch, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch, labels = next(data_iter)

        batch = batch.to(device=device)
        labels = labels.to(device=device)

        if pretrained_model is not None:
            labels = pretrained_model(batch)

        total_samples += batch.shape[0]  # don't include the number of instances

        sparsity_coeff = get_sparsity_coeff_linear_warmup(
            step=step,
            steps=config.steps,
            max_sparsity_coeff=config.max_sparsity_coeff,
            sparsity_warmup_pct=config.sparsity_warmup_pct,
        )

        out_topk = None
        topk_l2_loss = None
        if config.topk is not None:
            # First do a full forward pass and get the gradients w.r.t. inner_acts
            out, _, inner_acts = model(batch)
            attribution_scores = calc_attributions(out, inner_acts)

            # Get the topk indices of the attribution scores
            if config.batch_topk:
                if config.task_config.task_name == "piecewise":
                    batchsize, k = attribution_scores.shape
                    flattened_attribution_scores = attribution_scores.flatten()
                    topk_integer_indices = (
                        flattened_attribution_scores.abs().topk(config.topk * batchsize).indices
                    )
                    boolean_attributions = torch.zeros_like(flattened_attribution_scores).bool()
                    boolean_attributions[topk_integer_indices] = True
                    topk_indices = boolean_attributions.view(batchsize, -1)
                else:
                    batchsize, n_instances, k = attribution_scores.shape
                    flattened_attribution_scores = rearrange(attribution_scores, "b i k -> i (b k)")
                    topk_integer_indices = (
                        flattened_attribution_scores.abs()
                        .topk(config.topk * batchsize, dim=-1)
                        .indices
                    )
                    boolean_attributions = torch.zeros_like(flattened_attribution_scores).bool()
                    all_instance_indices = torch.arange(
                        n_instances, device=topk_integer_indices.device, dtype=torch.int64
                    )
                    all_instance_indices = all_instance_indices.unsqueeze(1).broadcast_to(
                        topk_integer_indices.shape
                    )
                    all_instance_indices = all_instance_indices.flatten()
                    topk_integer_indices = topk_integer_indices.flatten()
                    boolean_attributions[all_instance_indices, topk_integer_indices] = True
                    topk_indices = boolean_attributions.reshape((n_instances, batchsize, -1))
                    topk_indices = rearrange(topk_indices, "i b k -> b i k")

            else:
                topk_indices = attribution_scores.abs().topk(config.topk, dim=-1).indices

            # Do a forward pass with only the topk subnetworks
            out_topk, layer_acts, inner_acts_topk = model.forward_topk(
                batch, topk_indices=topk_indices
            )
            assert len(inner_acts_topk) == model.n_param_matrices
            if config.topk_l2_coeff > 0:
                topk_l2_loss = calc_topk_l2(model, topk_indices, device)

        else:
            out, layer_acts, inner_acts = model(batch)

        assert len(inner_acts) == model.n_param_matrices

        param_match_loss = torch.zeros(1, device=device)
        if config.loss_type == "param_match":
            assert pretrained_weights is not None
            for i, (A, B) in enumerate(zip(model.all_As(), model.all_Bs(), strict=True)):
                AB = torch.einsum("...fk,...kg->...fg", A, B)
                param_match_loss = param_match_loss + ((AB - pretrained_weights[i]) ** 2).mean(
                    dim=(-2, -1)
                )
            param_match_loss = param_match_loss / model.n_param_matrices

        out_recon_loss = calc_recon_mse(out, labels, has_instance_dim)

        if config.topk is None:
            sparsity_loss = torch.zeros_like(inner_acts[0], requires_grad=True)
            for feature_idx in range(out.shape[-1]):
                grad_layer_acts = torch.autograd.grad(
                    out[..., feature_idx].sum(),
                    layer_acts,
                    retain_graph=True,
                )
                sparsity_inner = torch.zeros_like(sparsity_loss, requires_grad=True)
                for param_matrix_idx in range(model.n_param_matrices):
                    # h_i * grad_h_i
                    sparsity_inner = sparsity_inner + (
                        inner_acts[param_matrix_idx]
                        * torch.einsum(
                            "...h,...kh->...k",
                            grad_layer_acts[param_matrix_idx].detach(),
                            model.all_Bs()[param_matrix_idx],
                        )
                    )

                sparsity_loss = sparsity_loss + sparsity_inner**2
            sparsity_loss = sparsity_loss / out.shape[-1] + 1e-16

            # Note the current_pnorm * 0.5 is because we have the squares of the sparsity inner
            # above
            sparsity_loss = ((sparsity_loss.abs() + 1e-16) ** (current_pnorm * 0.5)).sum(dim=-1)
            sparsity_loss = sparsity_loss.mean(dim=0)  # Mean over batch dim
        else:
            assert out_topk is not None
            sparsity_loss = calc_recon_mse(out_topk, labels, has_instance_dim)

        if step % config.print_freq == config.print_freq - 1 or step == 0:
            tqdm.write(f"Step {step}")
            tqdm.write(f"Current pnorm: {current_pnorm}")
            tqdm.write(f"Sparsity loss: \n{sparsity_loss}")
            tqdm.write(f"Reconstruction loss: \n{out_recon_loss}")
            if topk_l2_loss is not None:
                tqdm.write(f"topk l2 loss: \n{topk_l2_loss}")
            if config.loss_type == "param_match":
                param_match_loss_repr = (
                    param_match_loss.item() if len(param_match_loss) == 1 else param_match_loss
                )
                tqdm.write(f"Param match loss: \n{param_match_loss_repr}\n")

            fig = None
            if plot_results_fn is not None:
                fig = plot_results_fn(
                    model=model, device=device, topk=config.topk, step=step, out_dir=out_dir
                )

            if config.wandb_project:
                wandb.log(
                    {
                        "step": step,
                        "current_pnorm": current_pnorm,
                        "current_lr": step_lr,
                        "sparsity_loss": sparsity_loss.mean().item(),
                        "recon_loss": out_recon_loss.mean().item(),
                        "param_match_loss": param_match_loss.mean().item(),
                        "topk_l2_loss": topk_l2_loss.mean().item()
                        if topk_l2_loss is not None
                        else None,
                        "inner_acts": wandb.Image(fig) if fig else None,
                    },
                    step=step,
                )

        if config.save_freq is not None and step % config.save_freq == config.save_freq - 1:
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'model_{step}.pth'}")
            with open(out_dir / "config.json", "w") as f:
                json.dump(config.model_dump(), f, indent=4)
            tqdm.write(f"Saved config to {out_dir / 'config.json'}")

        out_recon_loss = out_recon_loss.mean()
        sparsity_loss = sparsity_loss.mean()
        param_match_loss = param_match_loss.mean()
        topk_l2_loss = topk_l2_loss.mean() if topk_l2_loss is not None else None

        if config.loss_type == "param_match":
            loss = param_match_loss + sparsity_coeff * sparsity_loss
        else:
            loss = out_recon_loss + sparsity_coeff * sparsity_loss

        if topk_l2_loss is not None:
            loss = loss + config.topk_l2_coeff * topk_l2_loss

        loss.backward()
        opt.step()

    torch.save(model.state_dict(), out_dir / f"model_{config.steps}.pth")
    logger.info(f"Saved model to {out_dir / f'model_{config.steps}.pth'}")
    if config.wandb_project:
        wandb.save(str(out_dir / f"model_{config.steps}.pth"))
