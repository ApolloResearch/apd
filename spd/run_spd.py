"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.log import logger
from spd.models.base import Model, SPDFullRankModel, SPDRankPenaltyModel
from spd.types import ModelPath, Probability
from spd.utils import calc_recon_mse, calc_topk_mask, calculate_attributions


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    k: PositiveInt
    feature_probability: Probability
    train_bias: bool
    bias_val: float
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )
    pretrained_model_path: ModelPath  # e.g. wandb:spd-tms/runs/si0zbfxf


class PiecewiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["piecewise"] = "piecewise"
    n_functions: PositiveInt
    neurons_per_function: PositiveInt
    n_layers: PositiveInt
    feature_probability: Probability
    range_min: float
    range_max: float
    k: PositiveInt
    init_scale: float = 1.0
    target_seed: int | None = None
    dataset_seed: int | None = None
    simple_bias: bool = False
    handcoded_AB: bool = False


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_mlp"] = "residual_mlp"
    k: PositiveInt
    feature_probability: Probability
    init_scale: float = 1.0
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    pretrained_model_path: ModelPath  # e.g. wandb:spd-resid-mlp/runs/j9kmavzi


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    spd_type: Literal["full_rank", "rank_penalty"] = "rank_penalty"
    seed: int = 0
    topk: PositiveFloat | None = None
    batch_topk: bool = True
    exact_topk: bool = False
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: PositiveInt | None = None
    slow_images: bool = False
    save_freq: PositiveInt | None = None
    lr: PositiveFloat
    out_recon_coeff: NonNegativeFloat | None = None
    act_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    topk_recon_coeff: NonNegativeFloat | None = None
    topk_l2_coeff: NonNegativeFloat | None = None
    schatten_coeff: NonNegativeFloat | None = None
    schatten_pnorm: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat | None = None
    distil_from_target: bool = False
    pnorm: PositiveFloat | None = None
    m: PositiveInt | None = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    unit_norm_matrices: bool = False
    attribution_type: Literal["gradient", "ablation", "activation"] = "gradient"
    task_config: PiecewiseConfig | TMSTaskConfig | ResidualMLPTaskConfig = Field(
        ..., discriminator="task_name"
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check valid combinations of topk and batch_size
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer():
                    logger.warning(
                        f"batch_size * topk={self.batch_size * self.topk} is not an integer, will "
                        f"round down from {self.batch_size * self.topk} to "
                        f"{int(self.batch_size * self.topk)} when calculating topk_mask"
                    )
            else:
                if not self.topk.is_integer():
                    raise ValueError("topk must be an integer when not using batch_topk")

        # Warn if neither topk_recon_coeff nor lp_sparsity_coeff is set
        if not self.topk_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither topk_recon_coeff nor lp_sparsity_coeff is set")

        # If topk_recon_coeff is set, topk must be set
        if self.topk_recon_coeff is not None:
            assert self.topk is not None, "topk must be set if topk_recon_coeff is set"

        # If lp_sparsity_coeff is set, pnorm must be set
        if self.lp_sparsity_coeff is not None:
            assert self.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"

        # Check that topk_l2_coeff and topk_recon_coeff are None if topk is None
        if self.topk is None:
            assert self.topk_l2_coeff is None, "topk_l2_coeff is not None but topk is"
            assert self.topk_recon_coeff is None, "topk_recon_coeff is not None but topk is"

        # Give a warning if both out_recon_coeff and param_match_coeff are > 0
        if (
            self.param_match_coeff is not None
            and self.param_match_coeff > 0
            and self.out_recon_coeff is not None
            and self.out_recon_coeff > 0
        ):
            logger.warning(
                "Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one."
            )

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.topk_l2_coeff == 0:
            logger.warning(f"topk_l2_coeff {msg}")
        if self.topk_recon_coeff == 0:
            logger.warning(f"topk_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert (
                self.lr_exponential_halflife is not None
            ), "lr_exponential_halflife must be set if lr_schedule is exponential"

        if self.spd_type in ["full_rank"]:
            assert not self.unit_norm_matrices, "Can't unit norm matrices if using full rank"

        if self.schatten_coeff is not None:
            assert (
                self.spd_type == "rank_penalty"
            ), "schatten_coeff is not None but spd_type is not rank_penalty"
            assert (
                self.schatten_pnorm is not None
            ), "schatten_pnorm must be set if schatten_coeff is set"

        if self.distil_from_target and not isinstance(self.task_config, PiecewiseConfig):
            raise ValueError("distil_from_target is currently only supported for piecewise")

        if self.m is not None:
            assert self.spd_type == "rank_penalty", "Cannot set m for non-rank penalty SPD"

        if isinstance(self.task_config, PiecewiseConfig) and self.task_config.handcoded_AB:
            assert (
                self.task_config.n_layers == 1
            ), "Handcoded AB not supported for >1 layer models due to a bug in the W_out matrices"

        if isinstance(self.task_config, ResidualMLPTaskConfig):
            assert self.spd_type == "rank_penalty", "Only rank penalty supported for residual mlp"

        return self


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.pnorm is not None:
        run_suffix += f"p{config.pnorm:.2e}_"
    if config.lp_sparsity_coeff is not None:
        run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    if config.topk is not None:
        run_suffix += f"topk{config.topk:.2e}_"
    if config.topk_recon_coeff is not None:
        run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
    if config.topk_l2_coeff is not None:
        run_suffix += f"topkl2_{config.topk_l2_coeff:.2e}_"
    if config.schatten_pnorm is not None:
        run_suffix += f"schatp{config.schatten_pnorm:.2e}_"
    if config.schatten_coeff is not None:
        run_suffix += f"schatten{config.schatten_coeff:.2e}_"
    if config.act_recon_coeff is not None:
        run_suffix += f"actrecon_{config.act_recon_coeff:.2e}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"attr-{config.attribution_type[:3]}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def calc_topk_l2_full_rank(
    subnet_param_vals: list[
        Float[Tensor, "k d_out"]
        | Float[Tensor, "k d_in d_out"]
        | Float[Tensor, "n_instances k d_out"]
        | Float[Tensor, "n_instances k d_in d_out"]
    ],
    topk_mask: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"],
    n_params: int,
    n_instances: int | None = None,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        subnetwork_params: The parameters of the subnetwork.
        topk_mask: The topk mask to use for the L2 penalty.
        n_params: The number of decomposable parameters in the model.
        n_instances: The number of instances in the model.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance.
    """
    assert len(subnet_param_vals) > 0, "No subnetwork parameters provided"

    accumulate_shape = (n_instances,) if n_instances is not None else ()

    topk_mask = topk_mask.to(subnet_param_vals[0].dtype)
    topk_l2_penalty = torch.zeros(accumulate_shape, device=subnet_param_vals[0].device)
    batch_size = topk_mask.shape[0]
    for subnetwork_param_val in subnet_param_vals:
        if n_instances is None:
            # subnetwork_param_val: [k, d_in, d_out] or [k, d_out] (if bias param)
            # topk_mask: [batch, k]
            ein_str = "k ... d_out, batch k -> batch ... d_out"
            # mean over all dims
            assert subnetwork_param_val.ndim in (3, 2), "Invalid number of dimensions"
            mean_dims = tuple(range(subnetwork_param_val.ndim))
        else:
            # subnetwork_param_val: [n_instances, k, d_in, d_out] or [n_instances, k, d_out]
            # topk_mask: [batch, n_instances, k]
            ein_str = "n_instances k ... d_out, batch n_instances k -> batch n_instances ... d_out"
            # mean over all dims except the n_instances dim
            assert subnetwork_param_val.ndim in (4, 3), "Invalid number of dimensions"
            mean_dims = (0, -2, -1) if subnetwork_param_val.ndim == 4 else (0, -1)

        topk_params = einops.einsum(subnetwork_param_val, topk_mask, ein_str)
        topk_l2_penalty = topk_l2_penalty + ((topk_params) ** 2).sum(dim=mean_dims)

    return topk_l2_penalty / n_params / batch_size


def calc_schatten_loss(
    As_and_Bs_vals: list[
        tuple[
            Float[Tensor, "n_instances k d_layer_in m"] | Float[Tensor, "k d_layer_in m"],
            Float[Tensor, "n_instances k m d_layer_out"] | Float[Tensor, "k m d_layer_out"],
        ]
    ],
    mask: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"],
    p: float,
    n_params: int,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Schatten p-norms of the topk subnetworks and sum them.

    Args:
        As_and_Bs_vals: List of tuples containing A and B matrices for each layer
        mask: The mask to use for the Schatten p-norm penalty. May be a binary mask (if topk) or
            a float mask (if lp sparsity).
        p: The Schatten p-norm to use (from config.schatten_pnorm)
        n_params: The number of parameters in the model
    Returns:
        The Schatten p-norm penalty for the topk subnetworks
    """
    n_instances = mask.shape[1] if mask.ndim == 3 else None
    accumulate_shape = (n_instances,) if n_instances is not None else ()

    schatten_penalty = torch.zeros(accumulate_shape, device=As_and_Bs_vals[0][0].device)
    batch_size = mask.shape[0]

    for A, B in As_and_Bs_vals:
        # A: [k, d_in, m] or [n_instances, k, d_in, m]
        # B: [k, m, d_out] or [n_instances, k, m, d_out]
        # mask: [batch, k] or [batch, n_instances, k]

        # Compute S_A = A^T A and S_B = B B^T
        S_A = einops.einsum(A, A, "... k d_in m, ... k d_in m -> ... k m")
        S_B = einops.einsum(B, B, "... k m d_out, ... k m d_out -> ... k m")

        S_AB = S_A * S_B

        # Apply topk mask
        S_AB_topk = einops.einsum(S_AB, mask, "... k m, batch ... k -> batch ... k m")

        # Sum the Schatten p-norm
        schatten_penalty = schatten_penalty + ((S_AB_topk + 1e-16) ** (0.5 * p)).sum(
            dim=(0, -2, -1)
        )

    return schatten_penalty / n_params / batch_size


def calc_param_match_loss(
    pretrained_weights: dict[str, Float[Tensor, "n_instances d_out"] | Float[Tensor, " d_out"]],
    subnetwork_params_summed: dict[
        str, Float[Tensor, "n_instances d_out"] | Float[Tensor, " d_out"]
    ],
    param_map: dict[str, str],
    n_params: int,
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the parameter match loss.

    This is the L2 difference between the combined parameter matrices of the SPD Model and the
    target params.

    Args:
        pretrained_weights: The pretrained weights to be matched. May have an n_instances and/or
            d_in dimension.
        subnetwork_params_summed: The parameters of the SPD Model (that have already been summed
            over the subnetwork dimension). May have an n_instances and/or d_in dimension.
        param_map: A map from keys in pretrained_weights to keys in subnetwork_params_summed.
        has_instance_dim: Whether the model has an n_instances dimension.
        n_params: The number of parameters in the model.

    Returns:
        The parameter match loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    device = next(iter(subnetwork_params_summed.values())).device
    param_match_loss = torch.tensor(0.0, device=device)
    for target_param_name, subnetwork_param_name in param_map.items():
        pretrained_weight = pretrained_weights[target_param_name]
        subnetwork_param = subnetwork_params_summed[subnetwork_param_name]
        if has_instance_dim:
            # params: [n_instances, d_out] or [n_instances, d_in, d_out]
            assert pretrained_weight.ndim in (3, 2)
            mean_dims = (-2, -1) if pretrained_weight.ndim == 3 else (-1,)
        else:
            # params: [d_out] or [d_in, d_out]
            assert pretrained_weight.ndim in (2, 1)
            mean_dims = (-2, -1) if pretrained_weight.ndim == 2 else (-1,)
        param_match_loss = param_match_loss + ((subnetwork_param - pretrained_weight) ** 2).sum(
            dim=mean_dims
        )
    return param_match_loss / n_params


def calc_lp_sparsity_loss(
    out: Float[Tensor, "batch n_instances d_model_out"] | Float[Tensor, "batch d_model_out"],
    attributions: Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"],
    step_pnorm: float,
) -> Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    Args:
        out: The output of the model.
        attributions: The attributions to use for the sparsity loss.
        step_pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension. Note that we keep the batch and k dimensions as we need them if calculating
            the schatten loss.
    """
    # Average the attributions over the output dimensions
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss_per_k = (attributions.abs() + 1e-16) ** (step_pnorm * 0.5)
    return lp_sparsity_loss_per_k


def calc_act_recon(
    target_post_acts: dict[
        str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]
    ],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """MSE between all target model activations and the output of each subnetwork in the SPD model.

    Args:
        target_post_acts: The activations after each layer in the target model.
        layer_acts: The activations after each subnetwork in the SPD model.

    Returns:
        The activation reconstruction loss. Will have an n_instances dimension if the model has an
            n_instances dimension, otherwise a scalar.
    """
    assert (
        target_post_acts.keys() == layer_acts.keys()
    ), f"Layer keys must match: {target_post_acts.keys()} != {layer_acts.keys()}"

    device = next(iter(layer_acts.values())).device

    total_act_dim = 0  # Accumulate the d_out over all layers for normalization
    loss = torch.zeros(1, device=device)
    for layer_name in target_post_acts:
        total_act_dim += target_post_acts[layer_name].shape[-1]

        error = ((target_post_acts[layer_name] - layer_acts[layer_name]) ** 2).sum(dim=-1)
        loss = loss + error

    # Normalize by the total number of output dimensions and mean over the batch dim
    return (loss / total_act_dim).mean(dim=0)


def optimize(
    model: SPDFullRankModel | SPDRankPenaltyModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model,
    param_map: dict[str, str] | None = None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)
    has_instance_dim = hasattr(model, "n_instances")

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    if config.param_match_coeff is not None:
        assert param_map is not None, "Need a param_map for param_match loss"
        # Check that our param_map contains all the decomposable param names
        assert set(param_map.keys()) == set(pretrained_model.all_decomposable_params().keys())
        assert set(param_map.values()) == set(model.all_subnetwork_params_summed().keys())

    pretrained_model.to(device=device)

    n_params = sum(p.numel() for p in list(model.all_subnetwork_params_summed().values()))
    if has_instance_dim:
        # All subnetwork param have an n_instances dimension
        n_params = n_params / model.n_instances

    epoch = 0
    total_samples = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(
                model, SPDRankPenaltyModel
            ), "Can only norm matrices in SPDRankPenaltyModel instances"
            model.set_matrices_to_unit_norm()

        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in opt.param_groups:
            group["lr"] = step_lr

        opt.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)[0]  # Ignore labels here, we use the output of pretrained_model
        except StopIteration:
            tqdm.write(f"Epoch {epoch} finished, starting new epoch")
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]

        batch = batch.to(device=device)
        target_out, pre_acts, post_acts = pretrained_model(batch)

        total_samples += batch.shape[0]

        # Do a forward pass with all subnetworks
        out, layer_acts, inner_acts = model(batch)

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, target_out, has_instance_dim)

        param_match_loss = None
        if config.param_match_coeff is not None:
            assert param_map is not None, "Need a param_map for param_match loss"
            param_match_loss = calc_param_match_loss(
                pretrained_weights=pretrained_model.all_decomposable_params(),
                subnetwork_params_summed=model.all_subnetwork_params_summed(),
                param_map=param_map,
                n_params=n_params,
                has_instance_dim=has_instance_dim,
            )

        attributions = calculate_attributions(
            model=model,
            batch=batch,
            out=out,
            target_out=target_out,
            pre_acts=pre_acts,
            post_acts=post_acts,
            inner_acts=inner_acts,
            attribution_type=config.attribution_type,
        )

        lp_sparsity_loss_per_k = None
        if config.lp_sparsity_coeff is not None:
            assert config.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"
            lp_sparsity_loss_per_k = calc_lp_sparsity_loss(
                out=out, attributions=attributions, step_pnorm=config.pnorm
            )

        (
            out_topk,
            topk_l2_loss,
            schatten_loss,
            topk_recon_loss,
            topk_mask,
            layer_acts_topk,
        ) = None, None, None, None, None, None
        if config.topk is not None:
            # We always assume the final subnetwork is the one we want to distil
            topk_attrs: Float[Tensor, "batch ... k"] = (
                attributions[..., :-1] if config.distil_from_target else attributions
            )
            if config.exact_topk:
                # Currently only valid for batch_topk and n_instances = 1. Would need to change the
                # topk argument in calc_topk_mask to allow for tensors if relaxing these constraints
                assert config.batch_topk, "exact_topk only works if batch_topk is True"
                assert (
                    hasattr(model, "n_instances") and model.n_instances == 1
                ), "exact_topk only works if n_instances = 1"
                # Get the exact number of active features over the batch
                exact_topk = ((batch != 0).sum() / batch.shape[0]).item()
                topk_mask = calc_topk_mask(topk_attrs, exact_topk, batch_topk=True)
            else:
                topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:
                # Add back the final subnetwork index to the topk mask and set it to True
                last_subnet_mask = torch.ones(
                    (*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device
                )
                topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1)

            # Do a forward pass with only the topk subnetworks
            out_topk, layer_acts_topk, inner_acts_topk = model(batch, topk_mask=topk_mask)

            if config.topk_l2_coeff is not None:
                topk_l2_loss = calc_topk_l2_full_rank(
                    subnet_param_vals=list(model.all_subnetwork_params().values()),
                    topk_mask=topk_mask,
                    n_params=n_params,
                    n_instances=getattr(model, "n_instances", None),
                )

            if config.topk_recon_coeff is not None:
                assert out_topk is not None
                topk_recon_loss = calc_recon_mse(out_topk, target_out, has_instance_dim)

        act_recon_loss = None
        if config.act_recon_coeff is not None:
            if isinstance(config.task_config, ResidualMLPTaskConfig):
                # For now, we treat resid-mlp special in that we take the post-relu activations
                assert layer_acts_topk is not None
                post_acts_after_relu = {}
                layer_acts_topk_after_relu = {}
                for i in range(len(model.layers)):
                    post_acts_after_relu[f"layers.{i}.linear1"] = torch.nn.functional.relu(
                        post_acts[f"layers.{i}.linear1"]
                    )
                    layer_acts_topk_after_relu[f"layers.{i}.linear1"] = torch.nn.functional.relu(
                        layer_acts_topk[f"layers.{i}.linear1"]
                    )

                act_recon_loss = calc_act_recon(
                    target_post_acts=post_acts_after_relu, layer_acts=layer_acts_topk_after_relu
                )
            else:
                act_recon_loss = calc_act_recon(
                    target_post_acts=post_acts,
                    layer_acts=layer_acts if layer_acts_topk is None else layer_acts_topk,
                )

        if config.schatten_coeff is not None:
            assert isinstance(
                model, SPDRankPenaltyModel
            ), "Schatten only supported for SPDRankPenaltyModel"
            mask = topk_mask if topk_mask is not None else lp_sparsity_loss_per_k
            assert mask is not None
            schatten_pnorm = config.schatten_pnorm if config.schatten_pnorm is not None else 1.0
            # Use the attributions as the mask in the lp case, and topk_mask otherwise
            schatten_loss = calc_schatten_loss(
                As_and_Bs_vals=list(model.all_As_and_Bs().values()),
                mask=mask,
                p=schatten_pnorm,
                n_params=n_params,
            )

        lp_sparsity_loss = None
        if lp_sparsity_loss_per_k is not None:
            # Sum over the k dimension (-1) and mean over the batch dimension (0)
            lp_sparsity_loss = lp_sparsity_loss_per_k.sum(dim=-1).mean(dim=0)

        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        if param_match_loss is not None:
            assert config.param_match_coeff is not None
            loss = loss + config.param_match_coeff * param_match_loss.mean()
        if config.out_recon_coeff is not None:
            loss = loss + config.out_recon_coeff * out_recon_loss.mean()
        if lp_sparsity_loss is not None:
            assert config.lp_sparsity_coeff is not None
            loss = loss + config.lp_sparsity_coeff * lp_sparsity_loss.mean()
        if topk_recon_loss is not None:
            assert config.topk_recon_coeff is not None
            loss = loss + config.topk_recon_coeff * topk_recon_loss.mean()
        if topk_l2_loss is not None:
            assert config.topk_l2_coeff is not None
            loss = loss + config.topk_l2_coeff * topk_l2_loss.mean()
        if act_recon_loss is not None:
            assert config.act_recon_coeff is not None
            loss = loss + config.act_recon_coeff * act_recon_loss.mean()
        if schatten_loss is not None:
            assert config.schatten_coeff is not None
            loss = loss + config.schatten_coeff * schatten_loss.mean()

        # Logging
        if step % config.print_freq == 0:
            # If using multiple instances, print the losses as tensors in new lines
            nl = "\n" if has_instance_dim else " "
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            if config.pnorm is not None:
                tqdm.write(f"Current pnorm:{nl}{config.pnorm}")
            if lp_sparsity_loss is not None:
                tqdm.write(f"LP sparsity loss:{nl}{lp_sparsity_loss}")
            if topk_recon_loss is not None:
                tqdm.write(f"Topk recon loss:{nl}{topk_recon_loss}")
            tqdm.write(f"Out recon loss:{nl}{out_recon_loss}")
            if topk_l2_loss is not None:
                tqdm.write(f"topk l2 loss:{nl}{topk_l2_loss}")
            if param_match_loss is not None:
                tqdm.write(f"Param match loss:{nl}{param_match_loss}")
            if act_recon_loss is not None:
                tqdm.write(f"Act recon loss:{nl}{act_recon_loss}")
            if schatten_loss is not None:
                tqdm.write(f"Schatten loss:{nl}{schatten_loss}")
            if config.wandb_project:
                wandb.log(
                    {
                        "pnorm": config.pnorm,
                        "lr": step_lr,
                        "total_loss": loss.mean().item(),
                        "lp_sparsity_loss": lp_sparsity_loss.mean().item()
                        if lp_sparsity_loss is not None
                        else None,
                        "topk_recon_loss": topk_recon_loss.mean().item()
                        if topk_recon_loss is not None
                        else None,
                        "recon_loss": out_recon_loss.mean().item(),
                        "param_match_loss": param_match_loss.mean().item()
                        if param_match_loss is not None
                        else None,
                        "topk_l2_loss": topk_l2_loss.mean().item()
                        if topk_l2_loss is not None
                        else None,
                        "act_recon_loss": act_recon_loss.mean().item()
                        if act_recon_loss is not None
                        else None,
                        "schatten_loss": schatten_loss.mean().item()
                        if schatten_loss is not None
                        else None,
                    },
                    step=step,
                )

        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
        ):
            fig_dict = plot_results_fn(
                model=model,
                target_model=pretrained_model,
                step=step,
                out_dir=out_dir,
                device=device,
                config=config,
                topk_mask=topk_mask,
                pre_acts=pre_acts,
                batch=batch,
            )
            if config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )

        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"spd_model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'spd_model_{step}.pth'}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"spd_model_{step}.pth"), base_path=out_dir, policy="now")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward()

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                assert isinstance(
                    model, SPDRankPenaltyModel
                ), "Can only norm matrices in SPDRankPenaltyModel instances"
                model.fix_normalized_adam_gradients()

            opt.step()
