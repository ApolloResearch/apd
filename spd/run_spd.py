"""Run SPD on a model."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.models.components import Gate, GateMLP, Linear, LinearComponent
from spd.module_utils import collect_nested_module_attrs, get_nested_module_attr
from spd.utils import calc_recon_mse, get_lr_schedule_fn, get_lr_with_warmup


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.masked_recon_coeff is not None:
        run_suffix += f"maskrecon{config.masked_recon_coeff:.2e}_"
        run_suffix += f"nrandmasks{config.n_random_masks}_"
    if config.act_recon_coeff is not None:
        run_suffix += f"actrecon_{config.act_recon_coeff:.2e}_"
    if config.random_mask_recon_coeff is not None:
        run_suffix += f"randrecon{config.random_mask_recon_coeff:.2e}_"
    run_suffix += f"p{config.pnorm:.2e}_"
    run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    run_suffix += f"m{config.m}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"attr-{config.attribution_type[:3]}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def _calc_param_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1:
        param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(
            dim=(-2, -1)
        )
    return param_match_loss / n_params


def calc_param_match_loss(
    param_names: list[str],
    target_model: HookedRootModule,
    spd_model: SPDModel,
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between the target model weights and the SPD model weights.

    Args:
        param_names: The names of the parameters to be matched.
        target_model: The target model to match.
        spd_model: The SPD model to match.
        n_params: The number of parameters in the model. Used for normalization.
        device: The device to use for calculations.
    """
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight")
    return _calc_param_mse(
        params1=target_params,
        params2=spd_params,
        n_params=n_params,
        device=device,
    )


def calc_lp_sparsity_loss(
    relud_masks: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
    pnorm: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        relud_masks: Dictionary of relu masks for each layer.
        pnorm: The pnorm to use for the sparsity loss.
        eps: A small epsilon to avoid division by zero when calculating gradients.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    # Initialize with zeros matching the shape of first mask
    total_loss = torch.zeros_like(next(iter(relud_masks.values())))

    for layer_relud_mask in relud_masks.values():
        total_loss = total_loss + (layer_relud_mask.abs() + eps).pow(pnorm)

    # Sum over the m dimension and mean over the batch dimension
    return total_loss.sum(dim=-1).mean(dim=0)


def calc_act_recon_mse(
    acts1: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
    acts2: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """MSE between each entry in acts1 and acts2.
    Returns:
        The activation reconstruction loss. Will have an n_instances dimension if the model has an
            n_instances dimension, otherwise a scalar.
    """
    assert acts1.keys() == acts2.keys(), f"Key mismatch: {acts1.keys()} != {acts2.keys()}"

    device = next(iter(acts1.values())).device
    m = next(iter(acts1.values())).shape[-1]

    loss = torch.zeros(1, device=device)
    for layer_name in acts1:
        loss = loss + ((acts1[layer_name] - acts2[layer_name]) ** 2).sum(dim=-1)

    # Normalize by the total number of output dimensions and mean over the batch dim
    return (loss / (m * len(acts1))).mean(dim=0)


def calc_masks(
    gates: dict[str, Gate | GateMLP],
    target_component_acts: dict[
        str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ],
    attributions: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]
    | None = None,
    detach_inputs: bool = False,
) -> tuple[
    dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
    dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
]:
    """Calculate the mask for the SPD model.

    TODO: Use attributions in our gate calculation too.

    Args:
        gates: The gates to use for the mask.
        component_acts: The activations after each subnetwork in the SPD model.
        attributions: The attributions to use for the mask.
        detach_inputs: Whether to detach the inputs to the gates.
    Returns:
        Dictionary of masks for each layer.
    """
    masks = {}
    relud_masks = {}
    for layer_name in gates:
        gate_input = target_component_acts[layer_name]
        if detach_inputs:
            gate_input = gate_input.detach()
        masks[layer_name] = gates[layer_name].forward(gate_input)
        relud_masks[layer_name] = gates[layer_name].forward_unclamped(gate_input)
    return masks, relud_masks


def calc_random_masks(
    masks: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
    n_random_masks: int,
) -> list[dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]]:
    """Calculate n_random_masks random masks with the formula `mask + (1 - mask) * rand_unif(0,1)`.

    Args:
        masks: The masks to use for the random masks.
        n_random_masks: The number of random masks to calculate.

    Return:
        A list of n_random_masks dictionaries, each containing the random masks for each layer.
    """
    random_masks = []
    for _ in range(n_random_masks):
        random_masks.append(
            {
                layer_name: mask + (1 - mask) * torch.rand_like(mask)
                for layer_name, mask in masks.items()
            }
        )
    return random_masks


def calc_random_masks_mse_loss(
    model: SPDModel,
    batch: Float[Tensor, "batch n_instances d_in"],
    random_masks: list[dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]],
    out_masked: Float[Tensor, "batch n_instances d_out"],
    has_instance_dim: bool,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE over all random masks."""
    loss = torch.tensor(0.0, device=out_masked.device)
    for i in range(len(random_masks)):
        out_masked_random_mask = model(batch, masks=random_masks[i])
        loss = loss + calc_recon_mse(out_masked, out_masked_random_mask, has_instance_dim)

    return loss / len(random_masks)


def calc_component_acts(
    pre_weight_acts: dict[
        str,
        Float[Tensor, "batch n_instances d_in"]
        | Float[Tensor, "batch d_in"]
        | Int[Tensor, "batch pos"],
    ],
    As: dict[str, Float[Tensor, "d_in m"] | Float[Tensor, "n_instances d_in m"]],
) -> dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]:
    """Calculate the component acts for each layer. I.e. (pre_weight_acts @ A).

    Args:
        pre_weight_acts: The activations before each layer in the target model.
        As: The A matrix at each layer.
    """
    component_acts = {}
    for param_name in pre_weight_acts:
        raw_name = param_name.removesuffix(".hook_pre")
        acts = pre_weight_acts[param_name]
        if not acts.dtype.is_floating_point:
            # Embedding layer
            component_acts[raw_name] = As[raw_name][acts]
        else:
            # Linear layer
            component_acts[raw_name] = einops.einsum(
                acts, As[raw_name], "... d_in, ... d_in m -> ... m"
            )
    return component_acts


def calc_masked_target_component_acts(
    pre_weight_acts: dict[
        str, Float[Tensor, "batch n_instances d_in"] | Float[Tensor, "batch d_in"]
    ],
    As: dict[str, Float[Tensor, "d_in m"] | Float[Tensor, "n_instances d_in m"]],
    masks: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
) -> dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]:
    """Calculate the masked target component acts for each layer."""
    masked_target_component_acts = {}
    for param_name in pre_weight_acts:
        raw_name = param_name.removesuffix(".hook_pre")
        masked_As = einops.einsum(
            As[raw_name], masks[raw_name], "... d_in m, batch ... m -> batch ... d_in m"
        )
        acts = pre_weight_acts[param_name]
        if not acts.dtype.is_floating_point:
            masked_target_component_acts[raw_name] = masked_As[acts]
        else:
            masked_target_component_acts[raw_name] = einops.einsum(
                acts,
                masked_As,
                "batch ... d_in, batch ... d_in m -> batch ... m",
            )
    return masked_target_component_acts


def calc_layerwise_recon_loss(
    param_names: list[str],
    target_model: HookedRootModule,
    spd_model: SPDModel,
    batch: Float[Tensor, "batch n_instances d_in"] | Float[Tensor, "batch d_in"],
    device: str,
    masks: list[dict[str, Float[Tensor, "batch n_instances m"] | Float[Tensor, "batch m"]]],
    target_out: Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"],
    has_instance_dim: bool,
) -> Float[Tensor, ""]:
    """Calculate the layerwise activation reconstruction loss using regular PyTorch hooks.

    Note that we support multiple masks for the case of calculating this loss over a list of random
    masks.
    """
    total_loss = torch.tensor(0.0, device=device)

    for mask in masks:
        for param_name in param_names:
            target_module = get_nested_module_attr(target_model, param_name)
            assert isinstance(target_module, Linear)

            component_module = get_nested_module_attr(spd_model, param_name)
            assert isinstance(component_module, LinearComponent)

            def hook(
                module: nn.Module,
                input: tuple[
                    Float[Tensor, "batch n_instances d_in"] | Float[Tensor, "batch d_in"], ...
                ],
                output: Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"],
                param_name: str,
                mask: dict[str, Float[Tensor, "batch n_instances m"] | Float[Tensor, "batch m"]],
                component_module: LinearComponent,
            ) -> Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]:
                linear_output = component_module(input[0], mask=mask[param_name])
                return linear_output

            handle = target_module.register_forward_hook(
                partial(hook, param_name=param_name, mask=mask, component_module=component_module)
            )
            modified_output = target_model(batch)
            handle.remove()

            mse_loss = calc_recon_mse(modified_output, target_out, has_instance_dim)
            total_loss = total_loss + mse_loss

    return total_loss / (len(param_names) * len(masks))


def init_As_and_Bs_(model: SPDModel, target_model: HookedRootModule) -> None:
    """Initialize the A and B matrices using a scale factor from the target weights."""
    As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)
    Bs = collect_nested_module_attrs(model, attr_name="B", include_attr_name=False)
    for param_name in As:
        A = As[param_name]  # (..., d_in, m)
        B = Bs[param_name]  # (..., m, d_out)
        target_weight = get_nested_module_attr(
            target_model, param_name + ".weight"
        )  # (..., d_in, d_out)

        # Make A and B have unit norm in the d_in and d_out dimensions
        A.data[:] = torch.randn_like(A.data)
        B.data[:] = torch.randn_like(B.data)
        A.data[:] = A.data / A.data.norm(dim=-2, keepdim=True)
        B.data[:] = B.data / B.data.norm(dim=-1, keepdim=True)

        m_norms = einops.einsum(
            A, B, target_weight, "... d_in m, ... m d_out, ... d_in d_out -> ... m"
        )
        # Scale B by m_norms. We leave A as is since this may get scaled with the unit_norm_matrices
        # config options.
        B.data[:] = B.data * m_norms.unsqueeze(-1)


def calc_mask_l_zero(
    masks: dict[str, Float[Tensor, "batch n_instances m"] | Float[Tensor, "batch m"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the masks, summed over the m dimension."""
    mask_l_zero = {}
    for layer_name, mask in masks.items():
        mean_dims = tuple(range(mask.ndim - 1))
        mask_l_zero[layer_name] = (mask > cutoff).float().mean(dim=mean_dims).sum().item()
    return mask_l_zero


def optimize(
    model: SPDModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    target_model: HookedRootModule,
    param_names: list[str],
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)
    target_model.to(device=device)

    init_As_and_Bs_(model=model, target_model=target_model)

    has_instance_dim = hasattr(model, "n_instances")

    # We used "-" instead of "." as module names can't have "." in them
    gates = {k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()}

    # Note that we expect weight decay to be problematic for spd models
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    n_params = 0
    for param_name in param_names:
        weight = get_nested_module_attr(target_model, param_name + ".weight")
        n_params += weight.numel()

    if has_instance_dim:
        # All subnetwork param have an n_instances dimension
        n_params = n_params / model.n_instances

    epoch = 0
    total_samples = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
            model.set_As_to_unit_norm()

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
            batch = next(data_iter)[0]  # Ignore labels here, we use the output of target_model
        except StopIteration:
            tqdm.write(f"Epoch {epoch} finished, starting new epoch")
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]

        batch = batch.to(device=device)
        total_samples += batch.shape[0]

        # Forward pass with target model
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            batch, names_filter=target_cache_filter
        )

        # Forward pass with all subnetworks
        out = model(batch)

        pre_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_pre")}
        As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)
        # attributions = calc_grad_attributions(
        #     target_out=target_out,
        #     pre_weight_acts=pre_weight_acts,
        #     post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        #     target_component_acts=target_component_acts,
        #     Bs=collect_nested_module_attrs(model, attr_name="B", include_attr_name=False),
        # )
        attributions = None

        masks, relud_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=attributions,
            detach_inputs=False,
        )

        # Masked forward pass
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out_masked, spd_cache_masked = model.run_with_cache(
            batch, names_filter=spd_cache_filter, masks=masks
        )

        random_masks_loss = None
        if config.random_mask_recon_coeff is not None:
            random_masks = calc_random_masks(masks=masks, n_random_masks=config.n_random_masks)
            random_masks_loss = calc_random_masks_mse_loss(
                model=model,
                batch=batch,
                random_masks=random_masks,
                out_masked=target_out,
                has_instance_dim=has_instance_dim,
            )

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, target_out, has_instance_dim)

        param_match_loss = calc_param_match_loss(
            param_names=param_names,
            target_model=target_model,
            spd_model=model,
            n_params=n_params,
            device=device,
        )

        lp_sparsity_loss = calc_lp_sparsity_loss(relud_masks=relud_masks, pnorm=config.pnorm)

        masked_recon_loss = calc_recon_mse(out_masked, target_out, has_instance_dim)

        act_recon_loss = None
        if config.act_recon_coeff is not None:
            masked_spd_component_acts = {
                k.removesuffix(".hook_component_acts"): v
                for k, v in spd_cache_masked.items()
                if k.endswith("hook_component_acts")
            }
            masked_target_component_acts = calc_masked_target_component_acts(
                pre_weight_acts=pre_weight_acts, As=As, masks=masks
            )
            act_recon_loss = calc_act_recon_mse(
                masked_spd_component_acts, masked_target_component_acts
            )

        layerwise_recon_loss = None
        if config.layerwise_recon_coeff is not None:
            layerwise_recon_loss = calc_layerwise_recon_loss(
                param_names=param_names,
                target_model=target_model,
                spd_model=model,
                batch=batch,
                device=device,
                masks=[masks],
                target_out=target_out,
                has_instance_dim=has_instance_dim,
            )

        layerwise_random_recon_loss = None
        if config.layerwise_random_recon_coeff is not None:
            layerwise_random_masks = calc_random_masks(
                masks=masks, n_random_masks=config.n_random_masks
            )
            layerwise_random_recon_loss = calc_layerwise_recon_loss(
                param_names=param_names,
                target_model=target_model,
                spd_model=model,
                batch=batch,
                device=device,
                masks=layerwise_random_masks,
                target_out=target_out,
                has_instance_dim=has_instance_dim,
            )

        loss_terms = {
            "param_match_loss": (param_match_loss, config.param_match_coeff),
            "out_recon_loss": (out_recon_loss, config.out_recon_coeff),
            "lp_sparsity_loss": (lp_sparsity_loss, config.lp_sparsity_coeff),
            "masked_recon_loss": (masked_recon_loss, config.masked_recon_coeff),
            "act_recon_loss": (act_recon_loss, config.act_recon_coeff),
            "random_masks_loss": (random_masks_loss, config.random_mask_recon_coeff),
            "layerwise_recon_loss": (layerwise_recon_loss, config.layerwise_recon_coeff),
            "layerwise_random_recon_loss": (
                layerwise_random_recon_loss,
                config.layerwise_random_recon_coeff,
            ),
        }
        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        for loss_name, (loss_term, coeff) in loss_terms.items():
            if coeff is not None:
                assert loss_term is not None, f"{loss_name} is None but coeff is not"
                loss = loss + coeff * loss_term.mean()  # Mean over n_instances dimension

        # Logging
        if step % config.print_freq == 0:
            mask_l_zero = calc_mask_l_zero(masks=masks)
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            tqdm.write(f"lr: {step_lr}")
            for loss_name, (val, _) in loss_terms.items():
                if val is not None:
                    val_repr = f"\n{val.tolist()}" if val.numel() > 1 else f" {val.item()}"
                    tqdm.write(f"{loss_name}:{val_repr}")

            if config.wandb_project:
                metrics = {
                    "pnorm": config.pnorm,
                    "lr": step_lr,
                    "total_loss": loss.item(),
                    **{"mask_l0_" + k: v for k, v in mask_l_zero.items()},
                    **{
                        name: val.mean().item() if val is not None else None
                        for name, (val, _) in loss_terms.items()
                    },
                }
                wandb.log(metrics, step=step)

        # Make plots
        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or config.image_on_first_step)
        ):
            fig_dict = plot_results_fn(
                model=model,
                target_model=target_model,
                step=step,
                out_dir=out_dir,
                device=device,
                config=config,
                masks=masks,
                gates=gates,
                batch=batch,
            )
            if config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )
                if out_dir is not None:
                    for k, v in fig_dict.items():
                        v.savefig(out_dir / f"{k}_{step}.png")
                        tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # Save model
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
            loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()

            opt.step()
