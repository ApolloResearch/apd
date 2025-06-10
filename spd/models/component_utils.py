import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.utils import extract_batch_data


def calc_masks(
    gates: dict[str, Gate | GateMLP],
    target_component_acts: dict[str, Float[Tensor, "batch C"]],
    detach_inputs: bool = False,
) -> tuple[
    dict[str, Float[Tensor, "batch C"]],
    dict[str, Float[Tensor, "batch C"]],
]:
    """Calculate the mask for the SPD model.

    Args:
        gates: The gates to use for the mask.
        component_acts: The activations after each subnetwork in the SPD model.
        detach_inputs: Whether to detach the inputs to the gates.
    Returns:
        Tuple of (masks, sparsity_masks) dictionaries for each layer.
    """
    masks = {}
    sparsity_masks = {}
    for layer_name in gates:
        gate_input = target_component_acts[layer_name]
        if detach_inputs:
            gate_input = gate_input.detach()
        masks[layer_name] = gates[layer_name].forward(gate_input)
        sparsity_masks[layer_name] = gates[layer_name].forward_unclamped(gate_input)
    return masks, sparsity_masks


def calc_random_masks(
    masks: dict[str, Float[Tensor, "batch C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "batch C"]]]:
    """Calculate n_mask_samples masks with the formula `mask + (1 - mask) * rand_unif(0,1)`.

    Args:
        masks: The masks to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {
                layer_name: mask + (1 - mask) * torch.rand_like(mask)
                for layer_name, mask in masks.items()
            }
        )
    return stochastic_masks


def calc_component_acts(
    pre_weight_acts: dict[str, Float[Tensor, "batch d_in"] | Int[Tensor, "batch pos"]],
    As: dict[str, Float[Tensor, "d_in C"]],
) -> dict[str, Float[Tensor, "batch C"]]:
    """Calculate the component acts for each layer. I.e. (pre_weight_acts @ A).

    Args:
        pre_weight_acts: The activations before each layer in the target model.
        As: The A matrix at each layer.
    """
    component_acts = {}
    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]
        if not acts.dtype.is_floating_point:
            # Embedding layer
            component_acts[param_name] = As[param_name][acts]
        else:
            # Linear layer
            component_acts[param_name] = einops.einsum(
                acts, As[param_name], "... d_in, d_in C -> ... C"
            )
    return component_acts


def calc_mask_l_zero(
    masks: dict[str, Float[Tensor, "... C"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the masks, summed over the C dimension."""
    mask_l_zero = {}
    for layer_name, mask in masks.items():
        mean_dims = tuple(range(mask.ndim - 1))
        mask_l_zero[layer_name] = (mask > cutoff).float().mean(dim=mean_dims).sum().item()
    return mask_l_zero


def component_activation_statistics(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    device: str,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.C, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.A for module_name, v in components.items()}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

        masks, sparsity_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            detach_inputs=False,
        )
        for module_name, mask in masks.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += mask.shape[:-1].numel()

            # Count the number of components that are active at all
            active_components = mask > 0
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(mask.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts
