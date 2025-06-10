from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent
from spd.utils import calc_kl_divergence_lm


def calc_embedding_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    component: EmbeddingComponent,
    masks: list[dict[str, Float[Tensor, "... C"]]],
    embed_module_name: str,
    unembed: bool = False,
) -> Float[Tensor, ""]:
    """
    Reconstruction loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponent``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the APD-augmented embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the APD-augmented embedding output
    and the target embedding output is used as the loss.
    """

    # --- original embedding output --------------------------------------------------------- #
    orig_module = model.model.get_submodule(embed_module_name)
    assert isinstance(orig_module, nn.Embedding), (
        f"Module {embed_module_name} expected to be nn.Embedding, got {type(orig_module)}"
    )
    target_out: Float[Tensor, "... d_emb"] = orig_module(batch)

    # --- APD-augmented embedding output ---------------------------------------------------- #
    loss = torch.tensor(0.0, device=component.A.device)
    for mask_info in masks:
        component.mask = mask_info[embed_module_name]

        apd_out: Float[Tensor, "... d_emb"] = component(batch)  # type: ignore[arg-type]
        component.mask = None

        if unembed:
            assert hasattr(model.model, "lm_head"), "Only supports unembedding named lm_head"
            target_out_unembed = model.model.lm_head(target_out)
            apd_out_unembed = model.model.lm_head(apd_out)
            loss += calc_kl_divergence_lm(pred=apd_out_unembed, target=target_out_unembed)
        else:
            loss += ((apd_out - target_out) ** 2).sum(dim=-1).mean()

    loss /= len(masks)

    return loss


def calc_schatten_loss(
    sparsity_masks: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    components: dict[str, LinearComponent | EmbeddingComponent],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the Schatten loss on the active components.

    The Schatten loss is calculated as:
        L = Σ_{components} mean(sparsity_mask^pnorm · (||A||_2^2 + ||B||_2^2))

    where:
        - sparsity_mask is the activation mask for each component
        - pnorm is the power to raise the mask to
        - A and B are the component matrices
        - ||·||_2 is the L2 norm

    Args:
        sparsity_masks: Dictionary of sparsity masks for each layer.
        pnorm: The pnorm to use for the sparsity loss. Must be positive.
        components: Dictionary of components for each layer.
        device: The device to compute the loss on.

    Returns:
        The Schatten loss as a scalar tensor.
    """

    total_loss = torch.tensor(0.0, device=device)
    for component_name, component in components.items():
        A_norms = component.A.square().sum(dim=-2)
        B_norms = component.B.square().sum(dim=-1)
        schatten_norms = A_norms + B_norms
        loss = einops.einsum(
            sparsity_masks[component_name] ** pnorm, schatten_norms, "... C, C -> ..."
        )
        total_loss += loss.mean()
    return total_loss


def calc_importance_loss(
    sparsity_masks: dict[str, Float[Tensor, "... C"]], pnorm: float
) -> Float[Tensor, ""]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        sparsity_masks: Dictionary of sparsity masks for each layer.
        pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss.
    """
    total_loss = torch.zeros_like(next(iter(sparsity_masks.values())))

    for layer_sparsity_mask in sparsity_masks.values():
        # Note, the paper uses an absolute value but our layer_sparsity_mask is already > 0
        total_loss = total_loss + layer_sparsity_mask**pnorm

    # Sum over the C dimension and mean over the other dimensions
    return total_loss.sum(dim=-1).mean()


def calc_layerwise_ci_masked_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    device: str,
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"] = "kl",
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            modified_out = model.forward_with_components(
                batch,
                components={component_name: component},
                masks={component_name: mask_info[component_name]},
            )
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            elif loss_type == "kl":
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))


def calc_ci_masked_recon_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_mdoel_out"],
    loss_type: Literal["mse", "kl"] = "mse",
) -> Float[Tensor, ""]:
    """Calculate the MSE over all masks."""
    # Do a forward pass with all components
    out_masked_random_mask = model.forward_with_components(
        batch, components=components, masks=masks
    )
    if loss_type == "mse":
        loss = ((out_masked_random_mask - target_out) ** 2).mean()
    elif loss_type == "kl":
        loss = calc_kl_divergence_lm(pred=out_masked_random_mask, target=target_out)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def _calc_tensors_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    faithfulness_loss = torch.tensor(0.0, device=device)
    for name in params1:
        faithfulness_loss = faithfulness_loss + ((params2[name] - params1[name]) ** 2).sum()
    return faithfulness_loss / n_params


def calc_faithfulness_loss(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (A@B + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)
        assert isinstance(submodule, nn.Linear | nn.Embedding)
        target_params[comp_name] = submodule.weight
        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss = _calc_tensors_mse(
        params1=component_params,
        params2=target_params,
        n_params=n_params,
        device=device,
    )
    return faithfulness_loss


def calc_ce_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "..."]],
    unmasked_component_logits: Float[Tensor, "..."],
    masked_component_logits: Float[Tensor, "..."],
    target_logits: Float[Tensor, "..."],
) -> dict[str, float]:
    """Calculate cross-entropy losses for various masking scenarios.

    Args:
        model: The component model
        batch: Input batch
        components: Dictionary of components
        masks: Dictionary of masks for components
        unmasked_component_logits: Logits from unmasked components
        masked_component_logits: Logits from masked components
        target_logits: Target model logits

    Returns:
        Dictionary containing CE losses for different scenarios
    """
    ce_losses = {}

    # Flatten logits and batch for CE calculation
    flat_all_component_logits = einops.rearrange(
        unmasked_component_logits, "... vocab -> (...) vocab"
    )
    flat_masked_component_logits = einops.rearrange(
        masked_component_logits, "... vocab -> (...) vocab"
    )
    flat_batch = batch.flatten()

    # CE vs true labels
    unmasked_ce_loss = F.cross_entropy(input=flat_all_component_logits[:-1], target=flat_batch[1:])
    masked_ce_loss = F.cross_entropy(input=flat_masked_component_logits[:-1], target=flat_batch[1:])

    flat_target_logits = einops.rearrange(target_logits, "... vocab -> (...) vocab")
    target_ce_loss = F.cross_entropy(input=flat_target_logits[:-1], target=flat_batch[1:])

    # CE when every component is fully masked (all-zero masks)
    zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
    zero_masked_component_logits = model.forward_with_components(
        batch, components=components, masks=zero_masks
    )
    flat_zero_masked_component_logits = einops.rearrange(
        zero_masked_component_logits, "... vocab -> (...) vocab"
    )
    zero_masked_ce_loss = F.cross_entropy(
        input=flat_zero_masked_component_logits[:-1], target=flat_batch[1:]
    )

    ce_losses["misc/unmasked_ce_loss_vs_labels"] = unmasked_ce_loss.item()
    ce_losses["misc/masked_ce_loss_vs_labels"] = masked_ce_loss.item()
    ce_losses["misc/target_ce_loss_vs_labels"] = target_ce_loss.item()
    ce_losses["misc/zero_masked_ce_loss_vs_labels"] = zero_masked_ce_loss.item()

    return ce_losses
