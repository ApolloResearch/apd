import math

import einops
import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_component_acts, calc_masks
from spd.models.components import (
    EmbeddingComponent,
    Gate,
    GateMLP,
    LinearComponent,
)


def permute_to_identity(
    mask: Float[Tensor, "batch m"],
) -> tuple[Float[Tensor, "batch m"], Float[Tensor, " m"]]:
    """Returns (permuted_mask, permutation_indices)."""

    original_shape = mask.shape
    if mask.ndim == 2:
        # Add instance dimension: (batch, m) -> (batch, 1, m)
        mask = mask.unsqueeze(1)
        batch, n_instances, m = mask.shape
        assert n_instances == 1
    elif mask.ndim == 3:
        batch, n_instances, m = mask.shape
    else:
        raise ValueError(f"Mask must have 2 or 3 dimensions, got {mask.ndim}")

    new_mask = mask.clone()
    effective_rows = min(batch, m)
    # Store permutation indices for each instance
    perm_indices = torch.zeros((n_instances, m), dtype=torch.long, device=mask.device)

    for inst in range(n_instances):
        mat: Tensor = mask[:, inst, :]
        perm: list[int] = [0] * m
        used: set[int] = set()
        for i in range(effective_rows):
            sorted_indices: list[int] = torch.argsort(mat[i, :], descending=True).tolist()
            chosen: int = next(
                (col for col in sorted_indices if col not in used), sorted_indices[0]
            )
            perm[i] = chosen
            used.add(chosen)
        remaining: list[int] = sorted(list(set(range(m)) - used))
        for idx, col in enumerate(remaining):
            perm[effective_rows + idx] = col
        new_mask[:, inst, :] = mat[:, perm]
        perm_indices[inst] = torch.tensor(perm, device=mask.device)

    # Return in original shape
    if len(original_shape) == 2:
        # Remove instance dimension: (batch, 1, m) -> (batch, m)
        new_mask = new_mask.squeeze(1)
        perm_indices = perm_indices.squeeze(0)  # (1, m) -> (m)

    return new_mask, perm_indices


def _plot_mask_figure(
    masks: dict[str, Float[Tensor, "batch m"]],
    title_suffix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
) -> plt.Figure:
    """Helper function to plot a single mask figure.

    Args:
        masks: Dictionary of masks to plot
        title_suffix: String to append to titles (e.g., "masks" or "sparsity masks")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension

    Returns:
        The matplotlib figure
    """
    fig, axs = plt.subplots(
        len(masks),
        1,
        figsize=(5, 5 * len(masks)),
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(masks.items()):
        # mask has shape (batch, m) or (batch, pos, m)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        im = axs[j, 0].matshow(mask_data, aspect="auto", cmap=colormap)
        images.append(im)

        # Move x-axis ticks to bottom
        axs[j, 0].xaxis.tick_bottom()
        axs[j, 0].xaxis.set_label_position("bottom")
        axs[j, 0].set_xlabel("Subcomponent index")
        axs[j, 0].set_ylabel("Input feature index")
        axs[j, 0].set_title(f"{mask_name} ({title_suffix})")

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in masks.values()),
        vmax=max(mask.max().item() for mask in masks.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Capitalize first letter of title suffix for the figure title
    fig.suptitle(f"{title_suffix.capitalize()} - Input magnitude: {input_magnitude}")

    return fig


def plot_mask_vals(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    plot_regular_masks: bool = True,
) -> tuple[dict[str, plt.Figure], dict[str, Float[Tensor, " m"]]]:
    """Plot the values of the mask for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features
        plot_regular_masks: Whether to plot the regular masks (blue plots)

    Returns:
        Tuple of:
            - Dictionary of figures with keys 'masks' (if plot_regular_masks=True) and 'sparsity_masks'
            - Dictionary of permutation indices for sparsity masks
    """
    # First, create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only plot the mask of the first pos dim
        batch = batch.unsqueeze(1)

    # Get mask values
    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=list(components.keys())
    )[1]
    As = {module_name: v.A for module_name, v in components.items()}

    target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

    masks_raw, sparsity_masks_raw = calc_masks(
        gates=gates,
        target_component_acts=target_component_acts,
        detach_inputs=False,
    )

    # Permute both mask types with their own optimal permutations
    masks = {}
    sparsity_masks = {}
    all_perm_indices_sparsity_masks = {}

    for k in masks_raw:
        # Compute optimal permutation for regular masks
        masks[k], _ = permute_to_identity(mask=masks_raw[k])
        # Compute optimal permutation for sparsity masks
        sparsity_masks[k], all_perm_indices_sparsity_masks[k] = permute_to_identity(
            mask=sparsity_masks_raw[k]
        )

    # Create figures dictionary
    figures = {}

    # Create masks figure only if requested
    if plot_regular_masks:
        masks_fig = _plot_mask_figure(
            masks=masks,
            title_suffix="masks",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
        )
        figures["masks"] = masks_fig

    # Always create sparsity masks figure
    sparsity_masks_fig = _plot_mask_figure(
        masks=sparsity_masks,
        title_suffix="sparsity masks",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
    )
    figures["sparsity_masks"] = sparsity_masks_fig

    return figures, all_perm_indices_sparsity_masks


def plot_subnetwork_attributions_statistics(
    mask: Float[Tensor, "batch_size n_instances m"],
) -> dict[str, plt.Figure]:
    """Plot vertical bar charts of the number of active subnetworks over the batch for each instance."""
    batch_size = mask.shape[0]
    if mask.ndim == 2:
        n_instances = 1
        mask = einops.repeat(mask, "batch m -> batch n_instances m", n_instances=1)
    else:
        n_instances = mask.shape[1]

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    for i, ax in enumerate(axs):
        values = mask[:, i].sum(dim=1).cpu().detach().numpy()
        bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
        counts, _ = np.histogram(values, bins=bins)
        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels([str(b) for b in bins[:-1]])

        # Only add y-label to first subplot
        if i == 0:
            ax.set_ylabel("Count")

        ax.set_xlabel("Number of active subnetworks")
        ax.set_title(f"Instance {i + 1}")

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_AB_matrices(
    components: dict[str, LinearComponent | EmbeddingComponent],
    all_perm_indices: dict[str, Float[Tensor, "n_instances m"]] | None = None,
) -> plt.Figure:
    """Plot A and B matrices for each instance, grouped by layer."""
    As = {k: v.A for k, v in components.items()}
    Bs = {k: v.B for k, v in components.items()}

    n_layers = len(As)

    # Create figure for plotting - 2 rows per layer (A and B)
    fig, axs = plt.subplots(
        2 * n_layers,
        1,
        figsize=(5, 5 * 2 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot A and B matrices for each layer
    for j, name in enumerate(sorted(As.keys())):
        # Plot A matrix
        A_data = As[name]
        if all_perm_indices is not None:
            A_data = A_data[:, all_perm_indices[name]]
        A_data = A_data.detach().cpu().numpy()
        im = axs[2 * j, 0].matshow(A_data, aspect="auto", cmap="coolwarm")
        axs[2 * j, 0].set_ylabel("d_in index")
        axs[2 * j, 0].set_xlabel("Component index")
        axs[2 * j, 0].set_title(f"{name} (A matrix)")
        images.append(im)

        # Plot B matrix
        B_data = Bs[name]
        if all_perm_indices is not None:
            B_data = B_data[all_perm_indices[name], :]
        B_data = B_data.detach().cpu().numpy()
        im = axs[2 * j + 1, 0].matshow(B_data, aspect="auto", cmap="coolwarm")
        axs[2 * j + 1, 0].set_ylabel("Component index")
        axs[2 * j + 1, 0].set_xlabel("d_out index")
        axs[2 * j + 1, 0].set_title(f"{name} (B matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = list(As.values()) + list(Bs.values())
    norm = plt.Normalize(
        vmin=min(M.min().item() for M in all_matrices),
        vmax=max(M.max().item() for M in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return fig


def create_embed_mask_sample_table(
    masks: dict[str, Float[Tensor, "... m"]],
) -> wandb.Table | None:
    """Create a wandb table visualizing embedding mask values.

    Args:
        masks: Dictionary of masks for each component.

    Returns:
        A wandb Table object or None if transformer.wte not in masks.
    """
    if "transformer.wte" not in masks:
        return None

    # Create a 20x10 table for wandb
    table_data = []
    # Add "Row Name" as the first column
    component_names = ["TokenSample"] + ["CompVal" for _ in range(10)]

    for i, ma in enumerate(masks["transformer.wte"][0, :20]):
        active_values = ma[ma > 0.1].tolist()
        # Cap at 10 components
        active_values = active_values[:10]
        formatted_values = [f"{val:.2f}" for val in active_values]
        # Pad with empty strings if fewer than 10 components
        while len(formatted_values) < 10:
            formatted_values.append("0")
        # Add row name as the first element
        table_data.append([f"{i}"] + formatted_values)

    return wandb.Table(data=table_data, columns=component_names)


def plot_mean_component_activation_counts(
    mean_component_activation_counts: dict[str, Float[Tensor, " m"]],
) -> plt.Figure:
    """Plots the mean activation counts for each component module in a grid."""
    n_modules = len(mean_component_activation_counts)
    max_cols = 6
    n_cols = min(n_modules, max_cols)
    # Calculate the number of rows needed, rounding up
    n_rows = math.ceil(n_modules / n_cols)

    # Create a figure with the calculated number of rows and columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    # Ensure axs is always a 2D array for consistent indexing, even if n_modules is 1
    axs = axs.flatten()  # Flatten the axes array for easy iteration

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, counts) in enumerate(mean_component_activation_counts.items()):
        ax = axs[i]
        ax.hist(counts.detach().cpu().numpy(), bins=100)
        ax.set_yscale("log")
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Mean Activation Count")
        ax.set_ylabel("Frequency")

    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_modules, n_rows * n_cols):
        axs[i].axis("off")

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()

    return fig


def plot_mask_histograms(
    masks: dict[str, Float[Tensor, "... m"]],
    bins: int = 100,
) -> dict[str, plt.Figure]:
    """Plot histograms of mask values for each layer.

    Args:
        masks: Dictionary of masks for each component.
        bins: Number of bins for the histogram.

    Returns:
        Dictionary mapping layer names to histogram figures.
    """
    fig_dict = {}

    for layer_name, layer_mask in masks.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(layer_mask.flatten().cpu().numpy(), bins=bins)
        ax.set_title(f"Mask values for {layer_name}")
        ax.set_xlabel("Mask value")
        # Use a log scale
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

        fig_dict[f"mask_vals_{layer_name}"] = fig

    return fig_dict
