import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.experiments.lm.models import ComponentModel
from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponentWithBias
from spd.module_utils import collect_nested_module_attrs
from spd.run_spd import calc_component_acts, calc_masks


def permute_to_identity(
    mask: Float[Tensor, "batch n_instances m"] | Float[Tensor, "batch m"],
) -> tuple[
    Float[Tensor, "batch n_instances m"] | Float[Tensor, "batch m"],
    Float[Tensor, "n_instances m"] | Float[Tensor, " m"],
]:
    """Returns (permuted_mask, permutation_indices)

    Supports both (batch, m) and (batch, n_instances, m) shaped masks.
    For (batch, m) input, returns (batch, m) mask and (m,) permutation indices.
    For (batch, n_instances, m) input, returns (batch, n_instances, m) mask and (n_instances, m) permutation indices.
    """

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


def plot_mask_vals(
    model: ComponentModel,
    components: dict[str, LinearComponentWithBias | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    batch_shape: tuple[int, ...],
    device: str,
    input_magnitude: float,
) -> tuple[plt.Figure, dict[str, Float[Tensor, "n_instances m"]]]:
    """Plot the values of the mask for a batch of inputs with single active features."""
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

    relud_masks_raw = calc_masks(
        gates=gates,
        target_component_acts=target_component_acts,
        attributions=None,
        detach_inputs=False,
    )[1]

    relud_masks = {}
    all_perm_indices = {}
    for k, v in relud_masks_raw.items():
        relud_masks[k], all_perm_indices[k] = permute_to_identity(mask=v)

    # Create figure with better layout and sizing
    fig, axs = plt.subplots(
        len(relud_masks),
        1,
        figsize=(5, 5 * len(relud_masks)),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(relud_masks.items()):
        # mask has shape (batch, m) or (batch, pos, m)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        im = axs[j, 0].matshow(mask_data, aspect="auto", cmap="Reds")
        images.append(im)

        axs[j, 0].set_xlabel("Mask index")
        axs[j, 0].set_ylabel("Input feature index")
        axs[j, 0].set_title(mask_name)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in relud_masks.values()),
        vmax=max(mask.max().item() for mask in relud_masks.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Add a title which shows the input magnitude
    fig.suptitle(f"Input magnitude: {input_magnitude}")

    return fig, all_perm_indices


def plot_mask_vals_tms(
    model: SPDModel,
    target_model: HookedRootModule,
    gates: dict[str, Gate | GateMLP],
    device: str,
    input_magnitude: float,
) -> tuple[plt.Figure, dict[str, Float[Tensor, "n_instances m"]]]:
    """Plot the values of the mask for a batch of inputs with single active features."""
    # First, create a batch of inputs with single active features
    n_features = model.n_features
    n_instances = model.n_instances
    batch = torch.eye(n_features, device=device) * input_magnitude
    batch = einops.repeat(
        batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances
    )

    # Forward pass with target model
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_cache = target_model.run_with_cache(batch, names_filter=target_cache_filter)[1]
    pre_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_pre")}
    As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)

    target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)

    relud_masks_raw = calc_masks(
        gates=gates, target_component_acts=target_component_acts, attributions=None
    )[1]

    relud_masks = {}
    all_perm_indices = {}
    for k, v in relud_masks_raw.items():
        relud_masks[k], all_perm_indices[k] = permute_to_identity(mask=v)

    # Create figure with better layout and sizing
    fig, axs = plt.subplots(
        len(relud_masks),
        n_instances,
        figsize=(5 * n_instances, 5 * len(relud_masks)),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []
    for i in range(n_instances):
        axs[0, i].set_title(f"Instance {i}")
        for j, (mask_name, mask) in enumerate(relud_masks.items()):
            # mask has shape (batch, n_instances, m)
            mask_data = mask[:, i, :].detach().cpu().numpy()
            im = axs[j, i].matshow(mask_data, aspect="auto", cmap="Reds")
            images.append(im)

            axs[j, i].set_xlabel("Mask index")
            if i == 0:  # Only set ylabel for leftmost plots
                axs[j, i].set_ylabel("Input feature index")
            axs[j, i].set_title(mask_name)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in relud_masks.values()),
        vmax=max(mask.max().item() for mask in relud_masks.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Add a title which shows the input magnitude
    fig.suptitle(f"Input magnitude: {input_magnitude}")

    return fig, all_perm_indices


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
    components: dict[str, LinearComponentWithBias | EmbeddingComponent],
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


def plot_AB_matrices_tms(
    model: SPDModel,
    device: str,
    all_perm_indices: dict[str, Float[Tensor, "n_instances m"]] | None = None,
) -> plt.Figure:
    """Plot A and B matrices for each instance, grouped by layer."""
    # Collect all A and B matrices
    As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)
    Bs = collect_nested_module_attrs(model, attr_name="B", include_attr_name=False)
    n_instances = model.n_instances

    # Verify that A and B matrices have matching names
    A_names = set(As.keys())
    B_names = set(Bs.keys())
    assert A_names == B_names, (
        f"A and B matrices must have matching names. Found A: {A_names}, B: {B_names}"
    )

    n_layers = len(As)

    # Create figure for plotting - 2 rows per layer (A and B)
    fig, axs = plt.subplots(
        2 * n_layers,
        n_instances,
        figsize=(5 * n_instances, 5 * 2 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot each layer's A and B matrices for each instance
    for i in range(n_instances):
        if i == 0:
            axs[0, i].set_title(f"Instance {i}")

        # Plot A and B matrices for each layer
        for j, name in enumerate(sorted(As.keys())):
            # Plot A matrix
            A_data = As[name][i]
            if all_perm_indices is not None:
                A_data = A_data[:, all_perm_indices[name][i]]
            A_data = A_data.detach().cpu().numpy()
            im = axs[2 * j, i].matshow(A_data, aspect="auto", cmap="coolwarm")
            if i == 0:
                axs[2 * j, i].set_ylabel("d_in index")
            axs[2 * j, i].set_xlabel("Component index")
            axs[2 * j, i].set_title(f"{name} (A matrix)")
            images.append(im)

            # Plot B matrix
            B_data = Bs[name][i]
            if all_perm_indices is not None:
                B_data = B_data[all_perm_indices[name][i], :]
            B_data = B_data.detach().cpu().numpy()
            im = axs[2 * j + 1, i].matshow(B_data, aspect="auto", cmap="coolwarm")
            if i == 0:
                axs[2 * j + 1, i].set_ylabel("Component index")
            axs[2 * j + 1, i].set_xlabel("d_out index")
            axs[2 * j + 1, i].set_title(f"{name} (B matrix)")
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
