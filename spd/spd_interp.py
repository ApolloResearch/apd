from typing import Any

import matplotlib.pyplot as plt

from spd.configs import Config
from spd.experiments.resid_mlp.models import ResidualMLP
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import plot_causal_importance_vals
from spd.settings import REPO_ROOT


def extract_ci_val_figures(run_id: str, input_magnitude: float = 0.75) -> dict[str, Any]:
    """Extract causal importances from a single run.

    Args:
        run_id: Wandb run ID to load model from
        input_magnitude: Magnitude of input features for causal importances plotting

    Returns:
        Dictionary containing causal importances data and metadata
    """
    model, config, _ = ComponentModel.from_pretrained(run_id)
    target_model = model.model
    assert isinstance(target_model, ResidualMLP | TMSModel), (
        "Target model must be a ResidualMLP or TMSModel"
    )
    n_features = target_model.config.n_features

    # Get components and gates from model
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    # Assume no position dimension
    batch_shape = (1, n_features)

    # Get device from model
    device = next(model.parameters()).device

    # Get mask values without plotting regular masks
    figures, all_perm_indices_ci_vals = plot_causal_importance_vals(
        model=model,
        components=components,
        gates=gates,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=input_magnitude,
        plot_raw_cis=False,
    )

    return {
        "figures": figures,
        "all_perm_indices_ci_vals": all_perm_indices_ci_vals,
        "config": config,
        "components": components,
        "n_features": n_features,
    }


def plot_increasing_importance_coeff_ci_vals(
    run_ids: list[str], input_magnitude: float = 0.75, best_idx: list[int] | None = None
) -> plt.Figure:
    """Plot increasing importance coeff for multiple runs in a combined figure.

    Args:
        run_ids: List of wandb run IDs to load models from
        input_magnitude: Magnitude of input features for causal importances plotting
        best_idx: List of indices indicating which runs are the best (for highlighting)

    Returns:
        Combined figure with causal importances from all runs
    """
    all_mask_data = {}
    all_components = []

    # Collect causal importances from all runs
    for run_id in run_ids:
        print(f"Loading model from {run_id}")

        # Extract causal importances using helper function
        extraction_result = extract_ci_val_figures(run_id, input_magnitude)
        figures = extraction_result["figures"]
        config = extraction_result["config"]
        assert isinstance(config, Config)

        # Extract causal importances data from the figure
        ci_vals_fig = figures["causal_importances"]

        # Get mask data from the figure axes
        mask_data = {}
        for i, ax in enumerate(ci_vals_fig.axes[:-1]):  # Skip colorbar axis
            # Get the image data from the axis
            images = ax.get_images()
            if images:
                data = images[0].get_array()
                # Get component name from axis title
                title = ax.get_title()
                component_name = title.split(" (")[0]  # Extract component name
                mask_data[component_name] = data

                # Track all unique component names
                if component_name not in all_components:
                    all_components.append(component_name)

        all_mask_data[run_id] = {
            "mask_data": mask_data,
            "importance_loss_coeff": config.importance_loss_coeff,
        }
        plt.close(ci_vals_fig)  # Close the individual figure

    # Create combined figure
    n_runs = len(run_ids)
    n_components = len(all_components)

    fig, axs = plt.subplots(
        n_components,
        n_runs,
        figsize=(5 * n_runs, 5 * n_components),
        constrained_layout=False,
        squeeze=False,
        dpi=300,
    )

    # Plot all masks
    images = []
    vmin, vmax = float("inf"), float("-inf")

    component_name_map = {"layers.0.mlp_in": "$W_{IN}$", "layers.0.mlp_out": "$W_{OUT}$"}
    for col_idx, run_id in enumerate(run_ids):
        for row_idx, component_name in enumerate(all_components):
            ax = axs[row_idx, col_idx]

            assert component_name in all_mask_data[run_id]["mask_data"]
            mask_data = all_mask_data[run_id]["mask_data"][component_name]
            im = ax.matshow(mask_data, aspect="auto", cmap="Reds")
            images.append(im)

            # Track min/max for unified colorbar
            vmin = min(vmin, mask_data.min())
            vmax = max(vmax, mask_data.max())

            component_name = component_name_map.get(component_name, component_name)
            # Add labels
            if col_idx == 0:
                ax.set_ylabel(f"{component_name}\nInput feature index", fontsize=14)
            else:
                ax.set_ylabel("")

            if row_idx == n_components - 1:
                ax.set_xlabel("Subcomponent index", fontsize=14)

            # Increase tick label font sizes
            ax.tick_params(axis="both", which="major", labelsize=12)
            # Move x-axis ticks to bottom
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position("bottom")

            if row_idx == 0:
                # Add importance_loss_coeff as column title
                lp_coeff = all_mask_data[run_id]["importance_loss_coeff"]
                title_text = f"Importance coeff={lp_coeff:.0e}"

                # Add "BEST" indicator if this is one of the best runs
                if best_idx is not None and col_idx in best_idx:
                    title_text += " (BEST)"

                ax.set_title(title_text, fontsize=14, pad=13)

    # Highlight best runs with visual distinctions
    if best_idx is not None:
        for best_col_idx in best_idx:
            if 0 <= best_col_idx < n_runs:
                # Add colored borders and background to all subplots in the best column
                for row_idx in range(n_components):
                    ax = axs[row_idx, best_col_idx]

                    # Add a thick colored border around the subplot
                    for spine in ax.spines.values():
                        spine.set_edgecolor("darkblue")
                        spine.set_linewidth(3)
                        spine.set_visible(True)

                    # Add a subtle background color
                    ax.set_facecolor("#f0f8ff")  # Very light blue background

                # Make the title more prominent for the best column
                if n_components > 0:  # Ensure we have at least one component
                    top_ax = axs[0, best_col_idx]
                    current_title = top_ax.get_title()
                    # Update title with bold formatting and color
                    top_ax.set_title(current_title, fontsize=16, color="darkblue", pad=18)

    # Add unified colorbar
    if images:
        # Normalize all images to the same scale
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Add colorbar
        cbar = fig.colorbar(images[0], ax=axs.ravel().tolist(), label="Importance value")
        cbar.set_label("Importance value", fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    return fig


if __name__ == "__main__":
    run_ids = [
        "wandb:spd-resid-mlp/runs/5whdnjhz",  # 1e-6
        "wandb:spd-resid-mlp/runs/18v49hfa",  # 3e-6
        "wandb:spd-resid-mlp/runs/howbugfl",  # Best. 1e-5
        "wandb:spd-resid-mlp/runs/flaqx6dr",  # 1e-4
        "wandb:spd-resid-mlp/runs/bfgxcmnb",  # 1e-3
    ]
    best_idx = [2]

    # Create and save the combined figure
    fig = plot_increasing_importance_coeff_ci_vals(run_ids, best_idx=best_idx)
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out/"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "resid_mlp_varying_importance_coeff_ci_vals.png",
        bbox_inches="tight",
        dpi=400,
    )
    print(f"Saved figure to {out_dir / 'resid_mlp_varying_importance_coeff_ci_vals.png'}")
    plt.show()
