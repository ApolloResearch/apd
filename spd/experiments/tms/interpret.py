import json

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.run_spd import (
    Config,
)
from spd.utils import REPO_ROOT

LABEL_FONT_SIZE = 10


def plot_vectors(
    subnets: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    n_instances: int | None = None,
) -> plt.Figure:
    """2D polygon plot of each subnetwork.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    if n_instances is not None:
        subnets = subnets[:n_instances]
    n_instances, n_subnets, n_features, n_hidden = subnets.shape

    # Make a new subnet index in the beginning which is the sum of all subnets
    subnets = torch.cat([subnets.sum(dim=1, keepdim=True), subnets], dim=1)
    n_subnets += 1

    # Use different colors for each subnetwork if there's only one instance
    color_vals = np.linspace(0, 1, n_features) if n_instances == 1 else np.zeros(n_features)
    colors = plt.cm.viridis(color_vals)  # type: ignore

    fig, axs = plt.subplots(n_instances, n_subnets, figsize=(2 * n_subnets, 3 * n_instances))
    axs = np.atleast_2d(np.array(axs))

    for j in range(n_subnets):
        for i, ax in enumerate(axs[:, j]):
            arr = subnets[i, j].cpu().detach().numpy()

            # Plot each feature with its unique color
            for k in range(n_features):
                ax.scatter(arr[k, 0], arr[k, 1], color=colors[k])
                ax.add_collection(
                    mc.LineCollection([[(0, 0), (arr[k, 0], arr[k, 1])]], colors=[colors[k]])
                )

            ax.set_aspect("equal")
            z = 1.5
            ax.set_facecolor("#f6f6f6")
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))
            ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

            if i == n_instances - 1:
                label = "Sum of subnets" if j == 0 else f"Subnet {j-1}"
                ax.set_xlabel(label, rotation=0, ha="center", labelpad=60, fontsize=LABEL_FONT_SIZE)
            if j == 0 and n_instances > 1:
                ax.set_ylabel(
                    f"Instance {i}", rotation=90, ha="center", labelpad=60, fontsize=LABEL_FONT_SIZE
                )

    return fig


def plot_networks(
    subnets: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    n_instances: int | None = None,
    has_labels: bool = True,
) -> plt.Figure:
    """Plot neural network diagrams for each W matrix in the subnet variable.

    Args:
        subnet: Tensor of shape [n_instances, n_subnets, n_features, n_hidden].
        n_instances: Number of data instances to plot. If None, plot all.
        has_labels: Whether to add labels to the plot.

    Returns:
        Matplotlib Figure object containing the network diagrams.
    """

    if n_instances is not None:
        subnets = subnets[:n_instances]
    n_instances, n_subnets, n_features, n_hidden = subnets.shape

    # Make a new subnet index in the beginning which is the sum of all subnets
    subnets = torch.cat([subnets.sum(dim=1, keepdim=True), subnets], dim=1)
    n_subnets += 1

    # Take the absolute value of the weights
    subnets = subnets.abs()

    # Find the maximum weight across each instance
    max_weights = subnets.amax(dim=(1, 2, 3))

    # Create a figure with subplots arranged as per the existing layout
    fig, axs = plt.subplots(
        n_instances,
        n_subnets,
        figsize=(2 * n_subnets, 3 * n_instances),
        constrained_layout=True,
    )
    axs = np.atleast_2d(np.array(axs))

    # axs[0, 0].set_xlabel("Outputs (before ReLU and biases)")
    # Add the above but in text because the x-axis is killed
    axs[0, 0].text(
        0.1,
        0.05,
        "Outputs (before\nbias and ReLU)",
        ha="left",
        va="center",
        transform=axs[0, 0].transAxes,
        fontsize=LABEL_FONT_SIZE,
    )
    # Also add "input label"
    axs[0, 0].text(
        0.1,
        0.95,
        "Inputs",
        ha="left",
        va="center",
        transform=axs[0, 0].transAxes,
        fontsize=LABEL_FONT_SIZE,
    )

    # Grayscale colormap. darker for larger weight
    cmap = plt.get_cmap("gray_r")

    for j in range(n_subnets):
        for i, ax in enumerate(axs[:, j]):
            arr = subnets[i, j].cpu().detach().numpy()

            # Define node positions (top to bottom)
            y_input, y_hidden, y_output = 0, -1, -2
            x_input = np.linspace(0.05, 0.95, n_features)
            x_hidden = np.linspace(0.25, 0.75, n_hidden)
            x_output = np.linspace(0.05, 0.95, n_features)

            # Add transparent grey box around hidden layer
            box_width = 0.8
            box_height = 0.4
            box = plt.Rectangle(
                (0.5 - box_width / 2, y_hidden - box_height / 2),
                box_width,
                box_height,
                fill=True,
                facecolor="#e4e4e4",
                edgecolor="none",
                alpha=0.33,
                transform=ax.transData,
            )
            ax.add_patch(box)

            # Plot nodes
            ax.scatter(
                x_input, [y_input] * n_features, s=200, color="grey", edgecolors="k", zorder=3
            )
            ax.scatter(
                x_hidden, [y_hidden] * n_hidden, s=200, color="grey", edgecolors="k", zorder=3
            )
            ax.scatter(
                x_output, [y_output] * n_features, s=200, color="grey", edgecolors="k", zorder=3
            )

            # Plot edges from input to hidden layer
            for idx_input in range(n_features):
                for idx_hidden in range(n_hidden):
                    weight = arr[idx_input, idx_hidden]
                    norm_weight = weight / max_weights[i]
                    color = cmap(norm_weight)
                    ax.plot(
                        [x_input[idx_input], x_hidden[idx_hidden]],
                        [y_input, y_hidden],
                        color=color,
                        linewidth=1,
                    )

            # Plot edges from hidden to output layer
            arr_T = arr.T  # Transpose of W for W^T
            for idx_hidden in range(n_hidden):
                for idx_output in range(n_features):
                    weight = arr_T[idx_hidden, idx_output]
                    norm_weight = weight / max_weights[i]
                    color = cmap(norm_weight)
                    ax.plot(
                        [x_hidden[idx_hidden], x_output[idx_output]],
                        [y_hidden, y_output],
                        color=color,
                        linewidth=1,
                    )

            # Remove axes for clarity
            ax.axis("off")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(y_output - 0.5, y_input + 0.5)

            if has_labels:
                if i == n_instances - 1:
                    label = "Sum of subnets" if j == 0 else f"Subnet {j - 1}"
                    ax.text(
                        0.5,
                        0,
                        label,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=LABEL_FONT_SIZE,
                    )
                if j == 0 and n_instances > 1:
                    ax.text(
                        -0.1,
                        0.5,
                        f"Instance {i}",
                        ha="center",
                        va="center",
                        rotation=90,
                        transform=ax.transAxes,
                        fontsize=LABEL_FONT_SIZE,
                    )

    return fig


if __name__ == "__main__":
    pretrained_path = REPO_ROOT / "spd/experiments/tms/demo_spd_model/model_30000.pth"

    with open(pretrained_path.parent / "config.json") as f:
        config_dict = json.load(f)
        config = Config(**config_dict)

    assert config.full_rank, "This script only works for full rank models"
    model = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    subnets = model["subnetwork_params"]
    bias = model["b_final"]
    subnets = torch.load(pretrained_path, map_location="cpu")["subnetwork_params"]
    vector_fig = plot_vectors(subnets, n_instances=1)
    vector_fig.savefig(pretrained_path.parent / "polygon_diagram.png", bbox_inches="tight", dpi=200)
    print(f"Saved figure to {pretrained_path.parent / 'polygon_diagram.png'}")

    subnet_fig = plot_networks(subnets, n_instances=1, has_labels=False)
    subnet_fig.savefig(pretrained_path.parent / "network_diagram.png", bbox_inches="tight", dpi=200)
    print(f"Saved figure to {pretrained_path.parent / 'network_diagram.png'}")
