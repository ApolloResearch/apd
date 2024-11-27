from typing import Any

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLPModel


def plot_individual_feature_response(
    model: ResidualMLPModel,
    device: str,
    task_config: dict[str, Any],
    sweep: bool = False,
    instance_idx: int = 0,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    n_instances = model.n_instances
    n_features = model.n_features
    batch_size = model.n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    inputs = torch.ones(n_features) if not sweep else torch.linspace(-1, 1, n_features)
    batch[torch.arange(n_features), instance_idx, torch.arange(n_features)] = inputs.to(device)
    out, _, _ = model(batch)

    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True)
    sweep_str = "set to 1" if not sweep else "between -1 and 1"
    title = (
        f"Feature response with one active feature {sweep_str}\n"
        f"Trained with p={task_config['feature_probability']}, "
        f"n_features={task_config['n_features']}, "
        f"d_embed={task_config['d_embed']}, "
        f"d_mlp={task_config['d_mlp']}"
    )
    fig.suptitle(title)
    for f in range(n_features):
        ax.plot(out[f, :].detach().cpu().numpy(), color=cmap_viridis(f / n_features))
    # Plot labels
    inputs = batch[torch.arange(n_features), instance_idx, torch.arange(n_features)]
    label_fn = F.relu if task_config["act_fn_name"] == "relu" else F.gelu
    targets = inputs + label_fn(inputs)
    ax.plot(torch.arange(n_features), targets.cpu().detach(), color="red", label="Target")
    ax.legend()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_viridis, norm=plt.Normalize(0, n_features))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Active input feature index")
    ax.set_xlabel("Output feature index")
    ax.set_ylabel("Output (all inputs superimposed)")
    return fig


def _calculate_snr(
    model: ResidualMLPModel, device: str, input_values: tuple[float, float]
) -> Tensor:
    n_features = model.n_features
    n_instances = model.n_instances
    batch_size = n_features**2
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    instance_idx = 0
    snr = torch.zeros(n_features, n_features)
    for f1 in range(n_features):
        for f2 in range(n_features):
            idx = f1 * n_features + f2
            batch[idx, instance_idx, f1] = input_values[0]
            batch[idx, instance_idx, f2] = input_values[1]
    out, _, _ = model(batch)
    out: Float[Tensor, "batch n_features"] = out[:, instance_idx, :]
    for f1 in range(n_features):
        for f2 in range(n_features):
            idx = f1 * n_features + f2
            signal = min(out[idx, f1].abs().item(), out[idx, f2].abs().item())
            noise = out[idx, :].std().item()
            snr[f1, f2] = signal / noise
    return snr


def plot_2d_snr(model: ResidualMLPModel, device: str):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, height_ratios=[1, 10, 10], constrained_layout=True, figsize=(4, 8)
    )  # type: ignore
    # Calculate SNR for (1, 1) and implicitly (1,) too.
    snr = _calculate_snr(model, device, input_values=(1, 1)).cpu().detach()
    # Plot diagonal in top subplot
    diagonal = torch.diag(snr)
    im1 = ax1.imshow(diagonal.unsqueeze(0), aspect="auto", vmin=1, vmax=snr.max())
    ax1.set_yticks([])
    ax1.set_title("SNR for single active features")
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    # Plot main SNR matrix without diagonal
    snr_no_diag = snr.clone()
    snr_no_diag.fill_diagonal_(torch.nan)
    im2 = ax2.imshow(snr_no_diag, aspect="auto", vmin=1, vmax=snr.max())
    ax2.set_title("SNR for pairs of active features set to (1, 1)")
    ax2.set_xlabel("Feature 2 (set to 1)")
    ax2.set_ylabel("Feature 1 (set to 1)")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    # Calculate SNR for (1, -1)
    snr = _calculate_snr(model, device, input_values=(1, -1)).cpu().detach()
    # Plot second SNR matrix without diagonal
    snr_no_diag = snr.clone()
    snr_no_diag.fill_diagonal_(torch.nan)
    im3 = ax3.imshow(snr_no_diag, aspect="auto", vmin=1, vmax=snr.max())
    ax3.set_title("SNR for pairs of active features set to (1, -1)")
    ax3.set_xlabel("Feature 2 (set to -1)")
    ax3.set_ylabel("Feature 1 (set to 1)")
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    return fig


def _calculate_virtual_weights(model: ResidualMLPModel, device: str) -> dict[str, Tensor]:
    n_instances = model.n_instances
    n_features = model.n_features
    d_embed = model.d_embed
    d_mlp = model.d_mlp
    has_bias1 = model.layers[0].bias1 is not None
    has_bias2 = model.layers[0].bias2 is not None
    n_layers = model.n_layers
    assert n_layers == 1, "Only implemented for 1 layer"
    # Get weights
    W_E: Float[Tensor, "n_instances n_features d_embed"] = model.W_E
    W_U: Float[Tensor, "n_instances d_embed n_features"] = model.W_U
    W_in: Float[Tensor, "n_instances d_embed d_mlp"] = model.layers[0].linear1.data
    W_out: Float[Tensor, "n_instances d_mlp d_embed"] = model.layers[0].linear2.data
    b_in: Float[Tensor, "n_instances d_mlp"] | None = (
        model.layers[0].bias1.data if has_bias1 else None
    )
    b_out: Float[Tensor, "n_instances d_embed"] | None = (
        model.layers[0].bias2.data if has_bias2 else None
    )
    assert W_E.shape == (n_instances, n_features, d_embed)
    assert W_U.shape == (n_instances, d_embed, n_features)
    assert W_in.shape == (n_instances, d_embed, d_mlp)
    assert W_out.shape == (n_instances, d_mlp, d_embed)
    assert b_in.shape == (n_instances, d_mlp) if b_in is not None else True
    assert b_out.shape == (n_instances, d_embed) if b_out is not None else True
    # Calculate connection strengths / virtual weights
    in_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_instances n_features d_embed, n_instances d_embed d_mlp -> n_instances n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_instances d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_instances d_mlp d_embed, n_instances n_features d_embed -> n_instances d_mlp n_features",
    )
    diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_instances n_features d_mlp, n_instances d_mlp n_features -> n_instances n_features d_mlp",
    )
    assert in_conns.shape == (n_instances, n_features, d_mlp)
    assert out_conns.shape == (n_instances, d_mlp, n_features)
    assert diag_relu_conns.shape == (n_instances, n_features, d_mlp)
    virtual_weights = {
        "W_E": W_E,
        "W_U": W_U,
        "W_in": W_in,
        "W_out": W_out,
        "in_conns": in_conns,
        "out_conns": out_conns,
        "diag_relu_conns": diag_relu_conns,
    }
    if b_in is not None:
        virtual_weights["b_in"] = b_in
    if b_out is not None:
        virtual_weights["b_out"] = b_out
    return virtual_weights


def relu_contribution_plot(model: ResidualMLPModel, device: str, instance_idx: int = 0):
    virtual_weights = _calculate_virtual_weights(model, device)
    diag_relu_conns: Float[Tensor, "n_features d_mlp"] = (
        virtual_weights["diag_relu_conns"][instance_idx].cpu().detach()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)  # type: ignore
    ax1.set_title("How much does each ReLU contribute to each feature?")
    ax1.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.n_features):
        ax1.scatter([i] * model.d_mlp, diag_relu_conns[i, :], alpha=0.3, marker=".", c="k")
        ax1.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(model.d_mlp):
            if diag_relu_conns[i, j] > 0.1:
                cmap_label = plt.get_cmap("hsv")
                ax1.text(i, diag_relu_conns[i, j], str(j), color=cmap_label(j / model.d_mlp))
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Weights to ReLUs")
    ax1.set_xlim(-0.5, model.n_features - 0.5)

    ax2.set_title("How much does each feature route through each ReLU?")
    ax2.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.d_mlp):
        ax2.scatter([i] * model.n_features, diag_relu_conns[:, i], alpha=0.3, marker=".", c="k")
        ax2.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(model.n_features):
            if diag_relu_conns[j, i] > 0.2:
                cmap_label = plt.get_cmap("hsv")
                ax2.text(i, diag_relu_conns[j, i], str(j), color=cmap_label(j / model.n_features))
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("ReLUs")
    ax2.set_ylabel("Weights to features")
    ax2.set_xlim(-0.5, model.d_mlp - 0.5)
    return fig


def plot_virtual_weights(model: ResidualMLPModel, device: str, instance_idx: int = 0):
    virtual_weights = _calculate_virtual_weights(model, device)
    in_conns = virtual_weights["in_conns"][instance_idx].cpu().detach()
    out_conns = virtual_weights["out_conns"][instance_idx].cpu().detach()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 20), constrained_layout=True)  # type: ignore
    ax1.matshow(in_conns.T, norm=CenteredNorm(), cmap="RdBu")
    ax1.set_title("Virtual input weights $(W_E W_{in})^T$")
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Neurons")
    ax2.matshow(out_conns, norm=CenteredNorm(), cmap="RdBu")
    ax2.set_title("Virtual output weights $W_{out} W_U$")
    ax2.xaxis.set_label_position("top")
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Neurons")
    return fig
