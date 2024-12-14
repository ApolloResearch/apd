from collections.abc import Callable
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.experiments.piecewise.plotting import plot_matrix
from spd.experiments.resid_mlp.models import (
    ResidualMLPConfig,
    ResidualMLPModel,
    ResidualMLPSPDRankPenaltyConfig,
    ResidualMLPSPDRankPenaltyModel,
)
from spd.utils import SPDOutputs


def plot_individual_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    sweep: bool = False,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    plot_type: Literal["line", "scatter"] = "scatter",  # for Lee
    ax: plt.Axes | None = None,
    cbar: bool = True,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = model_config.n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    inputs = torch.ones(n_features) if not sweep else torch.linspace(-1, 1, n_features)
    batch[torch.arange(n_features), instance_idx, torch.arange(n_features)] = inputs.to(device)
    out = model_fn(batch)

    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    sweep_str = "set to 1" if not sweep else "between -1 and 1"
    title = (
        f"Feature response with one active feature {sweep_str}\n"
        f"n_features={model_config.n_features}, "
        f"d_embed={model_config.d_embed}, "
        f"d_mlp={model_config.d_mlp}"
    )
    ax.set_title(title)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]
    for f in range(n_features):
        x = torch.arange(n_features)
        y = out[f, :].detach().cpu()
        if plot_type == "line":
            ax.plot(x, y, color=cmap_viridis(f / n_features))
        elif plot_type == "scatter":
            ax.scatter(x, y, c=cmap_viridis(f / n_features))
        else:
            raise ValueError("Unknown plot_type")
    # Plot labels
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    inputs = batch[torch.arange(n_features), instance_idx, torch.arange(n_features)].detach().cpu()
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    baseline = torch.zeros(n_features) if subtract_inputs else inputs
    if plot_type == "line":
        ax.plot(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label="Target ($x+\mathrm{ReLU}(x)$)",
        )
        ax.plot(
            torch.arange(n_features),
            baseline,
            color="red",
            linestyle=":",
            label="Baseline (Identity)",
        )
    elif plot_type == "scatter":
        ax.scatter(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label="Target ($x+\mathrm{ReLU}(x)$)",
            marker="x",
        )
    else:
        raise ValueError("Unknown plot_type")
    ax.legend()
    if cbar:
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_viridis, norm=plt.Normalize(0, n_features))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Active input feature index")
    ax.set_xlabel("Output index")
    ax.set_ylabel("Output values $y_i$ (superimposed)")
    return fig


def plot_single_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    feature_idx: int = 15,
    plot_type: Literal["line", "scatter"] = "scatter",  # for Lee
    ax: plt.Axes | None = None,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = 1
    batch_idx = 0
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[batch_idx, instance_idx, feature_idx] = 1
    out = model_fn(batch)

    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]
    x = torch.arange(n_features)
    y = out[batch_idx, :].detach().cpu()
    inputs = batch[batch_idx, instance_idx, :].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    if plot_type == "line":
        ax.plot(x, y, color=cmap_viridis(feature_idx / n_features), label="Model")
        ax.plot(torch.arange(n_features), targets.cpu().detach(), color="red", label="Labels")
    elif plot_type == "scatter":
        ax.scatter(x, y, c=cmap_viridis(feature_idx / n_features), label="Model")
        ax.scatter(
            torch.arange(n_features), targets.cpu().detach(), c="red", label="Labels", marker="x"
        )
    else:
        raise ValueError("Unknown plot_type")
    ax.legend()
    ax.set_xlabel("Output index")
    ax.set_ylabel(f"Output value $y_{{{feature_idx}}}$")
    ax.set_title(f"Output for a single input $x_{{{feature_idx}}}=1$")
    return fig


def plot_single_relu_curve(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    feature_idx: int = 15,
    ax: plt.Axes | None = None,
    label: bool = True,
):
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = 1000
    x = torch.linspace(-1, 1, batch_size)
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = x
    out = model_fn(batch)
    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]

    y = out[:, feature_idx].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(x) if subtract_inputs else x + label_fn(x)
    ax.plot(x, y, color=cmap_viridis(feature_idx / n_features), label="Model" if label else None)
    ax.plot(x, targets.cpu().detach(), color="red", label="Labels" if label else None)
    ax.legend()
    ax.set_xlabel(f"Input value $x_{{{feature_idx}}}$")
    ax.set_ylabel(f"Output value $y_{{{feature_idx}}}$")
    ax.set_title(f"Input-output response for feature {feature_idx}")
    return fig


def plot_all_relu_curves(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    ax: plt.Axes | None = None,
):
    n_features = model_config.n_features
    for feature_idx in range(n_features):
        fig = plot_single_relu_curve(
            model_fn=model_fn,
            device=device,
            model_config=model_config,
            subtract_inputs=subtract_inputs,
            instance_idx=instance_idx,
            feature_idx=feature_idx,
            ax=ax,
            label=False,
        )
    ax.set_title(f"Input-output response for all {n_features} features")
    ax.set_xlabel("Input values $x_i$")
    # ax.set_ylabel("Output values $y_i$ (superimposed)")
    ax.set_ylabel("")
    return fig


def _calculate_snr(
    model: ResidualMLPModel, device: str, input_values: tuple[float, float]
) -> Tensor:
    n_features = model.config.n_features
    n_instances = model.config.n_instances
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


def calculate_virtual_weights(model: ResidualMLPModel, device: str) -> dict[str, Tensor]:
    """Currently ignoring interactions between layers. Just flattening (n_layers, d_mlp)"""
    n_instances = model.config.n_instances
    n_features = model.config.n_features
    d_embed = model.config.d_embed
    d_mlp = model.config.d_mlp
    has_bias1 = model.layers[0].bias1 is not None
    has_bias2 = model.layers[0].bias2 is not None
    n_layers = model.config.n_layers
    # Get weights
    W_E: Float[Tensor, "n_instances n_features d_embed"] = model.W_E
    W_U: Float[Tensor, "n_instances d_embed n_features"] = model.W_U
    W_in: Float[Tensor, "n_instances d_embed d_mlp_eff"] = torch.cat(
        [model.layers[i].linear1.data for i in range(n_layers)], dim=-1
    )
    W_out: Float[Tensor, "n_instances d_mlp_eff d_embed"] = torch.cat(
        [model.layers[i].linear2.data for i in range(n_layers)],
        dim=-2,
    )
    b_in: Float[Tensor, "n_instances d_mlp_eff"] | None = (
        torch.cat([model.layers[i].bias1.data for i in range(n_layers)], dim=-1)
        if has_bias1
        else None
    )
    b_out: Float[Tensor, "n_instances d_embed"] | None = (
        torch.stack([model.layers[i].bias2.data for i in range(n_layers)]).sum(dim=0)
        if has_bias2
        else None
    )
    assert W_E.shape == (n_instances, n_features, d_embed)
    assert W_U.shape == (n_instances, d_embed, n_features)
    assert W_in.shape == (n_instances, d_embed, n_layers * d_mlp)
    assert W_out.shape == (n_instances, n_layers * d_mlp, d_embed)
    assert b_in.shape == (n_instances, n_layers * d_mlp) if b_in is not None else True
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
    assert in_conns.shape == (n_instances, n_features, n_layers * d_mlp)
    assert out_conns.shape == (n_instances, n_layers * d_mlp, n_features)
    assert diag_relu_conns.shape == (n_instances, n_features, n_layers * d_mlp)
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


def relu_contribution_plot(
    ax1: plt.Axes,
    ax2: plt.Axes,
    all_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"],
    model: ResidualMLPModel | ResidualMLPSPDRankPenaltyModel,
    device: str,
    instance_idx: int = 0,
):
    diag_relu_conns: Float[Tensor, "n_features d_mlp"] = (
        all_diag_relu_conns[instance_idx].cpu().detach()
    )
    d_mlp = model.config.d_mlp
    n_layers = model.config.n_layers
    n_features = model.config.n_features

    ax1.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.config.n_features):
        ax1.scatter([i] * d_mlp * n_layers, diag_relu_conns[i, :], alpha=0.3, marker=".", c="k")
        ax1.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(d_mlp * n_layers):
            if diag_relu_conns[i, j].item() > 0.1:
                cmap_label = plt.get_cmap("hsv")
                ax1.text(
                    i, diag_relu_conns[i, j].item(), str(j), color=cmap_label(j / d_mlp / n_layers)
                )
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlim(-0.5, model.config.n_features - 0.5)
    ax2.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(d_mlp * n_layers):
        ax2.scatter([i] * n_features, diag_relu_conns[:, i], alpha=0.3, marker=".", c="k")
        ax2.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(n_features):
            if diag_relu_conns[j, i].item() > 0.2:
                cmap_label = plt.get_cmap("hsv")
                ax2.text(i, diag_relu_conns[j, i].item(), str(j), color=cmap_label(j / n_features))
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Features")
    ax2.set_xlabel("ReLUs (consecutively enumerated throughout layers)")
    ax2.set_xlim(-0.5, d_mlp * n_layers - 0.5)


def spd_calculate_virtual_weights(
    model: ResidualMLPSPDRankPenaltyModel, device: str
) -> dict[str, Tensor]:
    """Currently ignoring interactions between layers. Just flattening (n_layers, d_mlp)"""
    n_instances = model.config.n_instances
    n_features = model.config.n_features
    d_embed = model.config.d_embed
    d_mlp = model.config.d_mlp
    k_max = model.config.k
    has_bias1 = model.layers[0].linear1.bias is not None
    has_bias2 = model.layers[0].linear2.bias is not None
    n_layers = model.config.n_layers
    # Get weights
    W_E: Float[Tensor, "n_instances n_features d_embed"] = model.W_E
    W_U: Float[Tensor, "n_instances d_embed n_features"] = model.W_U
    W_in: Float[Tensor, "n_instances k d_embed d_mlp_eff"] = torch.cat(
        [model.layers[i].linear1.subnetwork_params for i in range(n_layers)], dim=-1
    )
    W_out: Float[Tensor, "n_instances k d_mlp_eff d_embed"] = torch.cat(
        [model.layers[i].linear2.subnetwork_params for i in range(n_layers)],
        dim=-2,
    )
    b_in: Float[Tensor, "n_instances k d_mlp_eff"] | None = (
        torch.cat([model.layers[i].linear1.bias for i in range(n_layers)], dim=-1)
        if has_bias1
        else None
    )
    b_out: Float[Tensor, "n_instances k d_embed"] | None = (
        torch.stack([model.layers[i].linear2.bias for i in range(n_layers)]).sum(dim=0)
        if has_bias2
        else None
    )
    assert W_E.shape == (n_instances, n_features, d_embed)
    assert W_U.shape == (n_instances, d_embed, n_features)
    assert W_in.shape == (n_instances, k_max, d_embed, n_layers * d_mlp)
    assert W_out.shape == (n_instances, k_max, n_layers * d_mlp, d_embed)
    assert b_in.shape == (n_instances, k_max, n_layers * d_mlp) if b_in is not None else True
    assert b_out.shape == (n_instances, k_max, d_embed) if b_out is not None else True
    # Calculate connection strengths / virtual weights
    in_conns: Float[Tensor, "n_instances k n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_instances n_features d_embed, n_instances k d_embed d_mlp -> n_instances k n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_instances k d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_instances k d_mlp d_embed, n_instances n_features d_embed -> n_instances k d_mlp n_features",
    )
    diag_relu_conns: Float[Tensor, "n_instances k n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_instances k n_features d_mlp, n_instances k d_mlp n_features -> n_instances k n_features d_mlp",
    )
    assert in_conns.shape == (n_instances, k_max, n_features, n_layers * d_mlp)
    assert out_conns.shape == (n_instances, k_max, n_layers * d_mlp, n_features)
    assert diag_relu_conns.shape == (n_instances, k_max, n_features, n_layers * d_mlp)
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


def spd_calculate_diag_relu_conns(
    model: ResidualMLPSPDRankPenaltyModel,
    device: str,
    k_select: int | Literal["sum_before", "sum_nocrossterms", "sum_onlycrossterms"] = 0,
) -> Float[Tensor, "n_instances n_features d_mlp"]:
    virtual_weights = spd_calculate_virtual_weights(model, device)
    if isinstance(k_select, int):
        return virtual_weights["diag_relu_conns"][:, k_select]
    elif k_select == "sum_nocrossterms":
        return virtual_weights["diag_relu_conns"].sum(dim=1)
    else:
        in_conns: Float[Tensor, "n_instances k n_features d_mlp"] = virtual_weights["in_conns"]
        out_conns: Float[Tensor, "n_instances k d_mlp n_features"] = virtual_weights["out_conns"]
        if k_select == "sum_onlycrossterms":
            nocross_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = (
                virtual_weights["diag_relu_conns"].sum(dim=1)
            )
            all_diag_relu_conns: Float[Tensor, "n_instances k1 k2 n_features d_mlp"] = (
                einops.einsum(
                    in_conns,
                    out_conns,
                    "n_instances k1 n_features d_mlp, n_instance k2 d_mlp n_features -> n_instances k1 k2 n_features d_mlp",
                )
            )
            return all_diag_relu_conns.sum(dim=(-3, -4)) - nocross_diag_relu_conns
        elif k_select == "sum_before":
            sum_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
                in_conns.sum(dim=1),
                out_conns.sum(dim=1),
                "n_instances n_features d_mlp, n_instance d_mlp n_features -> n_instances n_features d_mlp",
            )
            return sum_diag_relu_conns
        else:
            raise ValueError(f"Invalid k_select: {k_select}")


def plot_spd_relu_contribution(
    spd_model: ResidualMLPSPDRankPenaltyModel,
    target_model: ResidualMLPModel,
    device: str = "cuda",
    k_select: int | Literal["sum_before", "sum_nocrossterms", "sum_onlycrossterms"] = 0,
    k_plot_limit: int | None = None,
):
    offset = 4
    nrows = (k_plot_limit or spd_model.config.k) + offset
    fig1, axes1 = plt.subplots(nrows, 1, figsize=(20, 3 + 2 * nrows), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # type: ignore
    fig2, axes2 = plt.subplots(nrows, 1, figsize=(10, 3 + 2 * nrows), constrained_layout=True)
    axes2 = np.atleast_1d(axes2)  # type: ignore

    virtual_weights = calculate_virtual_weights(target_model, device)
    relu_conns = virtual_weights["diag_relu_conns"]
    relu_contribution_plot(axes1[0], axes2[0], relu_conns, target_model, device)
    axes1[0].set_ylabel("Target model", fontsize=8)
    axes2[0].set_ylabel("Target model", fontsize=8)
    axes1[0].set_xlabel("")
    axes2[0].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_before")
    relu_contribution_plot(axes1[1], axes2[1], relu_conns, spd_model, device)
    axes1[1].set_ylabel("SPD model full sum of all subnets", fontsize=8)
    axes2[1].set_ylabel("SPD model full sum of all subnets", fontsize=8)
    axes1[1].set_xlabel("")
    axes2[1].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_nocrossterms")
    relu_contribution_plot(axes1[2], axes2[2], relu_conns, spd_model, device)
    axes1[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    axes2[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    axes1[2].set_xlabel("")
    axes2[2].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_onlycrossterms")
    relu_contribution_plot(axes1[3], axes2[3], relu_conns, spd_model, device)
    axes1[3].set_ylabel("SPD model sum only cross terms", fontsize=8)
    axes2[3].set_ylabel("SPD model sum only cross terms", fontsize=8)
    axes1[3].set_xlabel("")
    axes2[3].set_xlabel("")
    for k in range(k_plot_limit or spd_model.config.k):
        relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select=k)
        relu_contribution_plot(axes1[k + offset], axes2[k + offset], relu_conns, spd_model, device)
        axes1[k + offset].set_ylabel(f"k={k}")
        axes2[k + offset].set_ylabel(f"k={k}")
        if k < (k_plot_limit or spd_model.config.k) - 1:
            axes1[k + offset].set_xlabel("")
            axes2[k + offset].set_xlabel("")
    return fig1, fig2


def analyze_per_feature_performance(
    model_fn: Callable[[Float[Tensor, "batch n_instances"]], Float[Tensor, "batch n_instances"]],
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    device: str,
    batch_size: int = 128,
    ax: plt.Axes | None = None,
    label: str | None = None,
    sorted_indices: torch.Tensor | None = None,
    zorder: int = 0,
) -> torch.Tensor:
    """For each feature, run a bunch where only that feature varies, then measure loss"""
    n_features = model_config.n_features
    n_instances = model_config.n_instances
    features = torch.arange(model_config.n_features)
    losses = torch.zeros(model_config.n_features)
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    for i in range(model_config.n_features):
        batch_i = torch.zeros((batch_size, n_instances, n_features), device=device)
        batch_i[:, 0, i] = torch.linspace(-1, 1, batch_size)
        labels_i = torch.zeros((batch_size, n_instances, n_features), device=device)
        labels_i[:, 0, i] = batch_i[:, 0, i] + label_fn(batch_i[:, 0, i])
        model_output = model_fn(batch_i)
        loss = F.mse_loss(model_output, labels_i)
        losses[i] = loss.item()
    losses = losses.detach().cpu()
    sorted_indices = sorted_indices if sorted_indices is not None else losses.argsort()
    # Plot the losses as bar chart with x labels corresponding to feature index
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    color = f"C{zorder%10}"
    ax.bar(features, losses[sorted_indices], alpha=0.5, label=label, zorder=zorder)
    ax.set_xticks(features, features[sorted_indices].numpy(), fontsize=6, rotation=90)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Loss")
    return sorted_indices


def plot_virtual_weights_target_spd(
    target_model: ResidualMLPModel, model: ResidualMLPSPDRankPenaltyModel, device: str
):
    target_virtual_weights = calculate_virtual_weights(target_model, device)
    spd_virtual_weights = spd_calculate_virtual_weights(model=model, device=device)
    instance_idx = 0
    fig = plt.figure(constrained_layout=True, figsize=(10, 2 * model.config.k + 8))
    gs = fig.add_gridspec(ncols=2, nrows=model.config.k + 1 + 2)
    ax_ID = fig.add_subplot(gs[:2, :])
    W_E_W_U = einops.einsum(
        target_virtual_weights["W_E"][instance_idx],
        target_virtual_weights["W_U"][instance_idx],
        "n_features1 d_embed, d_embed n_features2 -> n_features1 n_features2",
    )
    plot_matrix(
        ax_ID,
        W_E_W_U,
        "Virtual weights $W_E W_U$",
        "Features",
        "Features",
        colorbar_format="%.2f",
    )
    norm = Normalize(vmin=-1, vmax=1)
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    in_conns = target_virtual_weights["in_conns"][instance_idx].cpu().detach()
    out_conns = target_virtual_weights["out_conns"][instance_idx].cpu().detach()
    plot_matrix(
        ax1,
        in_conns.T,
        "Virtual input weights $(W_E W_{in})^T$",
        "Features",
        "(Target Model) Neurons",
        colorbar_format="%.2f",
        norm=norm,
    )
    plot_matrix(
        ax2,
        out_conns,
        "Virtual output weights $W_{out} W_U$",
        "Features",
        "Neurons",
        colorbar_format="%.2f",
        norm=norm,
    )
    for ki in range(model.config.k):
        ax1 = fig.add_subplot(gs[3 + ki, 0])
        ax2 = fig.add_subplot(gs[3 + ki, 1])
        plot_matrix(
            ax1,
            spd_virtual_weights["in_conns"][instance_idx, ki].T,
            "$(W_E W_{in})^T$",
            "Features",
            f"k={ki} Neurons",
            colorbar_format="%.2f",
            norm=norm,
        )
        plot_matrix(
            ax2,
            spd_virtual_weights["out_conns"][instance_idx, ki],
            "$W_{out} W_U$",
            "Features",
            "Neurons",
            colorbar_format="%.2f",
            norm=norm,
        )
    return fig


def plot_resid_vs_mlp_out(
    target_model: ResidualMLPModel,
    device: str,
    ax: plt.Axes,
    topk_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances k"] | None,
        ],
        SPDOutputs,
    ]
    | None = None,
    subnet_indices: Float[Tensor, " k"] | None = None,
    instance_idx: int = 0,
    feature_idx: int = 0,
):
    tied_weights = True
    if not torch.allclose(target_model.W_U.data, target_model.W_E.data.transpose(-2, -1)):
        print("Warning: W_E and W_U are not tied")
        tied_weights = False
    batch_size = 1
    batch_idx = 0
    n_instances = target_model.config.n_instances
    n_features = target_model.config.n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = 1
    # Target model full output
    out = target_model(batch)[0][batch_idx, instance_idx, :].cpu().detach()
    # Target model residual stream contribution
    W_E = target_model.W_E[instance_idx].cpu().detach()
    W_U = target_model.W_U[instance_idx].cpu().detach()
    W_EU = einops.einsum(W_E, W_U, "f1 d_mlp, d_mlp f2 -> f1 f2")[feature_idx, :]
    # Compute MLP-out
    mlp_out = out - W_EU
    # Mask for noise & correlation
    mask = torch.ones_like(out).bool()
    mask[feature_idx] = False
    noise_out = F.mse_loss(out[mask], torch.zeros_like(out[mask])).item()
    corr = np.corrcoef(mlp_out[mask], W_EU[mask])[0, 1]
    ax.axhline(0, color="grey", linestyle="-", lw=0.5)
    ax.plot([], [], c="white", label=f"Full target model noise level ~ {noise_out:.2e}")
    ax.plot(
        mlp_out,
        color="C0",
        label=f"Target MLP output.\n"
        f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
        lw=2,
    )
    noise_W_EU = F.mse_loss(W_EU[mask], torch.zeros_like(W_EU[mask])).item()
    ax.plot(
        W_EU,
        color="C1",
        label=f"Target resid contribution (W_E W_U)\n" f"Noise level ~ {noise_W_EU:.2e}",
    )
    # If topk_model_fn is provided, use it to get the SPD model output
    if topk_model_fn is not None:
        # Get the SPD resid contribution by running with no subnetworks. This should be equivalent
        # to W_E W_U and but doesn't require access to the ResidMLP SPD model.
        topk_mask = torch.zeros_like(batch)
        spd_WEU = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        spd_WEU = spd_WEU.detach().cpu()
        if tied_weights:
            assert torch.allclose(spd_WEU, W_EU), "Tied weights but W_EU != SPD resid contribution"
        else:
            ax.plot(
                spd_WEU,
                color="C4",
                label="SPD resid contribution (no subnets).\n"
                "Note that embeddings are untied and numbers in legend are not applicable",
                ls=":",
            )
        # Get SPD forward pass, either from subnet_indices or attribution-based topk_mask
        if subnet_indices is None:
            topk_mask = None
        else:
            topk_mask = torch.zeros_like(batch)
            topk_mask[:, :, subnet_indices] = 1
        topk_out = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        topk_mlp_out = topk_out.detach().cpu() - spd_WEU
        topk_mlp_out_mse = F.mse_loss(topk_mlp_out, mlp_out).item()
        corr = np.corrcoef(topk_mlp_out[mask], W_EU[mask])[0, 1]
        ax.plot(
            topk_mlp_out,
            color="C2",
            label=f"SPD MLP output (topk) MSE: {topk_mlp_out_mse:.1e}.\n"
            f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
            ls="--",
        )
        # Full forward pass
        topk_mask = torch.ones_like(batch)
        full_out = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        full_mlp_out = full_out.detach().cpu() - spd_WEU
        full_mlp_out_mse = F.mse_loss(full_mlp_out, mlp_out).item()
        corr = np.corrcoef(full_mlp_out[mask], W_EU[mask])[0, 1]
        ax.plot(
            full_mlp_out,
            color="C3",
            label=f"SPD MLP output (full) MSE: {full_mlp_out_mse:.1e}.\n"
            f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
            ls=":",
        )
    # Can we scale W_EU by a scalar to make it match the model output in mask?
    # def difference(alpha):
    #     return F.mse_loss( float(alpha) * W_EU[mask], out[mask])
    # from scipy.optimize import minimize
    # res = minimize(difference, x0=0.1, method="Nelder-Mead")
    # ax.plot(W_EU * float(res.x[0]), color="C2", label="Scaled W_E W_U")
    ax.legend()
    ax.set_title(f"Instance {instance_idx}, feature {feature_idx}")


def plot_feature_response_with_subnets(
    topk_model_fn: Callable[
        [Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances k"]],
        SPDOutputs,
    ],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    feature_idx: int = 0,
    subnet_idx: int = 0,
    instance_idx: int = 0,
    ax: plt.Axes | None = None,
    batch_size: int | None = None,
    plot_type: Literal["line", "scatter"] = "scatter",  # for Lee
):
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = batch_size or n_features

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.figure

    cmap_blues = plt.get_cmap("Purples")
    cmap_reds = plt.get_cmap("Oranges")

    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = 1
    topk_mask_blue = torch.zeros_like(batch[:, :, :])
    topk_mask_red = torch.zeros_like(batch[:, :, :])
    topk_mask_blue[:, :, subnet_idx] = 1
    for s in range(batch_size):
        choice = torch.randperm(n_features - 1)[:s]
        # Exclude feature_idx from choice
        choice[choice >= subnet_idx] += 1
        topk_mask_blue[s, :, choice] = 1
        topk_mask_red[s, :, choice] = 1
    assert torch.allclose(
        topk_mask_blue[:, :, subnet_idx], torch.ones_like(topk_mask_blue[:, :, subnet_idx])
    )
    assert torch.allclose(
        topk_mask_red[:, :, subnet_idx], torch.zeros_like(topk_mask_red[:, :, subnet_idx])
    )
    zero_topk_mask = torch.zeros_like(batch[:, :, :])
    out_WE_WU_only = topk_model_fn(batch, zero_topk_mask).spd_topk_model_output[:, instance_idx, :]

    out_red = topk_model_fn(batch, topk_mask_red)
    out_blue = topk_model_fn(batch, topk_mask_blue)
    mlp_out_blue_spd = out_blue.spd_topk_model_output[:, instance_idx, :] - out_WE_WU_only
    mlp_out_red_spd = out_red.spd_topk_model_output[:, instance_idx, :] - out_WE_WU_only
    mlp_out_target = out_blue.target_model_output[:, instance_idx, :] - out_WE_WU_only

    x = torch.arange(n_features)
    for s in range(batch_size):
        yb = mlp_out_blue_spd[s, :].detach().cpu()
        yr = mlp_out_red_spd[s, :].detach().cpu()
        if plot_type == "line":
            ax.plot(x, yb, color=cmap_blues(s / batch_size), lw=0.3)
            ax.plot(x, yr, color=cmap_reds(s / batch_size), lw=0.3)
        elif plot_type == "scatter":
            ax.scatter(x, yb, c=cmap_blues(s / batch_size), lw=0.3, marker=".")
            ax.scatter(x, yr, c=cmap_reds(s / batch_size), lw=0.3, marker=".")
        else:
            raise ValueError("Unknown plot_type")
    yt = mlp_out_target[0, :].detach().cpu()
    if plot_type == "line":
        ax.plot(x, yt, color="red", lw=0.5, label="Target model")
    elif plot_type == "scatter":
        ax.scatter(x, yt, marker="x", lw=0.5, label="Target model", c="r")
    else:
        raise ValueError("Unknown plot_type")

    ax.set_ylabel("MLP output (forward pass minus W_E W_U contribution)")
    ax.set_xlabel("Output index")
    ax.plot([], [], color=cmap_blues(0), label=f"SPD with right subnet ({subnet_idx})")
    ax.plot([], [], color=cmap_reds(0), label=f"SPD without right subnet ({subnet_idx})")
    ax.scatter(
        x, mlp_out_target[0, :].detach().cpu(), color="wheat", label="Target", marker=".", zorder=-1
    )
    ax.set_title(f"SPD model output for increasing number of subnets, feature {feature_idx}")
    ax.legend()
    return {"feature_response_with_subnets": fig}


def get_feature_subnet_map(
    top1_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances k"] | None,
        ],
        SPDOutputs,
    ],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDRankPenaltyConfig,
    instance_idx: int = 0,
) -> dict[int, int]:
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[torch.arange(n_features), instance_idx, torch.arange(n_features)] = 1
    top1_out = top1_model_fn(batch, None)
    top1_mask = top1_out.topk_mask[:, instance_idx, :]
    subnet_indices = {
        int(feature_idx.item()): int(subnet_idx.item())
        for feature_idx, subnet_idx in top1_mask.nonzero()
    }
    return subnet_indices
