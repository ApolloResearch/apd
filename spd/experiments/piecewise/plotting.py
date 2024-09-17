from pathlib import Path
from typing import NamedTuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.run_spd import Config, calc_recon_mse
from spd.utils import (
    BatchedDataLoader,
    calc_attributions_full_rank,
    calc_attributions_rank_one,
    calc_topk_mask,
)


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.0f",
) -> None:
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm())
    for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
        ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=4)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_components_fullrank(
    model: PiecewiseFunctionSPDFullRankTransformer,
    step: int,
    out_dir: Path | None,
    slow_images: bool,
) -> dict[str, plt.Figure]:
    # Not implemented attribution score plots, or multi-layer plots, yet.
    assert model.n_layers == 1
    ncols = 2
    if slow_images:
        nrows = model.k + 1
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(16 * ncols, 3 * nrows), constrained_layout=True
        )
    else:
        nrows = 1
        fig, axs_row = plt.subplots(
            nrows, ncols, figsize=(16 * ncols, 3 * nrows), constrained_layout=True
        )
        axs = np.array([axs_row])

    assert isinstance(axs, np.ndarray)
    plot_matrix(
        axs[0, 0],
        einops.einsum(model.mlps[0].linear1.subnetwork_params, "k ... -> ..."),
        "W_in, sum over k",
        "Neuron index",
        "Embedding index",
    )
    plot_matrix(
        axs[0, 1],
        einops.einsum(model.mlps[0].linear2.subnetwork_params, "k ... -> ...").T,
        "W_out.T, sum over k",
        "Neuron index",
        "",
    )

    if slow_images:
        for k in range(model.k):
            mlp = model.mlps[0]
            W_in_k = mlp.linear1.subnetwork_params[k]
            ax = axs[k + 1, 0]  # type: ignore
            plot_matrix(ax, W_in_k, f"W_in_k, k={k}", "Neuron index", "Embedding index")
            W_out_k = mlp.linear2.subnetwork_params[k].T
            ax = axs[k + 1, 1]  # type: ignore
            plot_matrix(ax, W_out_k, f"W_out_k.T, k={k}", "Neuron index", "")
    return {"matrices_layer0": fig}


def plot_components(
    model: PiecewiseFunctionSPDTransformer,
    step: int,
    out_dir: Path | None,
    device: str,
    slow_images: bool,
) -> dict[str, plt.Figure]:
    # Create a batch of inputs with different control bits active
    x_val = torch.tensor(2.5, device=device)
    batch_size = model.n_inputs - 1  # Assuming first input is for x_val and rest are control bits
    x = torch.zeros(batch_size, model.n_inputs, device=device)
    x[:, 0] = x_val
    x[torch.arange(batch_size), torch.arange(1, batch_size + 1)] = 1
    # Forward pass to get the output and inner activations
    out, layer_acts, inner_acts = model(x)
    # Calculate attribution scores
    attribution_scores = calc_attributions_rank_one(out=out, inner_acts=inner_acts)
    attribution_scores_normed = attribution_scores / attribution_scores.std(dim=1, keepdim=True)
    # Get As and Bs and ABs
    n_layers = model.n_layers
    assert len(model.all_As()) == len(model.all_Bs()), "A and B matrices must have the same length"
    assert len(model.all_As()) % 2 == 0, "A and B matrices must have an even length (MLP in + out)"
    assert len(model.all_As()) // 2 == n_layers, "Number of A and B matrices must be 2*n_layers"
    As = model.all_As()
    Bs = model.all_Bs()
    ABs = [torch.einsum("...fk,...kg->...fg", As[i], Bs[i]) for i in range(len(As))]
    ABs_by_k = [torch.einsum("...fk,...kg->...kfg", As[i], Bs[i]) for i in range(len(As))]

    # Figure for attribution scores
    fig_a, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    fig_a.suptitle(f"Subnetwork Analysis (Step {step})")
    plot_matrix(
        ax,
        attribution_scores_normed,
        "Normalized attribution Scores",
        "Subnetwork index",
        "Function index",
    )

    # Figures for A, B, AB of each layer
    n_rows = 3 + model.k if slow_images else 3
    n_cols = 4
    figsize = (8 * n_cols, 4 + 4 * n_rows)
    figs = [plt.figure(figsize=figsize, constrained_layout=True) for _ in range(n_layers)]
    # Plot normalized attribution scores

    for n in range(n_layers):
        fig = figs[n]
        gs = fig.add_gridspec(n_rows, n_cols)
        plot_matrix(
            fig.add_subplot(gs[0, 0]),
            As[2 * n],
            f"A (W_in, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[0, 1:]),
            Bs[2 * n],
            f"B (W_in, layer {n})",
            "Neuron index",
            "Subnetwork index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, 0]),
            Bs[2 * n + 1].T,
            f"B (W_out, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, 1:]),
            As[2 * n + 1].T,
            f"A (W_out, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, :2]),
            ABs[2 * n],
            f"AB summed (W_in, layer {n})",
            "Neuron index",
            "Embedding index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, 2:]),
            ABs[2 * n + 1].T,
            f"AB.T  summed (W_out.T, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        if slow_images:
            for k in range(model.k):
                plot_matrix(
                    fig.add_subplot(gs[3 + k, :2]),
                    ABs_by_k[2 * n][k],
                    f"AB k={k} (W_in, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )
                plot_matrix(
                    fig.add_subplot(gs[3 + k, 2:]),
                    ABs_by_k[2 * n + 1][k].T,
                    f"AB.T k={k} (W_out.T, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )
    return {"attrib_scores": fig_a, **{f"matrices_layer{n}": fig for n, fig in enumerate(figs)}}


class SPDoutputs(NamedTuple):
    target_model_output: torch.Tensor | None
    spd_model_output: torch.Tensor
    spd_topk_model_output: torch.Tensor
    layer_acts: list[torch.Tensor]
    topk_layer_acts: list[torch.Tensor]
    inner_acts: list[torch.Tensor]
    topk_inner_acts: list[torch.Tensor]
    attribution_scores: torch.Tensor
    topk_mask: torch.Tensor


def run_spd_forward_pass(
    spd_model: PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer,
    target_model: PiecewiseFunctionTransformer | None,
    input_array: torch.Tensor,
    full_rank: bool,
    batch_topk: bool,
    topk: float,
) -> SPDoutputs:
    # non-SPD model, and SPD-model non-topk forward pass
    model_output_hardcoded = target_model(input_array) if target_model is not None else None
    model_output_spd, layer_acts, inner_acts = spd_model(input_array)

    # SPD-model topk forward pass, copy-pasted from run_spd
    if full_rank:
        attribution_scores = calc_attributions_full_rank(
            out=model_output_spd,
            inner_acts=inner_acts,
            layer_acts=layer_acts,
        )
    else:
        attribution_scores = calc_attributions_rank_one(out=model_output_spd, inner_acts=inner_acts)
    topk_mask = calc_topk_mask(attribution_scores, topk, batch_topk=batch_topk)
    model_output_spd_topk, layer_acts_topk, inner_acts_topk = spd_model.forward_topk(
        input_array, topk_mask=topk_mask
    )
    assert len(inner_acts_topk) == spd_model.n_param_matrices
    attribution_scores = attribution_scores.cpu().detach()
    return SPDoutputs(
        target_model_output=model_output_hardcoded,
        spd_model_output=model_output_spd,
        spd_topk_model_output=model_output_spd_topk,
        layer_acts=layer_acts,
        topk_layer_acts=layer_acts_topk,
        inner_acts=inner_acts,
        topk_inner_acts=inner_acts_topk,
        attribution_scores=attribution_scores,
        topk_mask=topk_mask,
    )


def plot_model_functions(
    spd_model: PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer,
    target_model: PiecewiseFunctionTransformer | None,
    full_rank: bool,
    device: str,
    start: float,
    stop: float,
    print_info: bool = False,
) -> dict[str, plt.Figure]:
    fig, axes = plt.subplots(nrows=3, figsize=(12, 12), constrained_layout=True)
    assert isinstance(axes, np.ndarray)
    [ax, ax_attrib, ax_inner] = axes
    # For these tests we run with unusual data where there's always 1 control bit active, which
    # might differ from training. Thus we manually set topk to 1. Note that this should work for
    # both, batch and non-batch topk.
    topk = 1
    # Disable batch_topk to rule out errors caused by batch -- non-batch is an easier task for SPD
    # in the toy setting used for plots.
    batch_topk = False
    fig.suptitle(
        f"Model outputs for each control bit. (Plot with topk={topk} & batch_topk={batch_topk}.\n"
        "This should differ from the training settings, topk>=1 ought to work for plotting.)"
    )
    # Get model outputs for simple example data. Create input array with 10_000 rows, 1000
    # rows for each function. Set the 0th column to be linspace(0, 5, 1000) repeated. Set the
    # control bits to [0,1,0,0,...] for the first 1000 rows, [0,0,1,0,...] for the next 1000 rows,
    # etc.
    n_samples = 1000
    n_functions = spd_model.num_functions
    # Set the control bits
    input_array = torch.eye(spd_model.n_inputs, dtype=torch.float32)[-n_functions:, :]
    input_array = input_array.repeat_interleave(n_samples, dim=0)
    input_array = input_array.to(device)
    # Set the 0th input to x_space
    x_space = torch.linspace(start, stop, n_samples)
    input_array[:, 0] = x_space.repeat(n_functions)

    spd_outputs = run_spd_forward_pass(
        spd_model=spd_model,
        target_model=target_model,
        input_array=input_array,
        full_rank=full_rank,
        batch_topk=batch_topk,
        topk=topk,
    )
    model_output_hardcoded = spd_outputs.target_model_output
    model_output_spd = spd_outputs.spd_model_output
    out_topk = spd_outputs.spd_topk_model_output
    inner_acts = spd_outputs.inner_acts
    attribution_scores = spd_outputs.attribution_scores
    topk_mask = spd_outputs.topk_mask

    if print_info:
        # Check if, ever, there are cases where the control bit is 1 but the topk_mask is False.
        # We check this by calculating whether topk_mask is True OR control bit is 0.
        control_bits = input_array[:, 1:].cpu().detach().numpy()
        topk_mask = topk_mask.cpu().detach().numpy()
        topk_mask_control_bits = topk_mask | (control_bits == 0)
        print(
            f"How often is topk_mask True or control_bits == 0: {topk_mask_control_bits.mean():.3%}"
        )
        if model_output_hardcoded is not None:
            # Calculate recon loss
            topk_recon_loss = calc_recon_mse(
                out_topk, model_output_hardcoded, has_instance_dim=False
            )
            print(f"Topk recon loss: {topk_recon_loss:.4f}")

    # Convert stuff to numpy
    model_output_spd = model_output_spd[:, 0].cpu().detach().numpy()
    if model_output_hardcoded is not None:
        model_output_hardcoded = model_output_hardcoded[:, 0].cpu().detach().numpy()
    out_topk = out_topk.cpu().detach().numpy()
    input_xs = input_array[:, 0].cpu().detach().numpy()

    # Plot for every k
    tab20 = plt.get_cmap("tab20")
    # cb stands for control bit which is active there; this differs from k due to permutation
    for cb in range(n_functions):
        d = 1 / n_functions
        color0 = tab20(cb / n_functions)
        color1 = tab20(cb / n_functions + d / 4)
        color2 = tab20(cb / n_functions + 2 * d / 4)
        color3 = tab20(cb / n_functions + 3 * d / 4)
        s = slice(cb * n_samples, (cb + 1) * n_samples)
        if model_output_hardcoded is not None:
            assert target_model is not None
            assert target_model.controlled_resnet is not None
            ax.plot(
                x_space,
                target_model.controlled_resnet.functions[cb](x_space),
                ls=":",
                color=color0,
            )
            ax.plot(input_xs[s], model_output_hardcoded[s], label=f"cb={cb}", color=color1)
        ax.plot(input_xs[s], model_output_spd[s], ls="-.", color=color2)
        ax.plot(input_xs[s], out_topk[s], ls="--", color=color3)
        k_cb = attribution_scores[s].mean(dim=0).argmax()
        for k in range(n_functions):
            # Find permutation
            if k == k_cb:
                ax_attrib.plot(
                    input_xs[s],
                    attribution_scores[s][:, k],
                    color=color1,
                    label=f"cb={cb}, k_cb={k}",
                )
                assert len(inner_acts) <= 2, "Didn't implement more than 2 SPD 'layers' yet"
                for j in range(len(inner_acts)):
                    ls = ["-", "--"][j]
                    if not isinstance(spd_model, PiecewiseFunctionSPDFullRankTransformer):
                        ax_inner.plot(
                            input_xs[s],
                            inner_acts[j].cpu().detach()[s][:, k],
                            color=color1,
                            ls=ls,
                            label=f"cb={cb}, k_cb={k}" if j == 0 else None,
                        )
            else:
                ax_attrib.plot(input_xs[s], attribution_scores[s][:, k], color=color1, alpha=0.2)
                for j in range(len(inner_acts)):
                    ls = ["-", "--"][j]
                    if not isinstance(spd_model, PiecewiseFunctionSPDFullRankTransformer):
                        ax_inner.plot(
                            input_xs[s],
                            inner_acts[j].cpu().detach()[s][:, k],
                            color="k",
                            ls=ls,
                            lw=0.2,
                        )
    ax_inner.plot([], [], color="C0", label="W_in", ls="-")
    ax_inner.plot([], [], color="C0", label="W_out", ls="--")
    ax_inner.plot([], [], color="k", label="k!=k_cb", ls="-", lw=0.2)

    # Add some additional (blue) legend lines explaining the different line styles
    if model_output_hardcoded is not None:
        ax.plot([], [], ls=":", color="C0", label="true function")
        ax.plot([], [], ls="-", color="C0", label="target model")
    ax.plot([], [], ls="-.", color="C0", label="spd model")
    ax.plot([], [], ls="--", color="C0", label="spd model topk")
    ax.legend(ncol=3)
    ax_attrib.legend(ncol=3)
    ax_attrib.set_yscale("log")
    ax_attrib.set_ylabel("attribution_scores (log)")
    ax_attrib.set_xlabel("x (model input dim 0)")
    ax_attrib.set_title(
        "Attributions of each subnetwork for every control bit case (k=k_cb in bold)"
    )
    ax_inner.legend(ncol=3)
    ax_inner.set_ylabel("inner acts (symlog)")
    ax_inner.set_title("'inner acts', coloured for the top subnetwork, black for the others")
    ax_inner.set_xlabel("x (model input dim 0)")
    ax.set_xlabel("x (model input dim 0)")
    ax.set_ylabel("f(x) (model output dim 0)")
    return {"model_functions": fig}


def plot_subnetwork_correlations(
    dataloader: BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
    spd_model: PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer,
    config: Config,
    device: str,
    n_forward_passes: int = 100,
) -> dict[str, plt.Figure]:
    topk_masks = []
    for batch, _ in dataloader:
        batch = batch.to(device=device)
        assert config.topk is not None
        spd_outputs = run_spd_forward_pass(
            spd_model=spd_model,
            target_model=None,
            input_array=batch,
            full_rank=config.full_rank,
            batch_topk=config.batch_topk,
            topk=config.topk,
        )
        topk_masks.append(spd_outputs.topk_mask)
        if len(topk_masks) > n_forward_passes:
            break
    topk_masks = torch.cat(topk_masks).float()
    # Calculate correlation matrix
    corr_matrix = torch.corrcoef(topk_masks.T).cpu()
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    im = ax.matshow(corr_matrix)
    ax.xaxis.set_ticks_position("bottom")
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="#EE7777",
                fontsize=8,
            )
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax.set_title("Subnetwork Correlation Matrix")
    ax.set_xlabel("Subnetwork")
    ax.set_ylabel("Subnetwork")
    return {"subnetwork_correlation_matrix": fig}
