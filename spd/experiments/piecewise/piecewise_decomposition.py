"""Linear decomposition script."""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from tqdm import tqdm

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.log import logger
from spd.run_spd import Config, PiecewiseConfig, calc_recon_mse, optimize
from spd.utils import (
    BatchedDataLoader,
    calc_attributions_rank_one,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def plot_components(
    model: PiecewiseFunctionSPDTransformer,
    step: int,
    out_dir: Path | None,
    device: str,
    slow_images: bool,
    **_,
) -> plt.Figure:
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
        fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
        if ylabel == "Function index":
            n_functions = matrix.shape[0]
            ax.set_yticks(range(n_functions))
            ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])

    # Create figure with subplots using gridspec
    n_rows = 3 + model.k if slow_images else 3
    n_cols = 4
    figsize = (8 * n_cols, 4 + 4 * n_rows)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)
    plt.suptitle(f"Subnetwork Analysis (Step {step})")

    # Plot attribution scores
    plot_matrix(
        fig.add_subplot(gs[0, 0]),
        attribution_scores,
        "Raw attribution Scores",
        "Subnetwork index",
        "Function index",
    )
    # Plot normalized attribution scores
    plot_matrix(
        fig.add_subplot(gs[0, 1]),
        attribution_scores_normed,
        "Normalized attribution Scores",
        "Subnetwork index",
        "Function index",
    )

    assert n_layers == 1, "Current implementation only supports 1 layer"
    for n in range(n_layers):
        plot_matrix(
            fig.add_subplot(gs[0, 2]),
            As[2 * n],
            f"A (W_in, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[0, 3]),
            Bs[2 * n + 1].T,
            f"B (W_out, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, :2]),
            Bs[2 * n],
            f"B (W_in, layer {n})",
            "Neuron index",
            "Subnetwork index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, 2:]),
            As[2 * n + 1].T,
            f"A (W_out, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, :2]),
            ABs[n],
            f"AB summed (W_in, layer {n})",
            "Neuron index",
            "Embedding index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, 2:]),
            ABs[n + 1].T,
            f"AB.T  summed (W_out.T, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        if slow_images:
            for k in range(model.k):
                plot_matrix(
                    fig.add_subplot(gs[3 + k, :2]),
                    ABs_by_k[n][k],
                    f"AB k={k} (W_in, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )
                plot_matrix(
                    fig.add_subplot(gs[3 + k, 2:]),
                    ABs_by_k[n + 1][k].T,
                    f"AB.T k={k} (W_out.T, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )

    if out_dir:
        fig.savefig(out_dir / f"subnetwork_analysis_{step}.png", dpi=300)
        plt.close(fig)
        tqdm.write(f"Saved subnetwork analysis to {out_dir / f'subnetwork_analysis_{step}.png'}\n")

    return fig


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, PiecewiseConfig)
        run_suffix = (
            f"lay{config.task_config.n_layers}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"topkrecon{config.topk_recon_coeff}_"
            f"lpsp{config.lp_sparsity_coeff}_"
            f"topkl2_{config.topk_l2_coeff}_"
            f"bs{config.batch_size}"
        )
        if config.task_config.handcoded_AB:
            run_suffix += "_hAB"
    return config.wandb_run_name_prefix + run_suffix


def get_model_and_dataloader(
    config: Config,
    device: str,
    out_dir: Path | None = None,
) -> tuple[
    PiecewiseFunctionTransformer,
    PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer,
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
]:
    """Set up the piecewise models and dataset."""
    assert isinstance(config.task_config, PiecewiseConfig)
    functions, function_params = generate_trig_functions(config.task_config.n_functions)

    if out_dir:
        with open(out_dir / "function_params.json", "w") as f:
            json.dump(function_params, f, indent=4)
        logger.info(f"Saved function params to {out_dir / 'function_params.json'}")

    piecewise_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        seed=config.seed,
        simple_bias=config.task_config.simple_bias,
    ).to(device)
    piecewise_model.eval()

    input_biases = [
        piecewise_model.mlps[i].input_layer.bias.detach().clone()
        for i in range(piecewise_model.n_layers)
    ]
    if config.full_rank:
        piecewise_model_spd = PiecewiseFunctionSPDFullRankTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            input_biases=input_biases,
        )
    else:
        piecewise_model_spd = PiecewiseFunctionSPDTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            input_biases=input_biases,
        )
    if config.task_config.handcoded_AB:
        logger.info("Setting handcoded A and B matrices (!)")
        piecewise_model_spd.set_handcoded_AB(piecewise_model)
    piecewise_model_spd.to(device)

    # Set requires_grad to False for all embeddings and all input biases
    for i in range(piecewise_model_spd.n_layers):
        piecewise_model_spd.mlps[i].bias1.requires_grad_(False)
    piecewise_model_spd.W_E.requires_grad_(False)
    piecewise_model_spd.W_U.requires_grad_(False)

    dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        batch_size=config.batch_size,
        return_labels=False,
    )
    dataloader = BatchedDataLoader(dataset)

    test_dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        batch_size=config.batch_size,
        return_labels=True,
    )
    test_dataloader = BatchedDataLoader(test_dataset)

    return piecewise_model, piecewise_model_spd, dataloader, test_dataloader


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)

    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, PiecewiseConfig)
    assert config.task_config.k is not None

    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    piecewise_model, piecewise_model_spd, dataloader, test_dataloader = get_model_and_dataloader(
        config, device, out_dir
    )

    # Evaluate the hardcoded model on 5 batches to get the labels
    n_batches = 5
    loss = 0

    for i, (batch, labels) in enumerate(test_dataloader):
        if i >= n_batches:
            break
        hardcoded_out = piecewise_model(batch.to(device))
        loss += calc_recon_mse(hardcoded_out, labels.to(device))
    loss /= n_batches
    logger.info(f"Loss of hardcoded model on 5 batches: {loss}")

    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model=piecewise_model,
        dataloader=dataloader,
        plot_results_fn=plot_components,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
