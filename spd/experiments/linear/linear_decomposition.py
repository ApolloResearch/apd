"""Linear decomposition script."""

from pathlib import Path

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearComponentModel,
    DeepLinearModel,
)
from spd.log import logger
from spd.run_spd import Config, DeepLinearConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    calc_attributions_full_rank,
    calc_attributions_rank_one,
    calc_topk_mask,
    init_wandb,
    load_config,
    permute_to_identity,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"topkrecon{config.topk_recon_coeff}_"
            f"lpsp{config.lp_sparsity_coeff}_"
            f"topkl2_{config.topk_l2_coeff}_"
            f"bs{config.batch_size}_"
        )
    return config.wandb_run_name_prefix + run_suffix


def _plot_multiple_subnetwork_params(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel, step: int
) -> plt.Figure:
    """Plot each subnetwork parameter matrix."""
    all_params = model.all_subnetwork_params()
    # Each param (of which there are n_layers): [n_instances, k, n_features, n_features]
    n_params = len(all_params)
    assert n_params >= 1

    n_instances, k, dim1, dim2 = all_params[0].shape

    fig, axs = plt.subplots(
        k,
        n_instances * n_params,
        # + n_params - 1 to account for the space between the subplots
        figsize=(2 * n_instances * n_params + n_params - 1, 2 * k),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for inst_idx in range(n_instances):
        for param_idx in range(n_params):
            for k_idx in range(k):
                row_idx = k_idx
                column_idx = param_idx + inst_idx * n_params
                ax = axs[row_idx, column_idx]  # type: ignore
                param = all_params[param_idx][inst_idx, k_idx].detach().cpu().numpy()
                ax.matshow(param, cmap="RdBu", norm=CenteredNorm())
                ax.set_xticks([])
                ax.set_yticks([])

                if column_idx == 0:
                    ax.set_ylabel(f"k={k_idx}", rotation=0, ha="right", va="center")
                if k_idx == k - 1:
                    ax.set_xlabel(f"Inst {inst_idx} Param {param_idx}", rotation=0, ha="center")

    fig.suptitle(f"Subnetwork Parameters (Step {step})")
    return fig


def _plot_subnetwork_activations_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    inner_acts: list[Float[Tensor, "batch n_instances k"]],
) -> plt.Figure:
    """Plot the inner acts for the first batch_elements in the batch.

    The first row is the raw batch information, the following rows are the inner acts per layer.
    """
    n_layers = len(inner_acts)
    n_instances = batch.shape[1]

    fig, axs = plt.subplots(
        n_layers + 1,
        n_instances,
        figsize=(2.5 * n_instances, 2.5 * (n_layers + 1)),
        squeeze=False,
        sharey=True,
    )

    cmap = "Blues"
    # Add the batch data
    for i in range(n_instances):
        ax = axs[0, i]
        data = batch[:, i, :].detach().cpu().float().numpy()
        ax.matshow(data, vmin=0, vmax=np.max(data), cmap=cmap)

        ax.set_title(f"Instance {i}")
        if i == 0:
            ax.set_ylabel("Inputs")
        elif i == n_instances - 1:
            ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
            ax.yaxis.set_label_position("right")

        # Set an xlabel for each plot
        ax.set_xlabel("n_features")

        ax.set_xticks([])
        ax.set_yticks([])

    # Add the inner acts
    for layer in range(n_layers):
        for i in range(n_instances):
            ax = axs[layer + 1, i]
            instance_data = inner_acts[layer][:, i, :].abs().detach().cpu().float().numpy()
            ax.matshow(instance_data, vmin=0, vmax=np.max(instance_data), cmap=cmap)

            if i == 0:
                ax.set_ylabel(f"h_{layer}")
            elif i == n_instances - 1:
                ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
                ax.yaxis.set_label_position("right")

            if layer == n_layers - 1:
                ax.set_xlabel("k")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


def _collect_permuted_subnetwork_activations(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel,
    device: str,
    topk: float | None = None,
    batch_topk: bool = True,
) -> tuple[
    Float[Tensor, "batch n_instances n_features"], list[Float[Tensor, "batch n_instances k"]]
]:
    """
    Collect subnetwork activations and permute them for visualization.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions, and then permutes them to align with the identity.

    Args:
        model (DeepLinearComponentModel | DeepLinearComponentFullRankModel): The model to collect
            attributions on.
        device (str): The device to run computations on.
        topk (int): The number of topk indices to use for the forward pass.

    Returns:
        - The input test batch (identity matrix expanded over instance dimension).
        - A list of permuted attributions for each layer.

    """
    test_batch = einops.repeat(
        torch.eye(model.n_features, device=device),
        "batch n_features -> batch n_instances n_features",
        n_instances=model.n_instances,
    )

    out, test_layer_acts, test_inner_acts = model(test_batch)
    if topk is not None:
        if isinstance(model, DeepLinearComponentModel):
            attribution_scores = calc_attributions_rank_one(out=out, inner_acts=test_inner_acts)
        else:
            assert isinstance(model, DeepLinearComponentFullRankModel)
            attribution_scores = calc_attributions_full_rank(
                out=out, inner_acts=test_inner_acts, layer_acts=test_layer_acts
            )
        topk_mask = calc_topk_mask(attribution_scores, topk, batch_topk=batch_topk)

        test_inner_acts = model.forward_topk(test_batch, topk_mask=topk_mask)[-1]

    # Full rank has an extra n_features dimension which we sum over
    if isinstance(model, DeepLinearComponentFullRankModel):
        for i in range(len(test_inner_acts)):
            test_inner_acts[i] = einops.einsum(
                test_inner_acts[i], "batch n_instances k n_features -> batch n_instances k"
            )

    test_inner_acts_permuted = []
    for layer in range(model.n_layers):
        test_inner_acts_layer_permuted = []
        for i in range(model.n_instances):
            test_inner_acts_layer_permuted.append(
                permute_to_identity(test_inner_acts[layer][:, i, :].abs())
            )
        test_inner_acts_permuted.append(torch.stack(test_inner_acts_layer_permuted, dim=1))

    return test_batch, test_inner_acts_permuted


def make_linear_plots(
    model: DeepLinearComponentModel,
    step: int,
    out_dir: Path | None,
    device: str,
    topk: float | None,
    batch_topk: bool,
    **_,
) -> dict[str, plt.Figure]:
    test_batch, test_inner_acts = _collect_permuted_subnetwork_activations(
        model, device, topk, batch_topk=batch_topk
    )

    act_fig = _plot_subnetwork_activations_fn(batch=test_batch, inner_acts=test_inner_acts)
    if out_dir is not None:
        act_fig.savefig(out_dir / f"inner_acts_{step}.png")
    plt.close(act_fig)

    param_fig = _plot_multiple_subnetwork_params(model, step)
    if out_dir is not None:
        param_fig.savefig(out_dir / f"subnetwork_params_{step}.png", dpi=300, bbox_inches="tight")
    plt.close(param_fig)

    if out_dir is not None:
        tqdm.write(f"Saved inner_acts to {out_dir / f'inner_acts_{step}.png'}")
        tqdm.write(f"Saved subnetwork_params to {out_dir / f'subnetwork_params_{step}.png'}")
    return {"inner_acts": act_fig, "subnetwork_params": param_fig}


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
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, DeepLinearConfig)

    if config.task_config.pretrained_model_path is not None:
        dl_model = DeepLinearModel.from_pretrained(config.task_config.pretrained_model_path).to(
            device
        )
        assert (
            config.task_config.n_features is None
            and config.task_config.n_layers is None
            and config.task_config.n_instances is None
        ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
        n_features = dl_model.n_features
        n_layers = dl_model.n_layers
        n_instances = dl_model.n_instances
    else:
        assert config.out_recon_coeff is not None, "Only out recon loss allows no pretrained model"
        dl_model = None
        n_features = config.task_config.n_features
        n_layers = config.task_config.n_layers
        n_instances = config.task_config.n_instances
        assert (
            n_features is not None and n_layers is not None and n_instances is not None
        ), "n_features, n_layers, and n_instances must be set"

    if config.full_rank:
        dlc_model = DeepLinearComponentFullRankModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        ).to(device)
    else:
        dlc_model = DeepLinearComponentModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        ).to(device)

    dataset = DeepLinearDataset(n_features, n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimize(
        model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=dl_model,
        plot_results_fn=make_linear_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
