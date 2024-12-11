"""Residual Linear decomposition script."""

import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from jaxtyping import Bool, Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
    ResidualMLPSPDRankPenaltyConfig,
    ResidualMLPSPDRankPenaltyModel,
)
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    plot_individual_feature_response,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import (
    ResidualMLPDataset,
)
from spd.log import logger
from spd.plotting import (
    plot_subnetwork_attributions_statistics,
    plot_subnetwork_correlations,
)
from spd.run_spd import (
    Config,
    ResidualMLPTaskConfig,
    get_common_run_name_suffix,
    optimize,
)
from spd.utils import (
    DatasetGeneratedDataLoader,
    collect_subnetwork_attributions,
    load_config,
    run_spd_forward_pass,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, n_features: int, n_layers: int, d_resid: int, d_mlp: int) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
    return config.wandb_run_name_prefix + run_suffix


def calc_dead_subnets(
    model: ResidualMLPSPDRankPenaltyModel, cutoff: float = 1e-2
) -> Bool[Tensor, "n_instances k"]:
    """We consider a subnet dead if it has a linear2 L2 of < cutoff."""
    linear1_l2 = (
        model.all_subnetwork_params()["layers.0.linear2"][:, :, :, :].pow(2).sum(dim=(-1, -2))
    )
    return linear1_l2 < cutoff


def plot_subnetwork_attributions(
    attribution_scores: Float[Tensor, "batch n_instances k"],
    out_dir: Path | None,
    step: int | None,
) -> plt.Figure:
    """Plot subnetwork attributions."""
    # Plot a row with n_instances
    # Each column is a different instance
    n_instances = attribution_scores.shape[1]
    fig, ax = plt.subplots(
        nrows=1, ncols=n_instances, figsize=(5 * n_instances, 5), constrained_layout=True
    )
    axs = np.array([ax]) if n_instances == 1 else np.array(ax)
    im = None
    for i in range(n_instances):
        im = axs[i].matshow(
            attribution_scores[:, i].detach().cpu().numpy(), aspect="auto", cmap="Reds"
        )
        axs[i].set_xlabel("Subnetwork Index")
        axs[i].set_ylabel("Batch Index")
        axs[i].set_title("Subnetwork Attributions")

        # Annotate each cell with the numeric value if less than 200 elements
        if attribution_scores.shape[0] * attribution_scores.shape[-1] < 200:
            for b in range(attribution_scores.shape[0]):
                for j in range(attribution_scores.shape[-1]):
                    axs[i].text(
                        j,
                        b,
                        f"{attribution_scores[b, i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )
    plt.colorbar(im)
    if out_dir:
        filename = (
            f"subnetwork_attributions_s{step}.png"
            if step is not None
            else "subnetwork_attributions.png"
        )
        fig.savefig(out_dir / filename, dpi=200)
    return fig


def plot_multiple_subnetwork_params(
    model: ResidualMLPSPDRankPenaltyModel,
    out_dir: Path | None,
    step: int | None = None,
) -> plt.Figure:
    """Plot each subnetwork parameter matrix."""
    all_params = model.all_subnetwork_params()
    # Each param (of which there are n_layers): [k, n_features, n_features]
    n_params = len(all_params)
    param_names = list(all_params.keys())

    weight_param = [param for param_name, param in all_params.items() if "linear" in param_name][0]
    n_instances, k, dim1, dim2 = weight_param.shape

    # Find global min and max for normalization
    all_values = []
    for param_name in param_names:
        param_values = all_params[param_name].detach().cpu().numpy()
        all_values.append(param_values)
    all_values_concat = np.concatenate([v.flatten() for v in all_values])
    vmax = np.abs(all_values_concat).max()
    norm = CenteredNorm(vcenter=0, halfrange=vmax)

    fig, axs = plt.subplots(
        n_instances * n_params,
        k,
        figsize=(2 * k, n_instances * n_params),
        constrained_layout=False,
    )
    axs = np.array(axs)

    for instance_idx in range(n_instances):
        for param_idx in range(n_params):
            param_name = param_names[param_idx]
            for subnet_idx in range(k):
                col_idx = subnet_idx
                row_idx = instance_idx * n_params + param_idx

                ax = axs[row_idx, col_idx]  # type: ignore
                param = all_params[param_name][instance_idx, subnet_idx].detach().cpu().numpy()
                # If it's a bias with a single dimension, unsqueeze it
                if param.ndim == 1:
                    param = param[:, None]

                # Set aspect ratio based on parameter dimensions
                height, width = param.shape
                aspect = width / height

                im = ax.matshow(param, cmap="RdBu", norm=norm, aspect=aspect)
                ax.set_xticks([])
                ax.set_yticks([])

                if col_idx == 0:
                    ax.set_ylabel(
                        f"Inst.{instance_idx}.{param_name}",
                        rotation=0,
                        ha="right",
                        va="center",
                    )

                if row_idx == ((n_instances * n_params) - 1):
                    ax.set_xlabel(f"Subnet {subnet_idx}", rotation=0, ha="center", va="top")

    # Add colorbar
    fig.colorbar(im, ax=axs.ravel().tolist(), location="right")  # type: ignore

    title_text = "Subnet Parameters"
    if step is not None:
        title_text += f" (Step {step})"
    fig.suptitle(title_text)
    if out_dir:
        fig.savefig(out_dir / f"subnetwork_params_s{step}.png", dpi=200)
    return fig


def resid_mlp_plot_results_fn(
    model: ResidualMLPSPDRankPenaltyModel,
    target_model: ResidualMLPModel,
    step: int | None,
    out_dir: Path | None,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, " batch_size k"] | None,
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]
    ]
    | None = None,
    **_,
) -> dict[str, plt.Figure]:
    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    fig_dict = {}

    # Save the number of dead subnets to wandb
    dead_subnets = calc_dead_subnets(model, cutoff=1e-2)
    n_dead_subnets: int | list[int] = dead_subnets.sum(dim=-1).detach().cpu().tolist()
    if config.wandb_project:
        wandb.log({"n_dead_subnets": n_dead_subnets}, step=step)
    logger.info(f"Number of dead subnets at step {step}: {n_dead_subnets}")

    ############################################################################################
    # Feature contributions
    ############################################################################################
    fig1, fig2 = plot_spd_relu_contribution(model, target_model, device)
    fig1.suptitle("How much does each ReLU contribute to each feature?")
    fig2.suptitle("How much does each feature route through each ReLU?")
    fig_dict["feature_contributions"] = fig1
    fig_dict["relu_contributions"] = fig2

    fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
    fig1.suptitle("How much does each ReLU contribute to each feature?")
    fig2.suptitle("How much does each feature route through each ReLU?")
    fig_dict["cropped_feature_contributions"] = fig1
    fig_dict["cropped_relu_contributions"] = fig2

    ############################################################################################
    # Individual feature responses + per-feature performance
    ############################################################################################
    def spd_model_fn(
        batch: Float[Tensor, "batch n_instances"],
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert config.topk is not None
        return run_spd_forward_pass(
            spd_model=model,
            target_model=target_model,
            input_array=batch,
            attribution_type=config.attribution_type,
            batch_topk=config.batch_topk,
            topk=config.topk,
            distil_from_target=config.distil_from_target,
        ).spd_topk_model_output

    def target_model_fn(
        batch: Float[Tensor, "batch n_instances"],
    ) -> Float[Tensor, "batch n_instances n_features"]:
        return target_model(batch)[0]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)
    axes = np.atleast_2d(axes)  # type: ignore
    plot_individual_feature_response(
        model_fn=target_model_fn,
        device=device,
        model_config=model.config,
        ax=axes[0, 0],
    )
    plot_individual_feature_response(
        model_fn=target_model_fn,
        device=device,
        model_config=model.config,
        sweep=True,
        ax=axes[1, 0],
    )
    plot_individual_feature_response(
        model_fn=spd_model_fn,
        device=device,
        model_config=model.config,
        ax=axes[0, 1],
    )
    plot_individual_feature_response(
        model_fn=spd_model_fn,
        device=device,
        model_config=model.config,
        sweep=True,
        ax=axes[1, 1],
    )
    axes[0, 0].set_ylabel(axes[0, 0].get_title())
    axes[1, 0].set_ylabel(axes[1, 0].get_title())
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].set_title("Target model")
    axes[0, 1].set_title("SPD model")
    axes[1, 0].set_title("")
    axes[1, 1].set_title("")
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    fig_dict["individual_feature_responses"] = fig

    fig, ax = plt.subplots(figsize=(15, 5))
    sorted_indices = analyze_per_feature_performance(
        model_fn=target_model_fn,
        model_config=target_model.config,
        ax=ax,
        label="Target",
        device=device,
        sorted_indices=None,
    )
    analyze_per_feature_performance(
        model_fn=spd_model_fn,
        model_config=model.config,
        ax=ax,
        label="SPD",
        device=device,
        sorted_indices=sorted_indices,
    )
    ax.legend()
    fig_dict["loss_by_feature"] = fig

    ############################################################################################
    # Virtual weights
    ############################################################################################

    fig = plot_virtual_weights_target_spd(target_model, model, device)
    fig_dict["virtual_weights"] = fig

    ############################################################################################
    # Subnetwork attributions
    ############################################################################################
    assert config.spd_type in ("full_rank", "rank_penalty")
    attribution_scores = collect_subnetwork_attributions(
        spd_model=model,
        target_model=target_model,
        device=device,
        n_instances=model.n_instances,
    )
    fig_dict["subnetwork_attributions"] = plot_subnetwork_attributions(
        attribution_scores, out_dir, step
    )

    if config.topk is not None:
        if dataloader is not None and config.task_config.k > 1:
            fig_dict_correlations = plot_subnetwork_correlations(
                dataloader=dataloader,
                target_model=target_model,
                spd_model=model,
                config=config,
                device=device,
            )
            fig_dict.update(fig_dict_correlations)

        assert topk_mask is not None
        fig_dict_attributions = plot_subnetwork_attributions_statistics(topk_mask=topk_mask)
        fig_dict.update(fig_dict_attributions)

    ############################################################################################
    # Subnetwork parameters
    ############################################################################################

    fig_dict["subnetwork_params"] = plot_multiple_subnetwork_params(
        model=model, out_dir=out_dir, step=step
    )

    # Save plots to files
    if out_dir:
        for k, v in fig_dict.items():
            out_file = out_dir / f"{k}_s{step}.png"
            v.savefig(out_file, dpi=100)
            tqdm.write(f"Saved plot to {out_file}")
    return fig_dict


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    resid_mlp: ResidualMLPModel,
    resid_mlp_train_config_dict: dict[str, Any],
    label_coeffs: Float[Tensor, " n_instances"],
) -> None:
    torch.save(resid_mlp.state_dict(), out_dir / "resid_mlp.pth")

    with open(out_dir / "resid_mlp_train_config.yaml", "w") as f:
        yaml.dump(resid_mlp_train_config_dict, f, indent=2)

    with open(out_dir / "label_coeffs.json", "w") as f:
        json.dump(label_coeffs.detach().cpu().tolist(), f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "resid_mlp.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "resid_mlp_train_config.yaml"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualMLPTaskConfig)

    target_model, target_model_train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(
        config,
        n_features=target_model.config.n_features,
        n_layers=target_model.config.n_layers,
        d_resid=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        resid_mlp=target_model,
        resid_mlp_train_config_dict=target_model_train_config_dict,
        label_coeffs=label_coeffs,
    )

    # Create the SPD model
    if config.spd_type == "rank_penalty":
        model_config = ResidualMLPSPDRankPenaltyConfig(
            **target_model.config.model_dump(mode="json"), k=config.task_config.k, m=config.m
        )
        model = ResidualMLPSPDRankPenaltyModel(config=model_config).to(device)
    else:
        raise ValueError(f"Unknown/unsupported spd_type: {config.spd_type}")

    # Use the target_model's embedding matrix and don't train it further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    first_pass_model, _, _ = ResidualMLPSPDRankPenaltyModel.from_pretrained(
        "wandb:spd-resid-mlp/runs/fstzjcl7"
    )
    first_pass_model.to(device)

    dead_subnets: Bool[Tensor, "n_instances k"] | Bool[Tensor, " k"] = calc_dead_subnets(
        first_pass_model, cutoff=1e-2
    )

    # Copy over the alive subnets from first_pass_model to SPD model
    for i in range(model.config.n_layers):
        model.layers[i].linear1.A.data[:, :, :, :] = torch.where(
            dead_subnets[:, :, None, None],
            model.layers[i].linear1.A.data[:, :, :, :],
            first_pass_model.layers[i].linear1.A.data[:, :, :, :],
        )
        model.layers[i].linear1.B.data[:, :, :, :] = torch.where(
            dead_subnets[:, :, None, None],
            model.layers[i].linear1.B.data[:, :, :, :],
            first_pass_model.layers[i].linear1.B.data[:, :, :, :],
        )
        model.layers[i].linear2.A.data[:, :, :, :] = torch.where(
            dead_subnets[:, :, None, None],
            model.layers[i].linear2.A.data[:, :, :, :],
            first_pass_model.layers[i].linear2.A.data[:, :, :, :],
        )
        model.layers[i].linear2.B.data[:, :, :, :] = torch.where(
            dead_subnets[:, :, None, None],
            model.layers[i].linear2.B.data[:, :, :, :],
            first_pass_model.layers[i].linear2.B.data[:, :, :, :],
        )

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(target_model.config.n_layers):
        if target_model.config.in_bias:
            model.layers[i].linear1.bias.data[:, :] = (
                target_model.layers[i].bias1.data.detach().clone()
            )
            model.layers[i].linear1.bias.requires_grad = False
        if target_model.config.out_bias:
            model.layers[i].linear2.bias.data[:, :] = (
                target_model.layers[i].bias2.data.detach().clone()
            )
            model.layers[i].linear2.bias.requires_grad = False

    param_map = {}
    for i in range(target_model.config.n_layers):
        # Map from pretrained model's `all_decomposable_params` to the SPD models'
        # `all_subnetwork_params_summed`.
        param_map[f"layers.{i}.linear1"] = f"layers.{i}.linear1"
        param_map[f"layers.{i}.linear2"] = f"layers.{i}.linear2"

    dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    plot_results_fn = partial(resid_mlp_plot_results_fn, dataloader=dataloader)
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        pretrained_model=target_model,
        param_map=param_map,
        out_dir=out_dir,
        plot_results_fn=plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
