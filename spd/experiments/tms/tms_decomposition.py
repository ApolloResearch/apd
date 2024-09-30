"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.colors import CenteredNorm
from tqdm import tqdm

from spd.experiments.tms.models import TMSModel, TMSSPDFullRankModel, TMSSPDModel
from spd.experiments.tms.utils import TMSDataset, plot_A_matrix
from spd.log import logger
from spd.run_spd import Config, TMSConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    init_wandb,
    load_config,
    permute_to_identity,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config, task_config: TMSConfig) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        if config.pnorm is not None:
            run_suffix += f"p{config.pnorm:.2e}_"
        if config.lp_sparsity_coeff is not None:
            run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
        if config.topk is not None:
            run_suffix += f"topk{config.topk:.2e}_"
        if config.topk_recon_coeff is not None:
            run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
        if config.topk_l2_coeff is not None:
            run_suffix += f"topkl2_{config.topk_l2_coeff:.2e}_"
        run_suffix += f"lr{config.lr:.2e}_"
        run_suffix += f"bs{config.batch_size}_"
        run_suffix += f"ft{task_config.n_features}_"
        run_suffix += f"hid{task_config.n_hidden}"
    return config.wandb_run_name_prefix + run_suffix


def plot_permuted_A(model: TMSSPDModel, step: int, out_dir: Path, **_) -> plt.Figure:
    permuted_A_T_list: list[torch.Tensor] = []
    for i in range(model.n_instances):
        permuted_matrix = permute_to_identity(model.A[i].T.abs())
        permuted_A_T_list.append(permuted_matrix)
    permuted_A_T = torch.stack(permuted_A_T_list, dim=0)

    fig = plot_A_matrix(permuted_A_T, pos_only=True)
    fig.savefig(out_dir / f"A_{step}.png")
    plt.close(fig)
    tqdm.write(f"Saved A matrix to {out_dir / f'A_{step}.png'}")
    return fig


def plot_subnetwork_params(
    model: TMSSPDFullRankModel | TMSSPDModel, step: int, out_dir: Path, **_
) -> plt.Figure:
    """Plot the subnetwork parameter matrix."""
    if isinstance(model, TMSSPDFullRankModel):
        all_params = model.all_subnetwork_params()
        if len(all_params) > 1:
            logger.warning(
                "Plotting multiple subnetwork params is currently not supported. Plotting the first."
            )
        subnet_params = all_params["W"]
    else:
        assert isinstance(model, TMSSPDModel)
        subnet_params = torch.einsum("ifk,ikh->ikfh", model.A, model.B)

    # subnet_params: [n_instances, k, n_features, n_hidden]
    n_instances, k, dim1, dim2 = subnet_params.shape

    fig, axs = plt.subplots(
        k,
        n_instances,
        figsize=(2 * n_instances, 2 * k),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for i in range(n_instances):
        for j in range(k):
            ax = axs[j, i]  # type: ignore
            param = subnet_params[i, j].detach().cpu().numpy()
            ax.matshow(param, cmap="RdBu", norm=CenteredNorm())
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_ylabel(f"k={j}", rotation=0, ha="right", va="center")
            if j == k - 1:
                ax.set_xlabel(f"Inst {i}", rotation=45, ha="right")

    fig.suptitle(f"Subnetwork Parameters (Step {step})")
    fig.savefig(out_dir / f"subnetwork_params_{step}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved subnetwork params to {out_dir / f'subnetwork_params_{step}.png'}")
    return fig


def make_plots(
    model: TMSSPDFullRankModel | TMSSPDModel, step: int, out_dir: Path, **_
) -> dict[str, plt.Figure]:
    plots = {}
    if isinstance(model, TMSSPDFullRankModel):
        plots["subnetwork_params"] = plot_subnetwork_params(model, step, out_dir)
    else:
        plots["A"] = plot_permuted_A(model, step, out_dir)
        plots["subnetwork_params"] = plot_subnetwork_params(model, step, out_dir)
    return plots


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)
    task_config = config.task_config
    assert isinstance(task_config, TMSConfig)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)
    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config, task_config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.full_rank:
        model = TMSSPDFullRankModel(
            n_instances=task_config.n_instances,
            n_features=task_config.n_features,
            n_hidden=task_config.n_hidden,
            k=task_config.k,
            train_bias=task_config.train_bias,
            bias_val=task_config.bias_val,
            device=device,
        )
    else:
        model = TMSSPDModel(
            n_instances=task_config.n_instances,
            n_features=task_config.n_features,
            n_hidden=task_config.n_hidden,
            k=task_config.k,
            train_bias=task_config.train_bias,
            bias_val=task_config.bias_val,
            device=device,
        )

    pretrained_model = None
    if task_config.pretrained_model_path:
        pretrained_model = TMSModel(
            n_instances=task_config.n_instances,
            n_features=task_config.n_features,
            n_hidden=task_config.n_hidden,
            device=device,
        )
        pretrained_model.load_state_dict(
            torch.load(task_config.pretrained_model_path, map_location=device)
        )
        pretrained_model.eval()

    param_map = None
    if task_config.pretrained_model_path:
        # Map from pretrained model's `all_decomposable_params` to the SPD models'
        # `all_subnetwork_params_summed`.
        param_map = {"W": "W", "W_T": "W_T"}

    dataset = TMSDataset(
        n_instances=task_config.n_instances,
        n_features=task_config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    optimize(
        model=model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        param_map=param_map,
        plot_results_fn=make_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
