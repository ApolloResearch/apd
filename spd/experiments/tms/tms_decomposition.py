"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import einops
import fire
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSModelConfig, TMSSPDModel, TMSSPDModelConfig
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.plotting import plot_AB_matrices_tms, plot_mask_vals_tms
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    SparseFeatureDataset,
    get_device,
    load_config,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, tms_model_config: TMSModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{tms_model_config.n_features}_"
        run_suffix += f"hid{tms_model_config.n_hidden}"
        run_suffix += f"hid-layers{tms_model_config.n_hidden_layers}"
    return config.wandb_run_name_prefix + run_suffix


def make_plots(
    model: TMSSPDModel,
    target_model: TMSModel,
    step: int,
    out_dir: Path,
    device: str,
    config: Config,
    gates: dict[str, Gate | GateMLP],
    masks: dict[str, Float[Tensor, "batch n_instances m"]],
    batch: Float[Tensor, "batch n_instances n_features"],
    **_,
) -> dict[str, plt.Figure]:
    plots = {}
    plots["masks"], all_perm_indices = plot_mask_vals_tms(
        model=model, target_model=target_model, gates=gates, device=device, input_magnitude=0.75
    )
    plots["AB_matrices"] = plot_AB_matrices_tms(
        model=model, device=device, all_perm_indices=all_perm_indices
    )
    return plots


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    tms_model: TMSModel,
    tms_model_train_config_dict: dict[str, Any],
) -> None:
    torch.save(tms_model.state_dict(), out_dir / "tms.pth")

    with open(out_dir / "tms_train_config.yaml", "w") as f:
        yaml.dump(tms_model_train_config_dict, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "tms.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "tms_train_config.yaml"), base_path=out_dir, policy="now")


def init_spd_model_from_target_model(model: TMSSPDModel, target_model: TMSModel, m: int) -> None:
    assert target_model.config.n_hidden_layers == 0, "Hidden layers not supported for now"
    assert m == target_model.config.n_features, "m must be equal to n_features"
    # We set the A to the identity and B to the target weight matrix
    model.linear1.A.data[:] = einops.repeat(
        torch.eye(m),
        "d_in m -> n_instances d_in m",
        n_instances=target_model.config.n_instances,
    )
    # The B matrix is just the target model's linear layer
    model.linear1.B.data[:] = target_model.linear1.weight.data.clone()
    logger.info("Initialized SPD model from target model")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    device = get_device()

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, tms_model_config=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        tms_model=target_model,
        tms_model_train_config_dict=target_model_train_config_dict,
    )

    tms_spd_model_config = TMSSPDModelConfig(
        **target_model.config.model_dump(mode="json"),
        m=config.m,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
    )
    model = TMSSPDModel(config=tms_spd_model_config)

    if config.init_from_target_model:
        init_spd_model_from_target_model(model=model, target_model=target_model, m=config.m)

    # Manually set the bias for the SPD model from the bias in the pretrained model
    model.b_final.data[:] = target_model.b_final.data.clone()
    model.b_final.requires_grad = False

    param_names = ["linear1", "linear2"]
    if model.hidden_layers is not None:
        for i in range(len(model.hidden_layers)):
            param_names.append(f"hidden_layers.{i}")

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = SparseFeatureDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=make_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
