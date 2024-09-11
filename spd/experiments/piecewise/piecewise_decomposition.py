"""Linear decomposition script."""

import json
from pathlib import Path

import fire
import torch
import wandb
from jaxtyping import Float
from torch import Tensor

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.plotting import (
    plot_components,
    plot_components_fullrank,
    plot_model_functions,
)
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.log import logger
from spd.run_spd import Config, PiecewiseConfig, calc_recon_mse, optimize
from spd.utils import (
    BatchedDataLoader,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def piecewise_plot_results_fn(
    model: PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer,
    target_model: PiecewiseFunctionTransformer | None,
    step: int,
    out_dir: Path | None,
    device: str,
    topk: float,
    batch_topk: bool,
    slow_images: bool,
):
    # Plot functions
    fig_dict_1 = plot_model_functions(
        spd_model=model,
        target_model=target_model,
        topk=topk,
        batch_topk=batch_topk,
        full_rank=isinstance(model, PiecewiseFunctionSPDFullRankTransformer),
        device=device,
        print_info=False,
    )
    # Plot components
    if isinstance(model, PiecewiseFunctionSPDFullRankTransformer):
        fig_dict_2 = plot_components_fullrank(
            model=model, step=step, out_dir=out_dir, slow_images=slow_images
        )
    else:
        fig_dict_2 = plot_components(
            model=model, step=step, out_dir=out_dir, device=device, slow_images=slow_images
        )
    return {**fig_dict_1, **fig_dict_2}


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, PiecewiseConfig)
        run_suffix += f"seed{config.seed}_"
        if config.task_config.target_seed is not None:
            run_suffix += f"target-seed{config.task_config.target_seed}_"
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
        if config.task_config.handcoded_AB:
            run_suffix += "hAB_"
        run_suffix += f"lr{config.lr:.2e}_"
        run_suffix += f"bs{config.batch_size}"
        run_suffix += f"lay{config.task_config.n_layers}"

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
    target_seed = (
        config.task_config.target_seed
        if config.task_config.target_seed is not None
        else config.seed
    )
    # Set seed for function generation and handcoded parameter setting
    set_seed(target_seed)
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

    set_seed(config.seed)
    if config.full_rank:
        piecewise_model_spd = PiecewiseFunctionSPDFullRankTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            input_biases=input_biases,
        )
        if config.task_config.handcoded_AB:
            logger.info("Setting handcoded A and B matrices (!)")
            non_full_rank_spd_model = PiecewiseFunctionSPDTransformer(
                n_inputs=piecewise_model.n_inputs,
                d_mlp=piecewise_model.d_mlp,
                n_layers=piecewise_model.n_layers,
                k=config.task_config.k,
                input_biases=input_biases,
            )
            non_full_rank_spd_model.set_handcoded_AB(piecewise_model)
            piecewise_model_spd.set_handcoded_AB(non_full_rank_spd_model)
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
        plot_results_fn=piecewise_plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
