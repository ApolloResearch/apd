"""
Vizualises the components of the model.
"""

import math

import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import LMTaskConfig
from spd.experiments.lm.models import EmbeddingComponent, LinearComponentWithBias, SSModel
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.run_spd import calc_component_acts, calc_masks
from spd.types import ModelPath


def component_activation_statistics(
    model: SSModel,
    dataloader: DataLoader[Float[Tensor, "batch pos"]],
    n_steps: int,
    device: str,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " m"]]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponentWithBias | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.m, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = next(data_iter)["input_ids"].to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.linear_component.A for module_name, v in components.items()}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

        masks, relud_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=None,
            detach_inputs=False,
        )
        for module_name, mask in masks.items():
            assert mask.ndim == 3  # (batch_size, pos, m)
            n_tokens[module_name] += mask.shape[0] * mask.shape[1]
            # Count the number of components that are active at all
            active_components = mask > 0
            total_n_active_components[module_name] += int(active_components.sum().item())
            component_activation_counts[module_name] += active_components.sum(dim=(0, 1))

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " m"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts


def plot_mean_component_activation_counts(
    mean_component_activation_counts: dict[str, Float[Tensor, " m"]],
) -> plt.Figure:
    """Plots the mean activation counts for each component module in a grid."""
    n_modules = len(mean_component_activation_counts)
    max_cols = 6
    n_cols = min(n_modules, max_cols)
    # Calculate the number of rows needed, rounding up
    n_rows = math.ceil(n_modules / n_cols)

    # Create a figure with the calculated number of rows and columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    # Ensure axs is always a 2D array for consistent indexing, even if n_modules is 1
    axs = axs.flatten()  # Flatten the axes array for easy iteration

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, counts) in enumerate(mean_component_activation_counts.items()):
        ax = axs[i]
        ax.hist(counts.detach().cpu().numpy(), bins=100)
        ax.set_yscale("log")
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Mean Activation Count")
        ax.set_ylabel("Frequency")

    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_modules, n_rows * n_cols):
        axs[i].axis("off")

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()

    return fig


def main(path: ModelPath) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ss_model, config, checkpoint_path = SSModel.from_pretrained(path)
    ss_model.to(device)

    out_dir = checkpoint_path

    assert isinstance(config.task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}",
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )

    dataloader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )
    # print(ss_model)
    print(config)

    mean_n_active_components_per_token, mean_component_activation_counts = (
        component_activation_statistics(
            model=ss_model,
            dataloader=dataloader,
            n_steps=100,
            device=device,
        )
    )
    logger.info(f"n_components: {ss_model.m}")
    logger.info(f"mean_n_active_components_per_token: {mean_n_active_components_per_token}")
    logger.info(f"mean_component_activation_counts: {mean_component_activation_counts}")
    fig = plot_mean_component_activation_counts(
        mean_component_activation_counts=mean_component_activation_counts,
    )
    # Save the entire figure once
    save_path = out_dir / "modules_mean_component_activation_counts.png"
    fig.savefig(save_path)
    logger.info(f"Saved combined plot to {str(save_path)}")


if __name__ == "__main__":
    path = "wandb:spd-lm/runs/151bsctx"
    main(path)
