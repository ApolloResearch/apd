"""TMS model, adapted from
https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from matplotlib import collections as mc
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from tqdm import tqdm, trange

from spd.experiments.tms.models import TMSModel, TMSModelConfig
from spd.log import logger
from spd.utils import DatasetGeneratedDataLoader, SparseFeatureDataset, set_seed

wandb.require("core")


class TMSTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    tms_model_config: TMSModelConfig
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr: float
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]
    fixed_identity_hidden_layers: bool = False
    fixed_random_hidden_layers: bool = False
    synced_inputs: list[list[int]] | None = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.fixed_identity_hidden_layers and self.fixed_random_hidden_layers:
            raise ValueError(
                "Cannot set both fixed_identity_hidden_layers and fixed_random_hidden_layers to True"
            )
        if self.synced_inputs is not None:
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: TMSModel,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    log_wandb: bool,
    importance: float = 1.0,
    steps: int = 5_000,
    print_freq: int = 100,
    lr: float = 5e-3,
    lr_schedule: Callable[[int, int], float] = linear_lr,
) -> None:
    hooks = []

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch, labels = next(data_iter)
            out = model(batch)
            error = importance * (labels.abs() - out) ** 2
            loss = error.mean()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                t.set_postfix(
                    loss=loss.item(),
                    lr=step_lr,
                )
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)


def plot_intro_diagram(model: TMSModel, filepath: Path) -> None:
    """2D polygon plot of the TMS model.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    WA = model.linear1.weight.detach()
    color = plt.cm.viridis(np.array([0.0]))  # type: ignore
    plt.rcParams["figure.dpi"] = 200
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    W = WA.cpu().detach().numpy()
    ax.scatter(W[:, 0], W[:, 1], c=color)
    ax.set_aspect("equal")
    ax.add_collection(
        mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=[color])  # type: ignore
    )

    z = 1.5
    ax.set_facecolor("#FCFBF8")
    ax.set_xlim((-z, z))
    ax.set_ylim((-z, z))
    ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position("center")
    plt.savefig(filepath)


def plot_cosine_similarity_distribution(
    model: TMSModel,
    filepath: Path,
) -> None:
    """Create scatter plot of cosine similarities between feature vectors.

    Args:
        model: The trained TMS model
        filepath: Where to save the plot
    """
    # Calculate cosine similarities
    rows = model.linear1.weight.detach()
    rows /= rows.norm(dim=-1, keepdim=True)
    cosine_sims = einops.einsum(rows, rows, "f1 h, f2 h -> f1 f2")
    mask = ~torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool)
    masked_sims = cosine_sims[mask]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sims = masked_sims.cpu().numpy()
    ax.scatter(sims, np.zeros_like(sims), alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cosine Similarity")
    ax.set_yticks([])  # Hide y-axis ticks

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def get_model_and_dataloader(
    config: TMSTrainConfig, device: str
) -> tuple[TMSModel, DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    model = TMSModel(config=config.tms_model_config)
    model.to(device)
    if (
        config.fixed_identity_hidden_layers or config.fixed_random_hidden_layers
    ) and model.hidden_layers is not None:
        for i in range(model.config.n_hidden_layers):
            if config.fixed_identity_hidden_layers:
                model.hidden_layers[i].weight.data[:, :, :] = torch.eye(
                    model.config.n_hidden, device=device
                )
            elif config.fixed_random_hidden_layers:
                model.hidden_layers[i].weight.data[:, :, :] = torch.randn_like(
                    model.hidden_layers[i].weight
                )
            model.hidden_layers[i].weight.requires_grad = False

    dataset = SparseFeatureDataset(
        n_features=config.tms_model_config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=config.synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    return model, dataloader


def run_train(config: TMSTrainConfig, device: str) -> None:
    model, dataloader = get_model_and_dataloader(config, device)

    model_cfg = config.tms_model_config
    run_name = (
        f"tms_n-features{model_cfg.n_features}_n-hidden{model_cfg.n_hidden}_"
        f"n-hidden-layers{model_cfg.n_hidden_layers}_"
        f"feat_prob{config.feature_probability}_seed{config.seed}"
    )
    if config.fixed_identity_hidden_layers:
        run_name += "_fixed-identity"
    elif config.fixed_random_hidden_layers:
        run_name += "_fixed-random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.wandb_project:
        wandb.init(project=config.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "tms_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
    )

    model_path = out_dir / "tms.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    if model_cfg.n_hidden == 2:
        plot_intro_diagram(model, filepath=out_dir / "polygon.png")
        logger.info(f"Saved diagram to {out_dir / 'polygon.png'}")

    plot_cosine_similarity_distribution(
        model, filepath=out_dir / "cosine_similarity_distribution.png"
    )
    logger.info(
        f"Saved cosine similarity distribution to {out_dir / 'cosine_similarity_distribution.png'}"
    )
    logger.info(f"1/sqrt(n_hidden): {1 / np.sqrt(model_cfg.n_hidden)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TMS 5-2
    # config = TMSTrainConfig(
    #     wandb_project="spd-train-tms",
    #     tms_model_config=TMSModelConfig(
    #         n_features=5,
    #         n_hidden=2,
    #         n_hidden_layers=0,
    #         device=device,
    #     ),
    #     feature_probability=0.05,
    #     batch_size=1024,
    #     steps=5000,
    #     seed=0,
    #     lr=5e-3,
    #     data_generation_type="at_least_zero_active",
    #     fixed_identity_hidden_layers=False,
    #     fixed_random_hidden_layers=False,
    # )
    # TMS 40-10
    config = TMSTrainConfig(
        wandb_project="spd-train-tms",
        tms_model_config=TMSModelConfig(
            n_features=40,
            n_hidden=10,
            n_hidden_layers=1,
            tied_weights=True,
            device=device,
        ),
        feature_probability=0.05,
        # feature_probability=0.02, # synced inputs
        batch_size=2048,
        steps=2000,
        seed=0,
        lr=1e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
        # synced_inputs=[[5, 6], [0, 2, 3]],
    )
    set_seed(config.seed)

    run_train(config, device)
