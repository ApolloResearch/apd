"""Trains a residual linear model on one-hot input vectors."""

import json
from collections.abc import Callable
from pathlib import Path

import einops
import torch
import wandb
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import (
    ResidualLinearDataset,
    create_label_function,
)
from spd.utils import DatasetGeneratedDataLoader, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    label_fn_seed: int = 0
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    feature_probability: PositiveFloat
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat


def train(
    config: Config,
    model: ResidualLinearModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[Float[Tensor, "batch n_features"]],
    label_fn: Callable[[Float[Tensor, "batch d_embed"]], Float[Tensor, "batch n_functions"]],
    device: str,
    out_dir: Path | None = None,
) -> float | None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.0)

    final_loss = None
    for step, (batch, _) in enumerate(dataloader):
        if step >= config.steps:
            break
        optimizer.zero_grad()
        batch = batch.to(device)
        out, _, _ = model(batch)

        raw_labels: Float[Tensor, "batch n_features"] = label_fn(batch)
        # We want our labels to be downprojected to the residual stream space because there is no
        # unembedding matrix.
        labels = einops.einsum(
            model.W_E, raw_labels, "n_features d_embed, batch n_features -> batch d_embed"
        )

        loss = F.mse_loss(out, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={final_loss}")

    if out_dir is not None:
        model_path = out_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        config_path = out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=4)
        print(f"Saved config to {config_path}")

    print(f"Final loss: {final_loss}")
    print("\nIn residual space (after embedding)")
    print(f"labels[20:30]:\n{labels[20:30]}")  # type: ignore
    print(f"out[20:30]:\n{out[20:30]}")  # type: ignore
    return final_loss


if __name__ == "__main__":
    device = "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        n_features=5,
        d_embed=5,
        d_mlp=5,
        n_layers=1,
        feature_probability=0.2,
        batch_size=256,
        steps=5_000,
        print_freq=100,
        lr=1e-2,
    )

    set_seed(config.seed)
    run_name = (
        f"resid_linear_n-features{config.n_features}_d-resid{config.d_embed}_"
        f"d-mlp{config.d_mlp}_n-layers{config.n_layers}_seed{config.seed}"
    )
    out_dir = Path(__file__).parent / "out" / run_name

    model = ResidualLinearModel(
        n_features=config.n_features,
        d_embed=config.d_embed,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
    ).to(device)

    # Don't train the Embedding matrix
    model.W_E.requires_grad = False
    trainable_params = [p for n, p in model.named_parameters() if "W_E" not in n]

    dataset = ResidualLinearDataset(
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
    )
    label_fn = create_label_function(config.d_embed, seed=config.label_fn_seed)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    train(
        config=config,
        model=model,
        trainable_params=trainable_params,
        dataloader=dataloader,
        label_fn=label_fn,
        device=device,
        out_dir=out_dir,
    )
