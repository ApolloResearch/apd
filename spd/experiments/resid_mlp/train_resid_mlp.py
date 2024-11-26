"""Trains a residual linear model on one-hot input vectors."""

import json
from pathlib import Path
from typing import Literal

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.resid_mlp_dataset import (
    ResidualMLPDataset,
)
from spd.run_spd import get_lr_schedule_fn
from spd.utils import DatasetGeneratedDataLoader, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    label_fn_seed: int = 0
    label_type: Literal["act_plus_resid", "abs"] = "act_plus_resid"
    use_trivial_label_coeffs: bool = False
    n_instances: PositiveInt
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"]
    in_bias: bool = False
    out_bias: bool = False
    feature_probability: PositiveFloat
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"


def train(
    config: Config,
    model: ResidualMLPModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_resid"]]
    ],
    device: str,
    out_dir: Path | None = None,
) -> float | None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    final_loss = None
    for step, (batch, labels) in enumerate(dataloader):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        out, _, _ = model(batch)
        loss = F.mse_loss(out, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={final_loss}, lr={current_lr}")

    if out_dir is not None:
        model_path = out_dir / "target_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        config_path = out_dir / "target_model_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        print(f"Saved config to {config_path}")

        # Save the coefficients used to generate the labels if label_type is act_plus_resid
        assert isinstance(dataloader.dataset, ResidualMLPDataset)
        assert dataloader.dataset.label_coeffs is not None
        label_coeffs = dataloader.dataset.label_coeffs.tolist()
        label_coeffs_path = out_dir / "label_coeffs.json"
        with open(label_coeffs_path, "w") as f:
            json.dump(label_coeffs, f)
        print(f"Saved label coefficients to {label_coeffs_path}")

    print(f"Final loss: {final_loss}")
    return final_loss


if __name__ == "__main__":
    device = "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        n_instances=10,
        n_features=5,
        d_embed=5,
        d_mlp=5,
        n_layers=1,
        act_fn_name="relu",
        label_type="abs",
        use_trivial_label_coeffs=True,
        in_bias=False,
        out_bias=False,
        feature_probability=0.2,
        batch_size=256,
        steps=10_000,
        print_freq=100,
        lr=1e-2,
        lr_schedule="cosine",
    )

    set_seed(config.seed)
    run_name = (
        f"resid_mlp_identity_{config.label_type}_n-instances{config.n_instances}_"
        f"n-features{config.n_features}_d-resid{config.d_embed}_"
        f"d-mlp{config.d_mlp}_n-layers{config.n_layers}_seed{config.seed}"
    )
    out_dir = Path(__file__).parent / "out" / run_name

    model = ResidualMLPModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        d_embed=config.d_embed,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
        act_fn_name=config.act_fn_name,
        in_bias=config.in_bias,
        out_bias=config.out_bias,
    ).to(device)

    # TODO: We likely want to remove this identity embedding. Leaving for now to test whether
    # resid_mlp gets the same results as resid_linear.
    # Make W_E the identity matrix
    assert model.W_E.shape == (config.n_instances, config.n_features, config.d_embed)
    eye = torch.eye(config.d_embed, device=device)
    model.W_E.data[:, :, :] = einops.repeat(
        eye, "d_emb1 d_emb2 -> n_instances d_emb1 d_emb2", n_instances=config.n_instances
    )

    # Don't train the Embedding matrix
    model.W_E.requires_grad = False
    trainable_params = [p for n, p in model.named_parameters() if "W_E" not in n]

    label_coeffs = None
    if config.use_trivial_label_coeffs:
        label_coeffs = torch.ones(config.n_instances, config.n_features, device=device)

    dataset = ResidualMLPDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=config.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        label_coeffs=label_coeffs,
        data_generation_type="at_least_zero_active",
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    train(
        config=config,
        model=model,
        trainable_params=trainable_params,
        dataloader=dataloader,
        device=device,
        out_dir=out_dir,
    )
