"""Trains a residual linear model on one-hot input vectors."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn
from tqdm import tqdm

from spd.experiments.resid_mlp.models import ResidualMLPConfig, ResidualMLPModel
from spd.experiments.resid_mlp.resid_mlp_dataset import (
    ResidualMLPDataset,
)
from spd.log import logger
from spd.run_spd import get_lr_schedule_fn
from spd.utils import DatasetGeneratedDataLoader, compute_feature_importances, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


class ResidMLPTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    seed: int = 0
    resid_mlp_config: ResidualMLPConfig
    label_fn_seed: int = 0
    label_type: Literal["act_plus_resid", "abs"] = "act_plus_resid"
    loss_type: Literal["readoff", "resid"] = "readoff"
    use_trivial_label_coeffs: bool = False
    feature_probability: PositiveFloat
    synced_inputs: list[list[int]] | None = None
    importance_val: float | None = None
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    fixed_random_embedding: bool = False
    fixed_identity_embedding: bool = False
    n_batches_final_losses: PositiveInt = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert not (
            self.fixed_random_embedding and self.fixed_identity_embedding
        ), "Can't have both fixed_random_embedding and fixed_identity_embedding"
        if self.fixed_identity_embedding:
            assert (
                self.resid_mlp_config.n_features == self.resid_mlp_config.d_embed
            ), "n_features must equal d_embed if we are using an identity embedding matrix"
        if self.synced_inputs is not None:
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


def loss_function(
    out: Float[Tensor, "batch n_instances n_features"] | Float[Tensor, "batch n_instances d_embed"],
    labels: Float[Tensor, "batch n_instances n_features"],
    feature_importances: Float[Tensor, "batch n_instances n_features"],
    post_acts: dict[str, Float[Tensor, "batch n_instances d_embed"]],
    model: ResidualMLPModel,
    config: ResidMLPTrainConfig,
) -> Float[Tensor, "batch n_instances d_embed"] | Float[Tensor, "batch n_instances d_embed"]:
    if config.loss_type == "readoff":
        loss = ((out - labels) ** 2) * feature_importances
    elif config.loss_type == "resid":
        assert torch.allclose(
            feature_importances, torch.ones_like(feature_importances)
        ), "feature_importances incompatible with loss_type resid"
        resid_out: Float[Tensor, "batch n_instances d_embed"] = out
        resid_labels: Float[Tensor, "batch n_instances d_embed"] = einops.einsum(
            labels,
            model.W_E,
            "batch n_instances n_features, n_instances n_features d_embed "
            "-> batch n_instances d_embed",
        )
        loss = (resid_out - resid_labels) ** 2
    else:
        raise ValueError(f"Invalid loss_type: {config.loss_type}")
    return loss


def train(
    config: ResidMLPTrainConfig,
    model: ResidualMLPModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances d_embed"],
        ]
    ],
    feature_importances: Float[Tensor, "batch_size n_instances n_features"],
    device: str,
    out_dir: Path,
    run_name: str,
) -> Float[Tensor, " n_instances"]:
    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, name=run_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = out_dir / "resid_mlp_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")

    # Save the coefficients used to generate the labels
    assert isinstance(dataloader.dataset, ResidualMLPDataset)
    assert dataloader.dataset.label_coeffs is not None
    label_coeffs = dataloader.dataset.label_coeffs.tolist()
    label_coeffs_path = out_dir / "label_coeffs.json"
    with open(label_coeffs_path, "w") as f:
        json.dump(label_coeffs, f)
    logger.info(f"Saved label coefficients to {label_coeffs_path}")
    if config.wandb_project:
        wandb.save(str(label_coeffs_path), base_path=out_dir, policy="now")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    current_losses = torch.tensor([])
    pbar = tqdm(range(config.steps), total=config.steps)
    for step, (batch, labels) in zip(pbar, dataloader, strict=False):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch: Float[Tensor, "batch n_instances n_features"] = batch.to(device)
        labels: Float[Tensor, "batch n_instances n_features"] = labels.to(device)
        out, pre_acts, post_acts = model(batch, return_residual=config.loss_type == "resid")
        loss: (
            Float[Tensor, "batch n_instances n_features"]
            | Float[Tensor, "batch n_instances d_embed"]
        ) = loss_function(out, labels, feature_importances, post_acts, model, config)
        loss = loss.mean(dim=(0, 2))
        current_losses = loss.detach()
        loss = loss.mean(dim=0)
        loss.backward()
        optimizer.step()
        if step % config.print_freq == 0:
            pbar.set_description(f"loss={current_losses.mean():.2e}, lr={current_lr:.2e}")
            if config.wandb_project:
                wandb.log({"loss": current_losses.mean(), "lr": current_lr}, step=step)

    model_path = out_dir / "resid_mlp.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    print(f"Saved model to {model_path}")

    # Calculate final losses by averaging many batches
    final_losses = []
    for _ in range(config.n_batches_final_losses):
        batch, labels = next(iter(dataloader))
        batch = batch.to(device)
        labels = labels.to(device)
        out, _, post_acts = model(batch, return_residual=config.loss_type == "resid")
        loss = loss_function(out, labels, feature_importances, post_acts, model, config)
        loss = loss.mean(dim=(0, 2))
        final_losses.append(loss)
    final_losses = torch.stack(final_losses).mean(dim=0).cpu().detach()
    print(f"Final losses: {final_losses.numpy()}")
    return final_losses


def run_train(config: ResidMLPTrainConfig, device: str) -> Float[Tensor, " n_instances"]:
    model_cfg = config.resid_mlp_config
    run_name = (
        f"resid_mlp_identity_{config.label_type}_n-instances{model_cfg.n_instances}_"
        f"n-features{model_cfg.n_features}_d-resid{model_cfg.d_embed}_"
        f"d-mlp{model_cfg.d_mlp}_n-layers{model_cfg.n_layers}_seed{config.seed}"
        f"_p{config.feature_probability}_random_embedding_{config.fixed_random_embedding}_"
        f"identity_embedding_{config.fixed_identity_embedding}_bias_{model_cfg.in_bias}_"
        f"{model_cfg.out_bias}_loss{config.loss_type}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"

    model = ResidualMLPModel(config=model_cfg).to(device)

    if config.fixed_random_embedding or config.fixed_identity_embedding:
        # Don't train the embedding matrices
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False
        if config.fixed_random_embedding:
            # Init with randn values and make unit norm
            model.W_E.data[:, :, :] = torch.randn(
                model_cfg.n_instances, model_cfg.n_features, model_cfg.d_embed, device=device
            )
            model.W_E.data /= model.W_E.data.norm(dim=-1, keepdim=True)
            # Set W_U to W_E^T
            model.W_U.data = model.W_E.data.transpose(-2, -1)
            assert torch.allclose(model.W_U.data, model.W_E.data.transpose(-2, -1))
        elif config.fixed_identity_embedding:
            assert (
                model_cfg.n_features == model_cfg.d_embed
            ), "n_features must equal d_embed for W_E=id"
            # Make W_E the identity matrix
            model.W_E.data[:, :, :] = einops.repeat(
                torch.eye(model_cfg.d_embed, device=device),
                "d_features d_embed -> n_instances d_features d_embed",
                n_instances=model_cfg.n_instances,
            )

    label_coeffs = None
    if config.use_trivial_label_coeffs:
        label_coeffs = torch.ones(model_cfg.n_instances, model_cfg.n_features, device=device)

    dataset = ResidualMLPDataset(
        n_instances=model_cfg.n_instances,
        n_features=model_cfg.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=model_cfg.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        label_coeffs=label_coeffs,
        data_generation_type=config.data_generation_type,
        synced_inputs=config.synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    feature_importances = compute_feature_importances(
        batch_size=config.batch_size,
        n_instances=model_cfg.n_instances,
        n_features=model_cfg.n_features,
        importance_val=config.importance_val,
        device=device,
    )

    final_losses = train(
        config=config,
        model=model,
        trainable_params=[p for p in model.parameters() if p.requires_grad],
        dataloader=dataloader,
        feature_importances=feature_importances,
        device=device,
        out_dir=out_dir,
        run_name=run_name,
    )
    return final_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ResidMLPTrainConfig(
        wandb_project="spd-train-resid-mlp",
        seed=0,
        resid_mlp_config=ResidualMLPConfig(
            n_instances=1,
            n_features=100,
            d_embed=1000,
            d_mlp=50,
            n_layers=1,
            act_fn_name="relu",
            apply_output_act_fn=False,
            in_bias=False,
            out_bias=False,
        ),
        label_fn_seed=0,
        label_type="act_plus_resid",
        loss_type="readoff",
        use_trivial_label_coeffs=True,
        feature_probability=0.01,
        # synced_inputs=[[0, 1], [2, 3]],
        importance_val=1,
        data_generation_type="at_least_zero_active",
        batch_size=2048,
        steps=10000,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        fixed_random_embedding=True,
        fixed_identity_embedding=False,
        n_batches_final_losses=10,
    )

    set_seed(config.seed)

    run_train(config, device)
