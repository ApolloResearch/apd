from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from tqdm import tqdm

from spd.log import logger
from spd.data import VisionDatasetConfig, create_image_data_loader


import datasets
import pydantic
import torch
import torch.nn.functional as F
import yaml
import wandb
from datasets import load_dataset
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    default_data_collator,
)

class ViTTrainConfig(pydantic.BaseModel):
    # Model / data
    pretrained_model_path: str = "google/vit-base-patch16-224-in21k"
    dataset_path: str = "zh-plus/tiny-imagenet"
    grouped_labels: list[list[int]] = field(default_factory=list)
    # Training hyper‑parameters
    batch_size: int = 4096
    lr: float = 1e-3
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    max_train_steps: int = 40_000
    n_eval_steps: int = 100
    weight_decay: float = 0.0
    seed: int = 0
    num_workers: int = 8
    # Logging / output
    wandb_project: str | None = None
    output_dir: str = "runs/vit_finetune"
    train_split: str = "train"
    eval_split: str = "valid"
    test_split: str = "valid"

    @pydantic.field_validator("grouped_labels")
    def label_validator(cls, value: Sequence[Sequence[int]]):  # noqa: D401
        # Validate grouped_labels uniqueness & no duplicates
        seen: set[int] = set()
        for grp in value:
            if len(set(grp)) != len(grp):
                raise ValueError("Duplicate label inside a grouped list")
            overlap = seen & set(grp)
            if overlap:
                raise ValueError(f"Labels {overlap} appear in multiple groups")
            seen.update(grp)
        return value

    @property
    def num_labels(self) -> int:
        # groups + 1 (not-class)
        return len(self.grouped_labels) + 1

    @property
    def label_mapping(self):
        id2label = {gi: f"group_{gi}" for gi in range(len(self.grouped_labels))}
        id2label[self.num_labels - 1] = "not"
        return id2label, {v: k for k, v in id2label.items()}


# ────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optim: torch.optim.Optimizer, cfg: ViTTrainConfig):
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optim, cfg.max_train_steps)
    if cfg.lr_schedule == "linear":
        return LambdaLR(optim, lambda step: 1 - step / cfg.max_train_steps)
    return LambdaLR(optim, lambda _: 1.0)  # constant


# ────────────────────────────────────────────────────────────────────────
# Running training
# ────────────────────────────────────────────────────────────────────────


def accuracy(logits: torch.Tensor, labels: torch.Tensor):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]], device="cuda"):
    model.eval()
    accs = []
    for batch in tqdm(loader, desc="Evaluating", leave=False, ncols=80):
        labels = batch.pop("labels").to(device)
        batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}
        accs.append(accuracy(model(**batch).logits, labels))
    model.train()
    return sum(accs) / len(accs)


def run_train(cfg: ViTTrainConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = (
        f"vit_finetune_{cfg.pretrained_model_path.split('/')[-1]}_"
        f"{cfg.dataset_path.split('/')[-1]}_"
        f"label_groups_{'+'.join(','.join(map(str, grp)) for grp in cfg.grouped_labels)}"
    )

    # Timestamped output dir
    out_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.__dict__, f)

    if cfg.wandb_project:
        wandb.init(project=cfg.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "vision_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, indent=2)
    if cfg.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    id2label, label2id = cfg.label_mapping
    processor = AutoImageProcessor.from_pretrained(cfg.pretrained_model_path)
    model = ViTForImageClassification.from_pretrained(
        cfg.pretrained_model_path,
        num_labels=cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # in case of different num_labels
    )
    model = model.to(device) # type: ignore

    train_data_config = VisionDatasetConfig(
        name=cfg.dataset_path,
        split=cfg.train_split,
        hf_image_processor_path=cfg.pretrained_model_path,
    )
    train_loader, _ = create_image_data_loader(
        ds_cfg=train_data_config,
        batch_size=cfg.batch_size,
        grouped_labels=cfg.grouped_labels,
        global_seed=cfg.seed,
        ddp_rank=0,
        ddp_world_size=1,
        trust_remote_code=True,  # for ViT models
        shuffle=True,
        balanced=True,
    )

    # Create validation and test datasets
    val_data_config = VisionDatasetConfig(
        name=cfg.dataset_path,
        split=cfg.eval_split,
        hf_image_processor_path=cfg.pretrained_model_path,
    )
    val_loader, _ = create_image_data_loader(
        ds_cfg=val_data_config,
        batch_size=cfg.batch_size,
        grouped_labels=cfg.grouped_labels,
        global_seed=cfg.seed,
        ddp_rank=0,
        ddp_world_size=1,
        trust_remote_code=True,  # for ViT models
        balanced=True,
    )
    test_data_config = VisionDatasetConfig(
        name=cfg.dataset_path,
        split=cfg.test_split,
        hf_image_processor_path=cfg.pretrained_model_path,
    )
    test_loader, _ = create_image_data_loader(
        ds_cfg=test_data_config,
        batch_size=cfg.batch_size,
        grouped_labels=cfg.grouped_labels,
        global_seed=cfg.seed,
        ddp_rank=0,
        ddp_world_size=1,
        trust_remote_code=True,  # for ViT models
        balanced=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_scheduler(opt, cfg)

    model_path = out_dir / "best_model"

    pbar = trange(cfg.max_train_steps, ncols=0)
    train_iter = iter(train_loader)

    best_val_acc = 0.0
    model.save_pretrained(model_path)  # Save initial model state

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        labels = batch.pop("labels").to(device)
        batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}

        loss = model(**batch, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if (step + 1) % cfg.n_eval_steps == 0:
            val_acc = evaluate(model, val_loader, device)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{val_acc:.4f}")
            if cfg.wandb_project:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/acc": val_acc,
                        "lr": sched.get_last_lr()[0],
                    },
                    step=step + 1,
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(model_path)
        if step + 1 >= cfg.max_train_steps:
            break

    # Load best model from model_path
    model.from_pretrained(model_path)
    if cfg.wandb_project:
        artifact = wandb.Artifact(
            name=run_name,
            type="model",
            description="Best model from training",
        )
        artifact.add_dir(str(model_path), name="model")
        wandb.log_artifact(artifact)
    logger.info(f"Saved best model to {model_path}")

    # Run test
    test_acc = evaluate(model, test_loader, device)
    logger.info(f"Test accuracy: {test_acc:.4f}")

# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────


def load_cfg(path: str | Path) -> ViTTrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ViTTrainConfig(**data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML/JSON config file")
    args = ap.parse_args()
    run_train(load_cfg(args.config))
