import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import einops
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST, KMNIST, MNIST, FashionMNIST, VisionDataset
from tqdm import tqdm

from spd.experiments.multitask.single import (
    E10MNIST,
    MultiMNISTDataset,
    SingleMNISTModel,
    transform,
)


def copy_weights(srcs: list[nn.Linear], dst: nn.Linear) -> None:
    """Copy the weights of each source linear into block-diagonal sections of the destination linear."""
    n_sources = len(srcs)
    in_sizes = [src.in_features for src in srcs]
    out_sizes = [src.out_features for src in srcs]
    assert sum(in_sizes) == dst.in_features, f"{(in_sizes)} != {dst.in_features}"
    assert sum(out_sizes) == dst.out_features, f"{(out_sizes)} != {dst.out_features}"
    in_idx = 0
    out_idx = 0
    for i in range(n_sources):
        src = srcs[i]
        print(
            f"Copying from {src.weight.data.shape} into {dst.weight.data[out_idx : out_idx + src.out_features, in_idx : in_idx + src.in_features].shape} of {dst.weight.data.shape}"
        )
        dst.weight.data[out_idx : out_idx + src.out_features, in_idx : in_idx + src.in_features] = (
            src.weight.data
        )
        dst.bias.data[out_idx : out_idx + src.out_features] = src.bias.data
        in_idx += src.in_features
        out_idx += src.out_features
    return dst


class CombinedMNISTModel(nn.Module):
    def __init__(self, models: list[SingleMNISTModel]):
        super().__init__()
        self.n_models = len(models)
        self.input_size = sum(model.input_size for model in models)
        self.hidden_size = sum(model.hidden_size for model in models)
        self.num_classes = sum(model.num_classes for model in models)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        copy_weights([m.fc1 for m in models], self.fc1)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        copy_weights([m.fc2 for m in models], self.fc2)
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes)
        copy_weights([m.fc3 for m in models], self.fc3)
        self.act = nn.ReLU()

    def forward(
        self, x: Float[torch.Tensor, "batch input_size"]
    ) -> Float[torch.Tensor, "batch num_classes"]:
        is_active = x.sum(dim=-1) > 0
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return einops.einsum(out, is_active, "batch n, batch -> batch n")


def main():
    models = []
    for name, fname in [
        ("mnist", "MNIST.pth"),
        ("kmnist", "KMNIST.pth"),
        ("fashion", "FashionMNIST.pth"),
        ("e10mnist", "E10MNIST.pth"),
    ]:
        print(f"Loading {name} model")
        model = SingleMNISTModel(input_size=28**2, hidden_size=512, num_classes=10)
        model.load_state_dict(torch.load(f"models/{name}/{fname}", weights_only=True))
        models.append(model)
    combined_model = CombinedMNISTModel(models)
    print(combined_model)

    data_dir = "/data/apollo/torch_datasets/"
    # Load the dataset and splits
    kwargs = {"root": data_dir, "download": True, "transform": transform, "train": False}
    test_datasets = [
        MNIST(**kwargs),
        KMNIST(**kwargs),
        FashionMNIST(**kwargs),
        E10MNIST(**kwargs),
    ]
    test_dataset = MultiMNISTDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    for i, (x, y) in enumerate(test_loader):
        print(
            f"images {i}",
            x[0][:784].abs().sum(),
            x[0][784 : 2 * 784].abs().sum(),
            x[0][2 * 784 : 3 * 784].abs().sum(),
            x[0][3 * 784 : 4 * 784].abs().sum(),
        )
        print(f"labels {i}", y[0][0], y[0][10], y[0][20], y[0][30])
        if i > 10:
            break


if __name__ == "__main__":
    fire.Fire(main)
