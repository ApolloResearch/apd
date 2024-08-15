import math
import os
import random
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import wandb
import yaml
from dotenv import load_dotenv
from jaxtyping import Float
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import Tensor

from spd.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)


def to_root_path(path: str | Path):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def permute_to_identity(x: torch.Tensor, normalize_rows: bool = False) -> torch.Tensor:
    """Permute the rows of a matrix such that the maximum value in each column is on the leading
    diagonal.

    Args:
        x: The input matrix.
        normalize_rows: Whether to normalize the rows of the output matrix.
    """

    # Assert that arr only has two dimensions and that it is square
    assert x.dim() == 2
    assert x.shape[0] == x.shape[1], "Must have the same number of subnetworks (k) as features"

    # Get the number of rows and columns
    n_rows, n_cols = x.shape

    # Find the row index of the maximum value in each column
    max_row_indices_raw = torch.argmax(x, dim=0).tolist()

    # Get the indices of the non unique max_row_indices
    unique_indices = set()
    duplicate_indices = []
    for i in range(n_rows):
        if max_row_indices_raw[i] in unique_indices:
            duplicate_indices.append(i)
        else:
            unique_indices.add(max_row_indices_raw[i])

    remaining_indices = [i for i in range(n_rows) if i not in unique_indices]
    # Now we want to swap out the duplicate indices with any remaining indices
    for i in range(len(duplicate_indices)):
        max_row_indices_raw[duplicate_indices[i]] = remaining_indices[i]

    # Ensure that we output a permuted version and have no duplicate rows
    assert set(max_row_indices_raw) == set(range(n_rows))

    out_rows = x[max_row_indices_raw]

    if normalize_rows:
        out_rows = out_rows / out_rows.norm(dim=1, p=2, keepdim=True)
    return out_rows


def calculate_closeness_to_identity(x: Float[Tensor, "... a b"]) -> float:
    """Frobenius norm of the difference between the input matrix and the identity matrix.

    If x has more than two dimensions, the result is meaned over all but the final two dimensions.
    """
    eye = torch.eye(n=x.shape[-2], m=x.shape[-1], device=x.device)
    return torch.norm(x - eye, p="fro", dim=(-2, -1)).mean().item()


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def load_config(config_path_or_obj: Path | str | T, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def init_wandb(config: T, project: str, sweep_config_path: Path | str | None) -> T:
    """Initialize Weights & Biases and return a config updated with sweep hyperparameters.

    If no sweep config is provided, the config is returned as is.

    If a sweep config is provided, wandb is first initialized with the sweep config. This will
    cause wandb to choose specific hyperparameters for this instance of the sweep and store them
    in wandb.config. We then update the config with these hyperparameters.

    Args:
        config: The base config.
        project: The name of the wandb project.
        sweep_config_path: The path to the sweep config file. If provided, updates the config with
            the hyperparameters from this instance of the sweep.

    Returns:
        Config updated with sweep hyperparameters (if any).
    """
    if sweep_config_path is not None:
        with open(sweep_config_path) as f:
            sweep_data = yaml.safe_load(f)
        wandb.init(config=sweep_data, save_code=True)
    else:
        load_dotenv(override=True)
        wandb.init(project=project, entity=os.getenv("WANDB_ENTITY"), save_code=True)

    # Update the config with the hyperparameters for this sweep (if any)
    config = replace_pydantic_model(config, wandb.config)

    # Update the non-frozen keys in the wandb config (only relevant for sweeps)
    wandb.config.update(config.model_dump(mode="json"))
    return config


def init_param_(param: torch.Tensor) -> None:
    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
