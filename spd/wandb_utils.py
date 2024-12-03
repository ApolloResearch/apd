import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import wandb
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from wandb.apis.public import File, Run

from spd.settings import REPO_ROOT
from spd.utils import replace_pydantic_model

T = TypeVar("T", bound=BaseModel)


def fetch_latest_wandb_checkpoint(run: Run) -> File:
    """Fetch the latest checkpoint from a wandb run.

    NOTE: Assumes that the only files that end in `.pth` are checkpoints.
    """
    # Get the latest checkpoint. Assume format is <name>_<step>.pth or <name>.pth
    checkpoints = [file for file in run.files() if file.name.endswith(".pth")]
    if not checkpoints:
        raise ValueError(f"No checkpoint files found in run {run.name}")

    if len(checkpoints) == 1:
        latest_checkpoint_remote = checkpoints[0]
    else:
        # Assume format is <name>_<step>.pth
        latest_checkpoint_remote = sorted(
            checkpoints, key=lambda x: int(x.name.split(".pth")[0].split("_")[-1])
        )[-1]
    return latest_checkpoint_remote


def fetch_wandb_run_dir(run_id: str) -> Path:
    """Find or create a directory in the W&B cache for a given run.

    We first check if we already have a directory with the suffix "run_id" (if we created the run
    ourselves, a directory of the name "run-<timestamp>-<run_id>" should exist). If not, we create a
    new wandb_run_dir.
    """
    # Default to REPO_ROOT/wandb if SPD_CACHE_DIR not set
    base_cache_dir = Path(os.environ.get("SPD_CACHE_DIR", REPO_ROOT / "wandb"))

    # Set default wandb_run_dir
    wandb_run_dir = base_cache_dir / run_id / "files"

    # Check if we already have a directory with the suffix "run_id"
    presaved_run_dirs = [
        d for d in base_cache_dir.iterdir() if d.is_dir() and d.name.endswith(run_id)
    ]
    # If there is more than one dir, just ignore the presaved dirs and use the new wandb_run_dir
    if presaved_run_dirs and len(presaved_run_dirs) == 1:
        presaved_file_path = presaved_run_dirs[0] / "files"
        if presaved_file_path.exists():
            # Found a cached run directory, use it
            wandb_run_dir = presaved_file_path

    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    return wandb_run_dir


def download_wandb_file(run: Run, wandb_run_dir: Path, file_name: str) -> Path:
    """Download a file from W&B. Don't overwrite the file if it already exists.

    Args:
        run: The W&B run to download from
        file_name: Name of the file to download
        wandb_run_dir: The directory to download the file to
    Returns:
        Path to the downloaded file
    """
    file_on_wandb = run.file(file_name)
    assert isinstance(file_on_wandb, File)
    path = Path(file_on_wandb.download(exist_ok=True, replace=False, root=str(wandb_run_dir)).name)
    return path


def init_wandb(
    config: T, project: str, sweep_config_path: Path | str | None = None, name: str | None = None
) -> T:
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
        name: The name of the wandb run.

    Returns:
        Config updated with sweep hyperparameters (if any).
    """
    if sweep_config_path is not None:
        with open(sweep_config_path) as f:
            sweep_data = yaml.safe_load(f)
        wandb.init(config=sweep_data, save_code=True, name=name)
    else:
        load_dotenv(override=True)
        wandb.init(project=project, entity=os.getenv("WANDB_ENTITY"), save_code=True, name=name)

    # Update the config with the hyperparameters for this sweep (if any)
    config = replace_pydantic_model(config, wandb.config)

    # Update the non-frozen keys in the wandb config (only relevant for sweeps)
    wandb.config.update(config.model_dump(mode="json"))
    return config


def save_config_to_wandb(config: BaseModel, filename: str = "final_config.yaml") -> None:
    # Save the config to wandb
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / filename
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        wandb.save(str(config_path), policy="now", base_path=tmp_dir)
        # Unfortunately wandb.save is async, so we need to wait for it to finish before
        # continuing, and wandb python api provides no way to do this.
        # TODO: Find a better way to do this.
        time.sleep(1)
