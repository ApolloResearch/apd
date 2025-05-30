import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.log import logger
from spd.module_utils import init_param_
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class ResidualMLPPaths(BaseModel):
    """Paths to output files from a ResidualMLPModel training run."""

    resid_mlp_train_config: Path
    label_coeffs: Path
    checkpoint: Path


class ResidualMLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"] = Field(
        description="Defines the activation function in the model. Also used in the labeling "
        "function if label_type is act_plus_resid."
    )
    in_bias: bool
    out_bias: bool


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn

        self.mlp_in = nn.Linear(d_model, d_mlp, bias=in_bias)
        self.mlp_out = nn.Linear(d_mlp, d_model, bias=out_bias)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        mid_pre_act_fn = self.mlp_in(x)
        mid = self.act_fn(mid_pre_act_fn)
        out = self.mlp_out(mid)
        return out


class ResidualMLPModel(nn.Module):
    def __init__(self, config: ResidualMLPConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty(config.n_features, config.d_embed))
        init_param_(self.W_E, fan_val=config.n_features, nonlinearity="linear")
        self.W_U = nn.Parameter(torch.empty(config.d_embed, config.n_features))
        init_param_(self.W_U, fan_val=config.d_embed, nonlinearity="linear")

        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu
        self.layers = nn.ModuleList(
            [
                MLP(
                    d_model=config.d_embed,
                    d_mlp=config.d_mlp,
                    act_fn=self.act_fn,
                    in_bias=config.in_bias,
                    out_bias=config.out_bias,
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(
        self,
        x: Float[Tensor, "... n_features"],
        return_residual: bool = False,
    ) -> Float[Tensor, "... n_features"] | Float[Tensor, "... d_embed"]:
        residual = einops.einsum(x, self.W_E, "... n_features, n_features d_embed -> ... d_embed")
        for layer in self.layers:
            out = layer(residual)
            residual = residual + out
        if return_residual:
            return residual
        out = einops.einsum(
            residual,
            self.W_U,
            "... d_embed, d_embed n_features -> ... n_features",
        )
        return out

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> ResidualMLPPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        resid_mlp_train_config_path = download_wandb_file(
            run, run_dir, "resid_mlp_train_config.yaml"
        )
        label_coeffs_path = download_wandb_file(run, run_dir, "label_coeffs.json")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        logger.info(f"Downloaded checkpoint from {checkpoint_path}")
        return ResidualMLPPaths(
            resid_mlp_train_config=resid_mlp_train_config_path,
            label_coeffs=label_coeffs_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls, path: ModelPath
    ) -> tuple["ResidualMLPModel", dict[str, Any], Float[Tensor, " n_features"]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained ResidualMLPModel
            resid_mlp_train_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
            label_coeffs: The label coefficients used to train the model
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = ResidualMLPPaths(
                resid_mlp_train_config=Path(path).parent / "resid_mlp_train_config.yaml",
                label_coeffs=Path(path).parent / "label_coeffs.json",
                checkpoint=Path(path),
            )

        with open(paths.resid_mlp_train_config) as f:
            resid_mlp_train_config_dict = yaml.safe_load(f)

        with open(paths.label_coeffs) as f:
            label_coeffs = torch.tensor(json.load(f))

        # Remove n_instances, apply_output_act_fn, and init_scale from the arguments
        # For backward compatibility
        resid_mlp_train_config_dict["resid_mlp_config"].pop("n_instances", None)
        resid_mlp_train_config_dict["resid_mlp_config"].pop("apply_output_act_fn", None)
        resid_mlp_train_config_dict["resid_mlp_config"].pop("init_scale", None)
        resid_mlp_config = ResidualMLPConfig(**resid_mlp_train_config_dict["resid_mlp_config"])
        resid_mlp = cls(resid_mlp_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        # Squeeze all parameters
        params = {k: v.squeeze() for k, v in params.items()}
        # Rename "layers.0.linear1" to "layers.0.mlp_in.weight" for each layer
        params["layers.0.mlp_in.weight"] = params.pop("layers.0.linear1").T
        params["layers.0.mlp_out.weight"] = params.pop("layers.0.linear2").T
        resid_mlp.load_state_dict(params)

        return resid_mlp, resid_mlp_train_config_dict, label_coeffs


# class ResidualMLPSPDPaths(BaseModel):
#     """Paths to output files from a ResidualMLPSPDModel training run."""

#     final_config: Path
#     resid_mlp_train_config: Path
#     label_coeffs: Path
#     checkpoint: Path


# class ResidualMLPSPDConfig(BaseModel):
#     model_config = ConfigDict(extra="forbid", frozen=True)
#     n_instances: PositiveInt
#     n_features: PositiveInt
#     d_embed: PositiveInt
#     d_mlp: PositiveInt
#     n_layers: PositiveInt
#     act_fn_name: Literal["gelu", "relu"]
#     apply_output_act_fn: bool
#     in_bias: bool
#     out_bias: bool
#     m: PositiveInt
#     n_gate_hidden_neurons: PositiveInt | None = None
#     init_type: Literal["kaiming_uniform", "xavier_normal"] = "xavier_normal"


# class ResidualMLPSPDModel(SPDModel):
#     def __init__(
#         self,
#         config: ResidualMLPSPDConfig,
#     ):
#         super().__init__()
#         self.config = config
#         self.n_features = config.n_features  # Required for backward compatibility
#         self.n_instances = config.n_instances  # Required for backward compatibility
#         self.m = config.m

#         assert config.act_fn_name in ["gelu", "relu"]
#         self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu

#         self.W_E = nn.Parameter(torch.empty(config.n_instances, config.n_features, config.d_embed))
#         self.W_U = nn.Parameter(torch.empty(config.n_instances, config.d_embed, config.n_features))
#         init_param_(self.W_E, fan_val=config.n_features, nonlinearity="linear")
#         init_param_(self.W_U, fan_val=config.d_embed, nonlinearity="linear")

#         self.layers = nn.ModuleList()

#         # Use GateMLP if n_gate_hidden_neurons is provided, otherwise use Gate
#         gate_class = GateMLP if config.n_gate_hidden_neurons is not None else Gate
#         gate_kwargs = {"m": self.m, "n_instances": config.n_instances}
#         if config.n_gate_hidden_neurons is not None:
#             gate_kwargs["n_gate_hidden_neurons"] = config.n_gate_hidden_neurons

#         self.gates = nn.ModuleDict()
#         for i in range(config.n_layers):
#             self.layers.append(
#                 MLP(
#                     n_instances=config.n_instances,
#                     d_model=config.d_embed,
#                     d_mlp=config.d_mlp,
#                     in_bias=config.in_bias,
#                     out_bias=config.out_bias,
#                     act_fn=self.act_fn,
#                     spd_kwargs={"m": self.m},
#                 )
#             )
#             self.gates[f"layers-{i}-mlp_in"] = gate_class(**gate_kwargs)
#             self.gates[f"layers-{i}-mlp_out"] = gate_class(**gate_kwargs)

#         self.setup()

#     def forward(
#         self,
#         x: Float[Tensor, "batch n_instances n_features"],
#         masks: dict[str, Float[Tensor, "batch n_instances m"]] | None = None,
#     ) -> Float[Tensor, "batch n_instances d_embed"]:
#         """
#         Returns:
#             x: The output of the model
#         """
#         residual = einops.einsum(
#             x,
#             self.W_E,
#             "batch n_instances n_features, n_instances n_features d_embed -> batch n_instances d_embed",
#         )
#         for i, layer in enumerate(self.layers):
#             mlp_in_mask = masks[f"layers.{i}.mlp_in"] if masks is not None else None
#             mlp_out_mask = masks[f"layers.{i}.mlp_out"] if masks is not None else None
#             residual = residual + layer(
#                 residual, mlp_in_mask=mlp_in_mask, mlp_out_mask=mlp_out_mask
#             )
#         out = einops.einsum(
#             residual,
#             self.W_U,
#             "batch n_instances d_embed, n_instances d_embed n_features -> batch n_instances n_features",
#         )
#         if self.config.apply_output_act_fn:
#             out = self.act_fn(out)
#         return out

#     @staticmethod
#     def _download_wandb_files(wandb_project_run_id: str) -> ResidualMLPSPDPaths:
#         """Download the relevant files from a wandb run."""
#         api = wandb.Api()
#         run: Run = api.run(wandb_project_run_id)

#         checkpoint = fetch_latest_wandb_checkpoint(run, prefix="spd_model")

#         run_dir = fetch_wandb_run_dir(run.id)

#         final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
#         resid_mlp_train_config_path = download_wandb_file(
#             run, run_dir, "resid_mlp_train_config.yaml"
#         )
#         label_coeffs_path = download_wandb_file(run, run_dir, "label_coeffs.json")
#         checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
#         logger.info(f"Downloaded checkpoint from {checkpoint_path}")
#         return ResidualMLPSPDPaths(
#             final_config=final_config_path,
#             resid_mlp_train_config=resid_mlp_train_config_path,
#             label_coeffs=label_coeffs_path,
#             checkpoint=checkpoint_path,
#         )

#     @classmethod
#     def from_pretrained(
#         cls, path: str | Path
#     ) -> tuple["ResidualMLPSPDModel", Config, Float[Tensor, "n_instances n_features"]]:
#         """Fetch a pretrained model from wandb or a local path to a checkpoint.

#         Args:
#             path: The path to local checkpoint or wandb project. If a wandb project, the format
#                 must be `wandb:entity/project/run_id`. If `api.entity` is set (e.g. via setting
#                 WANDB_ENTITY in .env), this can be in the form `wandb:project/run_id` and if
#                 form `wandb:project/run_id` and if `api.project` is set this can just be
#                 `wandb:run_id`. If local path, assumes that `resid_mlp_train_config.yaml` and
#                 `label_coeffs.json` are in the same directory as the checkpoint.

#         Returns:
#             model: The pretrained ResidualMLPSPDModel
#             config: The config used to train the model
#             label_coeffs: The label coefficients used to train the model
#         """
#         if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
#             wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
#             paths = cls._download_wandb_files(wandb_path)
#         else:
#             paths = ResidualMLPSPDPaths(
#                 final_config=Path(path).parent / "final_config.yaml",
#                 resid_mlp_train_config=Path(path).parent / "resid_mlp_train_config.yaml",
#                 label_coeffs=Path(path).parent / "label_coeffs.json",
#                 checkpoint=Path(path),
#             )

#         with open(paths.final_config) as f:
#             final_config_dict = yaml.safe_load(f)

#         final_config_dict.pop("post_relu_act_recon", None)
#         config = Config(**final_config_dict)

#         with open(paths.resid_mlp_train_config) as f:
#             resid_mlp_train_config_dict = yaml.safe_load(f)

#         with open(paths.label_coeffs) as f:
#             label_coeffs = torch.tensor(json.load(f))

#         assert isinstance(config.task_config, ResidualMLPTaskConfig)
#         resid_mlp_spd_config = ResidualMLPSPDConfig(
#             **resid_mlp_train_config_dict["resid_mlp_config"],
#             m=config.m,
#             n_gate_hidden_neurons=config.n_gate_hidden_neurons,
#         )
#         model = cls(config=resid_mlp_spd_config)
#         params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")

#         params = replace_deprecated_param_names(
#             params, name_map={"linear1": "mlp_in", "linear2": "mlp_out"}
#         )

#         model.load_state_dict(params)
#         return model, config, label_coeffs
