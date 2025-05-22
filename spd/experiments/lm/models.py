"""
Defines a SSModel class that is a wrapper around a llama model from SimpleStories
"""

import fnmatch
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from torch import Tensor
from wandb.apis.public import Run

from spd.configs import Config, LMTaskConfig
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class LinearComponentWithBias(nn.Module):
    """A LinearComponent with a bias parameter."""

    def __init__(self, linear_component: LinearComponent, bias: Tensor | None):
        super().__init__()
        self.linear_component = linear_component
        self.bias = bias
        self.mask: Float[Tensor, "... m"] | None = None  # Gets set on sparse forward passes
        self.A = linear_component.A
        self.B = linear_component.B
        self.weight = linear_component.weight

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # Note: We assume bias is added *after* the component multiplication
        # Also assume input is (batch, seq_len, d_in)
        out = self.linear_component(x, mask=self.mask)
        if self.bias is not None:
            out += self.bias
        return out


def linear_module_to_component(
    linear_module: nn.Linear,
    m: int,
) -> LinearComponentWithBias:
    """Convert an nn.Linear into a LinearComponentWithBias."""
    d_out, d_in = linear_module.weight.shape
    linear_component = LinearComponent(d_in=d_in, d_out=d_out, m=m, n_instances=None)
    # # Initialize with A = W (original weights) and B = I (identity)
    # # This provides a starting point where the component exactly equals the original
    # linear_component.A.data[:] = linear_module.weight.t()  # (d_in, m)
    # linear_component.B.data[:] = torch.eye(m)
    bias = linear_module.bias.clone() if linear_module.bias is not None else None  # type: ignore
    return LinearComponentWithBias(linear_component, bias)


class SSModelPaths(BaseModel):
    """Paths to output files from a SSModel training run."""

    model: Path
    config: Path


class SSModel(nn.Module):
    """Wrapper around a llama model from SimpleStories for running SPD."""

    def __init__(
        self,
        llama_model: Llama,
        target_module_patterns: list[str],
        m: int,
        n_gate_hidden_neurons: int | None,
    ):
        super().__init__()
        self.model = llama_model
        self.m = m
        self.components = self.create_target_components(
            target_module_patterns=target_module_patterns, m=m
        )

        # Use GateMLP if n_gate_hidden_neurons is provided, otherwise use Gate
        gate_class = GateMLP if n_gate_hidden_neurons is not None else Gate
        gate_kwargs = {"m": m}
        if n_gate_hidden_neurons is not None:
            gate_kwargs["n_gate_hidden_neurons"] = n_gate_hidden_neurons

        self.gates = nn.ModuleDict({name: gate_class(**gate_kwargs) for name in self.components})

    def create_target_components(self, target_module_patterns: list[str], m: int) -> nn.ModuleDict:
        """Create target components for the model."""
        components: dict[str, LinearComponentWithBias | EmbeddingComponent] = {}
        for name, module in self.model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    if isinstance(module, nn.Linear):
                        # Replace "." with "-" in the name to avoid issues with module dict keys
                        components[name.replace(".", "-")] = linear_module_to_component(module, m=m)
                    elif isinstance(module, nn.Embedding):
                        components[name.replace(".", "-")] = EmbeddingComponent(
                            vocab_size=module.num_embeddings,
                            embedding_dim=module.embedding_dim,
                            m=m,
                        )
                    else:
                        raise ValueError(
                            f"Module '{name}' matched pattern '{pattern}' but is not nn.Linear or "
                            f"nn.Embedding. Found type: {type(module)}"
                        )
                    break
        return nn.ModuleDict(components)

    def to(self, *args: Any, **kwargs: Any) -> "SSModel":
        """Move the model and components to a device."""
        self.model.to(*args, **kwargs)
        for component in self.components.values():
            component.to(*args, **kwargs)
        for gate in self.gates.values():
            gate.to(*args, **kwargs)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model."""
        return self.model(*args, **kwargs)

    def forward_with_component(
        self,
        *args: Any,
        module_name: str,
        component: LinearComponentWithBias | EmbeddingComponent,
        mask: Float[Tensor, "batch pos m"] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with a single component replacement."""
        # Note that module_name uses "." separators but self.components use "-" separators
        old_module = self.model.get_submodule(module_name)
        assert old_module is not None

        self.model.set_submodule(module_name, component)
        if mask is not None:
            component.mask = mask

        out = self.model(*args, **kwargs)

        self.model.set_submodule(module_name, old_module)

        component.mask = None

        return out

    def forward_with_components(
        self,
        *args: Any,
        components: dict[str, LinearComponentWithBias | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "batch pos m"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacement."""
        # Note that components and masks uses "-" separators
        old_modules = {}
        for component_name, component in components.items():
            module_name = component_name.replace("-", ".")
            # component: LinearComponentWithBias = self.components[module_name.replace(".", "-")]
            old_module = self.model.get_submodule(module_name)
            assert old_module is not None
            old_modules[module_name] = old_module

            if masks is not None:
                component.mask = masks[component_name]
            self.model.set_submodule(module_name, component)

        out = self.model(*args, **kwargs)

        # Restore the original modules
        for module_name, old_module in old_modules.items():
            self.model.set_submodule(module_name, old_module)

        # Remove the masks attribute from the components
        for component in components.values():
            component.mask = None

        return out

    def forward_with_pre_forward_cache_hooks(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at in the input to the modules given by `module_names`.

        Args:
            module_names: List of module names to cache the inputs to.
        """
        cache = {}

        def cache_hook(module: nn.Module, input: tuple[Tensor, ...], param_name: str) -> Tensor:
            cache[param_name] = input[0]
            return input[0]

        handles: list[torch.utils.hooks.RemovableHandle] = []
        for module_name in module_names:
            module = self.model.get_submodule(module_name)
            assert module is not None
            handles.append(
                module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            )

        out = self.forward(*args, **kwargs)

        for handle in handles:
            handle.remove()

        return out, cache

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> SSModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)

        return SSModelPaths(model=checkpoint_path, config=final_config_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["SSModel", Config, Path]:
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            api = wandb.Api()
            run: Run = api.run(wandb_path)
            paths = cls._download_wandb_files(wandb_path)
            out_dir = fetch_wandb_run_dir(run.id)

        else:
            paths = SSModelPaths(model=Path(path), config=Path(path).parent / "final_config.yaml")
            out_dir = Path(path).parent

        model_weights = torch.load(paths.model, map_location="cpu", weights_only=True)
        with open(paths.config) as f:
            config = Config(**yaml.safe_load(f))

        assert isinstance(config.task_config, LMTaskConfig)
        model_config_dict = MODEL_CONFIGS[config.task_config.model_size]
        model_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
        llama_model = Llama.from_pretrained(model_path, model_config_dict)

        n_unique_tokens = 1000
        from spd.utils import replace_pydantic_model

        model_config_dict = replace_pydantic_model(
            model_config_dict, {"vocab_size": n_unique_tokens}
        )

        llama_model.transformer.wte = nn.Embedding(
            num_embeddings=model_config_dict.vocab_size,
            embedding_dim=model_config_dict.n_embd,
        )

        ss_model = SSModel(
            llama_model=llama_model,
            target_module_patterns=config.task_config.target_module_patterns,
            m=config.m,
            n_gate_hidden_neurons=config.n_gate_hidden_neurons,
        )
        ss_model.load_state_dict(model_weights)
        return ss_model, config, out_dir
