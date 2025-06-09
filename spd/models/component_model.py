import fnmatch
from functools import partial
from pathlib import Path
from typing import Any

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.configs import Config
from spd.models.components import (
    EmbeddingComponent,
    Gate,
    GateMLP,
    LinearComponent,
)
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils import load_pretrained
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class ComponentModelPaths(BaseModel):
    """Paths to output files from a ComponentModel training run."""

    model: Path
    config: Path


class ComponentModel(nn.Module):
    """Wrapper around an arbitrary model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    match the patterns you pass in `target_module_patterns`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_module_patterns: list[str],
        m: int,
        n_gate_hidden_neurons: int | None,
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()
        self.model = base_model
        self.m = m
        self.pretrained_model_output_attr = pretrained_model_output_attr
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
        components: dict[str, LinearComponent | EmbeddingComponent] = {}
        matched_patterns: set[str] = set()

        for name, module in self.model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    matched_patterns.add(pattern)
                    if isinstance(module, nn.Linear):
                        d_out, d_in = module.weight.shape
                        # Replace "." with "-" in the name to avoid issues with module dict keys
                        components[name.replace(".", "-")] = LinearComponent(
                            d_in=d_in, d_out=d_out, m=m, bias=module.bias
                        )
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

        unmatched_patterns = set(target_module_patterns) - matched_patterns
        if unmatched_patterns:
            raise ValueError(
                f"The following patterns in target_module_patterns did not match any modules: "
                f"{sorted(unmatched_patterns)}"
            )

        if not components:
            raise ValueError(
                f"No modules found matching target_module_patterns: {target_module_patterns}"
            )
        return nn.ModuleDict(components)

    def to(self, *args: Any, **kwargs: Any) -> "ComponentModel":
        """Move the model and components to a device."""
        self.model.to(*args, **kwargs)
        for component in self.components.values():
            component.to(*args, **kwargs)
        for gate in self.gates.values():
            gate.to(*args, **kwargs)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model.

        If `model_output_attr` is set, return the attribute of the model's output.
        """
        raw_out = self.model(*args, **kwargs)
        if self.pretrained_model_output_attr is None:
            out = raw_out
        else:
            out = getattr(raw_out, self.pretrained_model_output_attr)
        return out

    def forward_with_component(
        self,
        *args: Any,
        module_name: str,
        component: LinearComponent | EmbeddingComponent,
        mask: Float[Tensor, "... m"] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with a single component replacement."""
        # Note that module_name uses "." separators but self.components use "-" separators
        old_module = self.model.get_submodule(module_name)
        assert old_module is not None

        self.model.set_submodule(module_name, component)
        if mask is not None:
            component.mask = mask

        out = self(*args, **kwargs)

        # Restore the original module
        self.model.set_submodule(module_name, old_module)

        component.mask = None

        return out

    def forward_with_components(
        self,
        *args: Any,
        components: dict[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... m"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacement."""
        # Note that components and masks uses "-" separators
        old_modules = {}
        for component_name, component in components.items():
            module_name = component_name.replace("-", ".")
            # component: LinearComponent = self.components[module_name.replace(".", "-")]
            old_module = self.model.get_submodule(module_name)
            assert old_module is not None
            old_modules[module_name] = old_module

            if masks is not None:
                component.mask = masks[component_name]
            self.model.set_submodule(module_name, component)

        out = self(*args, **kwargs)

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

        out = self(*args, **kwargs)

        for handle in handles:
            handle.remove()

        return out, cache

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> ComponentModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)

        return ComponentModelPaths(model=checkpoint_path, config=final_config_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["ComponentModel", Config, Path]:
        """Load a trained ComponentModel checkpoint along with its original config.

        The method supports two storage schemes:
        1.  A direct local path to the checkpoint file (plus `final_config.yaml` in
            the same directory).
        2.  A WandB reference of the form ``wandb:<entity>/<project>/runs/<run_id>``.
        """

        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            api = wandb.Api()
            run: Run = api.run(wandb_path)
            paths = cls._download_wandb_files(wandb_path)
            out_dir = fetch_wandb_run_dir(run.id)
        else:
            paths = ComponentModelPaths(
                model=Path(path), config=Path(path).parent / "final_config.yaml"
            )
            out_dir = Path(path).parent

        model_weights = torch.load(paths.model, map_location="cpu", weights_only=True)
        with open(paths.config) as f:
            config = Config(**yaml.safe_load(f))

        assert (
            config.pretrained_model_path is not None and config.pretrained_model_class is not None
        ), (
            "pretrained_model_name and pretrained_model_class must be specified in the config to "
            "reload a ComponentModel."
        )

        base_model_raw = load_pretrained(
            path_to_class=config.pretrained_model_class,
            model_path=config.pretrained_model_path,
            model_name_hf=config.pretrained_model_name_hf,
        )
        base_model = base_model_raw[0] if isinstance(base_model_raw, tuple) else base_model_raw

        comp_model = ComponentModel(
            base_model=base_model,
            target_module_patterns=config.target_module_patterns,
            m=config.m,
            n_gate_hidden_neurons=config.n_gate_hidden_neurons,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )
        comp_model.load_state_dict(model_weights)
        return comp_model, config, out_dir


def init_As_and_Bs_(
    model: ComponentModel, components: dict[str, LinearComponent | EmbeddingComponent]
) -> None:
    """Initialize the A and B matrices.
    1. Normalize every component to 1.
    2. Take inner product with original model
    3. This gives you roughly how much overlap there is with the target model.
    4. Scale the Bs by this value (just so it doesn't interfere with config.unit_norm_matrices
    """
    # NOTE: This may increase memory usage if done on GPU.
    for param_name, component in components.items():
        A = component.A
        B = component.B
        target_weight = model.model.get_parameter(param_name + ".weight")
        if isinstance(component, EmbeddingComponent):
            target_weight = target_weight.T  # (d_out d_in)

        # Make A and B have unit norm in the d_in and d_out dimensions
        A.data[:] = torch.randn_like(A.data)
        B.data[:] = torch.randn_like(B.data)
        A.data[:] = A.data / A.data.norm(dim=-2, keepdim=True)
        B.data[:] = B.data / B.data.norm(dim=-1, keepdim=True)

        # Calculate inner products
        m_norms = einops.einsum(A, B, target_weight, "d_in m, m d_out, d_out d_in -> m")
        # Scale B by the inner product.
        B.data[:] = B.data * m_norms.unsqueeze(-1)
