import json
from collections.abc import Callable
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.base import Model, SPDModel
from spd.models.bool_circuit_models import MLPComponents
from spd.scripts.multilayer_functions.piecewise_linear import MLP, ControlledResNet


class PiecewiseFunctionTransformer(Model):
    def __init__(self, n_inputs: int, d_mlp: int, num_layers: int, d_embed: int | None = None):
        super().__init__()
        self.n_inputs = n_inputs
        self.num_layers = num_layers
        self.d_embed = self.n_inputs + 1 if d_embed is None else d_embed
        self.d_control = self.d_embed - 2

        self.num_functions = n_inputs - 1
        self.n_outputs = 1  # this is hardcoded. This class isn't defined for multiple outputs

        self.superposition = self.num_functions > self.d_control
        if not self.superposition:
            assert self.num_functions == self.d_control

        self.W_E = nn.Linear(n_inputs, self.d_embed, bias=False)
        self.W_U = nn.Linear(self.d_embed, self.n_outputs, bias=False)

        self.initialise_embeds()

        self.mlps = nn.ModuleList(
            [MLP(d_model=self.d_embed, d_mlp=d_mlp) for _ in range(num_layers)]
        )

    def initialise_embeds(self, random_matrix: Tensor | None = None):
        self.W_E.weight.data = torch.zeros(self.d_embed, self.n_inputs)
        self.W_E.weight.data[0, 0] = 1.0
        if not self.superposition:
            self.W_E.weight.data[1:-1, 1:] = torch.eye(self.num_functions)
        else:
            random_matrix = (
                torch.randn(self.d_control, self.num_functions)
                if random_matrix is None
                else random_matrix
            )
            random_normalised = random_matrix / torch.norm(random_matrix, dim=1, keepdim=True)
            self.W_E.weight.data[1:-1, 1:] = random_normalised

        self.W_U.weight.data = torch.zeros(self.n_outputs, self.d_embed)
        self.W_U.weight.data[:, -1] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        residual = self.W_E(x)
        for layer in self.mlps:
            residual = residual + layer(residual)
        return self.W_U(residual)

    @property
    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        params = []
        for mlp in self.mlps:
            params.append(mlp.input_layer.weight.T)
            params.append(mlp.output_layer.weight.T)
        return params

    @classmethod
    def from_handcoded(
        cls, functions: list[Callable[[float], float]]
    ) -> "PiecewiseFunctionTransformer":
        n_inputs = len(functions) + 2
        neurons_per_function = 20
        num_layers = 4
        d_mlp = neurons_per_function * len(functions) // num_layers
        d_embed = n_inputs
        start = 0
        end = 5
        model = cls(n_inputs=n_inputs, d_mlp=d_mlp, num_layers=num_layers, d_embed=d_embed)
        # Note that our MLP differs from the bool_circuit_models.MLP in having b_out
        # Also different names
        handcoded_model = ControlledResNet(
            functions,
            start=start,
            end=end,
            neurons_per_function=neurons_per_function,
            num_layers=num_layers,
            d_control=d_embed - 2,
            negative_suppression=end + 1,
        )
        # Copy the weights from the hand-coded model to the model

        # the control_W_E of the ControlledResNet class is just a part of the W_E of this class. In
        # particular it is the part that is sliced in by the random matrix (or identity)
        model.initialise_embeds(handcoded_model.control_W_E)

        for i, mlp in enumerate(handcoded_model.mlps):
            model.mlps[i].input_layer.weight.data = mlp.input_layer.weight
            model.mlps[i].output_layer.weight.data = mlp.output_layer.weight

            model.mlps[i].input_layer.bias.data = mlp.input_layer.bias
            assert torch.all(mlp.output_layer.bias.data == 0), "Output layer bias should be zero"
            model.mlps[i].output_layer.bias.data = mlp.output_layer.bias

        return model


class PiecewiseFunctionSPDTransformer(SPDModel):
    def __init__(
        self, n_inputs: int, d_mlp: int, num_layers: int, k: int, d_embed: int | None = None
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.num_layers = num_layers
        self.k = k
        self.d_embed = self.n_inputs + 1 if d_embed is None else d_embed
        self.d_control = self.d_embed - 2

        self.num_functions = n_inputs - 1
        self.n_outputs = 1  # this is hardcoded. This class isn't defined for multiple outputs

        self.superposition = self.num_functions > self.d_control
        if not self.superposition:
            assert self.num_functions == self.d_control

        self.W_E = nn.Linear(n_inputs, self.d_embed, bias=False)
        self.W_U = nn.Linear(self.d_embed, self.n_outputs, bias=False)

        self.initialise_embeds()

        self.mlps = nn.ModuleList(
            [MLPComponents(self.d_embed, d_mlp, k) for _ in range(num_layers)]
        )  # TODO: Check what is going on with bias2 in MLPComponents

    def initialise_embeds(self):
        self.W_E.weight.data = torch.zeros(self.d_embed, self.n_inputs)
        self.W_E.weight.data[0, 0] = 1.0
        if not self.superposition:
            self.W_E.weight.data[1:-1, 1:] = torch.eye(self.num_functions)
        else:
            random_matrix = torch.randn(self.d_control, self.num_functions)
            random_normalised = random_matrix / torch.norm(random_matrix, dim=1, keepdim=True)
            self.W_E.weight.data[1:-1, 1:] = random_normalised

        self.W_U.weight.data = torch.zeros(self.n_outputs, self.d_embed)
        self.W_U.weight.data[:, -1] = 1.0

    @property
    def all_As(self) -> list[Float[Tensor, "dim k"]]:
        all_A_pairs = [
            (self.mlps[i].linear1.A, self.mlps[i].linear2.A) for i in range(self.n_layers)
        ]
        As = [A for A_pair in all_A_pairs for A in A_pair]
        assert len(As) == self.n_param_matrices
        return As

    @property
    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        # Get all B matrices
        all_B_pairs = [
            (self.mlps[i].linear1.B, self.mlps[i].linear2.B) for i in range(self.n_layers)
        ]
        As = [B for B_pair in all_B_pairs for B in B_pair]
        assert len(As) == self.n_param_matrices
        return As

    def forward(
        self, x: Float[Tensor, "... inputs"]
    ) -> tuple[
        Float[Tensor, "... outputs"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Returns:
            x: The output of the model
            layer_acts: A list of activations for each layer in each MLP.
            inner_acts: A list of component activations for each layer in each MLP.
        """
        layer_acts = []
        inner_acts = []
        residual = self.W_E(x)
        for layer in self.mlps:
            residual, layer_acts_i, inner_acts_i = layer(residual)
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)
        return self.W_U(residual), layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... inputs"],
        topk: int,
        all_grads: list[Float[Tensor, "... k"]] | None = None,
    ) -> tuple[
        Float[Tensor, "... outputs"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each component activation.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            all_grads: Optional list of gradients for each layer's components

        Returns:
            output: The output of the transformer
            layer_acts: A list of activations for each layer in each MLP
            inner_acts: A list of component activations for each layer in each MLP
        """
        layer_acts = []
        inner_acts = []
        residual = self.W_E(x)

        n_param_matrices_per_layer = self.n_param_matrices // self.n_layers

        for i, layer in enumerate(self.mlps):
            # A single layer contains multiple parameter matrices
            layer_grads = (
                all_grads[i * n_param_matrices_per_layer : (i + 1) * n_param_matrices_per_layer]
                if all_grads is not None
                else None
            )
            residual, layer_acts_i, inner_acts_i = layer.forward_topk(residual, topk, layer_grads)
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)

        return self.W_U(residual), layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "PiecewiseFunctionSPDTransformer":
        path = Path(path)
        with open(path.parent / "config.json") as f:
            config = json.load(f)

        params = torch.load(path)

        model = cls(
            n_inputs=config["n_inputs"],
            d_mlp=config["d_mlp"],
            num_layers=config["num_layers"],
            k=config["k"],
            d_embed=config["d_embed"],
        )
        model.load_state_dict(params)
        return model
