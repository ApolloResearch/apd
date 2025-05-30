from typing import Any

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.hooks import HookPoint
from spd.module_utils import init_param_


def hard_sigmoid(x: Tensor) -> Tensor:
    return F.relu(torch.clamp(x, max=1))


def upper_leaky_hard_sigmoid(x: Tensor) -> Tensor:
    return torch.where(x > 1, 1 + 0.01 * (x - 1), F.relu(x))


def lower_leaky_hard_sigmoid(x: Tensor) -> Tensor:
    return F.leaky_relu(torch.clamp(x, max=1))


class Gate(nn.Module):
    """A gate that maps a single input to a single output."""

    def __init__(self, m: int, n_instances: int | None = None):
        super().__init__()
        self.n_instances = n_instances
        shape = (n_instances, m) if n_instances is not None else (m,)
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        fan_val = 1  # Since each weight gets applied independently
        init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

    def forward(
        self, x: Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ) -> Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]:
        return hard_sigmoid(x * self.weight + self.bias)

    def forward_relu(
        self, x: Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ) -> Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]:
        return F.relu(x * self.weight + self.bias)


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, m: int, n_gate_hidden_neurons: int, n_instances: int | None = None):
        super().__init__()
        self.n_instances = n_instances
        self.n_gate_hidden_neurons = n_gate_hidden_neurons

        # Define weight shapes based on instances
        shape = (
            (n_instances, m, n_gate_hidden_neurons)
            if n_instances is not None
            else (m, n_gate_hidden_neurons)
        )
        in_bias_shape = (
            (n_instances, m, n_gate_hidden_neurons)
            if n_instances is not None
            else (m, n_gate_hidden_neurons)
        )
        out_bias_shape = (n_instances, m) if n_instances is not None else (m,)

        self.mlp_in = nn.Parameter(torch.empty(shape))
        self.in_bias = nn.Parameter(torch.zeros(in_bias_shape))
        self.mlp_out = nn.Parameter(torch.empty(shape))
        self.out_bias = nn.Parameter(torch.zeros(out_bias_shape))

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        # init_param_(self.in_bias, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=n_gate_hidden_neurons, nonlinearity="linear")

    def _compute_pre_activation(
        self, x: Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ) -> Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]:
        """Compute the output before applying the final activation function."""
        # First layer with gelu activation
        hidden = einops.einsum(
            x,
            self.mlp_in,
            "batch ... m, ... m n_gate_hidden_neurons -> batch ... m n_gate_hidden_neurons",
        )
        hidden = hidden + self.in_bias
        hidden = F.gelu(hidden)

        # Second layer
        out = einops.einsum(
            hidden,
            self.mlp_out,
            "batch ... m n_gate_hidden_neurons, ... m n_gate_hidden_neurons -> batch ... m",
        )
        out = out + self.out_bias
        return out

    def forward(
        self, x: Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ) -> Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]:
        # return hard_sigmoid(self._compute_pre_activation(x))
        return lower_leaky_hard_sigmoid(self._compute_pre_activation(x))

    def forward_relu(
        self, x: Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ) -> Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]:
        # return F.relu(self._compute_pre_activation(x))
        return upper_leaky_hard_sigmoid(self._compute_pre_activation(x))


class Linear(nn.Module):
    """A linear transformation with an optional n_instances dimension."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_instances: int | None = None,
    ):
        super().__init__()
        shape = (n_instances, d_in, d_out) if n_instances is not None else (d_in, d_out)
        self.weight = nn.Parameter(torch.empty(shape))
        # Note: init assumes no relu/gelu after this layer (which won't be the case for mlp_in, but
        # sqrt(2) ~= 1 so we're ignoring this for now.)
        init_param_(self.weight, fan_val=d_in, nonlinearity="linear")

        self.hook_pre = HookPoint()  # (batch ... d_in)
        self.hook_post = HookPoint()  # (batch ... d_out)

    def forward(
        self, x: Float[Tensor, "batch ... d_in"], *args: Any, **kwargs: Any
    ) -> Float[Tensor, "batch ... d_out"]:
        x = self.hook_pre(x)
        out = einops.einsum(x, self.weight, "batch ... d_in, ... d_in d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        m: int,
        n_instances: int | None = None,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.m = m

        # Initialize A and B matrices
        shape_A = (n_instances, d_in, m) if n_instances is not None else (d_in, m)
        shape_B = (n_instances, m, d_out) if n_instances is not None else (m, d_out)
        self.A = nn.Parameter(torch.empty(shape_A))
        self.B = nn.Parameter(torch.empty(shape_B))
        self.hook_pre = HookPoint()  # (batch d_in) or (batch n_instances d_in)
        self.hook_component_acts = HookPoint()  # (batch m) or (batch n_instances m)
        self.hook_post = HookPoint()  # (batch d_out) or (batch n_instances d_out)

        init_param_(self.A, fan_val=d_in, nonlinearity="linear")
        init_param_(self.B, fan_val=m, nonlinearity="linear")

    @property
    def weight(self) -> Float[Tensor, "... d_in d_out"]:
        """A @ B"""
        return einops.einsum(self.A, self.B, "... d_in m, ... m d_out -> ... d_in d_out")

    def forward(
        self, x: Float[Tensor, "batch ... d_in"], mask: Float[Tensor, "batch ... m"] | None = None
    ) -> Float[Tensor, "batch ... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all subnetworks
        """
        x = self.hook_pre(x)

        # First multiply by A to get to intermediate dimension m
        component_acts = einops.einsum(x, self.A, "batch ... d_in, ... d_in m -> batch ... m")
        if mask is not None:
            component_acts *= mask

        component_acts = self.hook_component_acts(component_acts)
        # Then multiply by B to get to output dimension
        out = einops.einsum(component_acts, self.B, "batch ... m, ... m d_out -> batch ... d_out")

        out = self.hook_post(out)
        return out


class TransposedLinear(Linear):
    """Linear layer that uses a transposed weight from another Linear layer.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original Linear layer.
    """

    def __init__(self, original_weight: nn.Parameter):
        # Copy the relevant parts from Linear.__init__. Don't copy operations that will call
        # TransposedLinear.weight.
        nn.Module.__init__(self)
        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_weight", original_weight, persistent=False)

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        return einops.rearrange(self.original_weight, "... d_in d_out -> ... d_out d_in")


class TransposedLinearComponent(LinearComponent):
    """LinearComponent that uses a transposed weight from another LinearComponent.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original LinearComponent.
    """

    def __init__(self, original_A: nn.Parameter, original_B: nn.Parameter):
        # Copy the relevant parts from LinearComponent.__init__. Don't copy operations that will
        # call TransposedLinear.A or TransposedLinear.B.
        nn.Module.__init__(self)
        self.n_instances, _, self.m = original_A.shape

        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_component_acts = HookPoint()  # (batch ... m)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_A", original_A, persistent=False)
        self.register_buffer("original_B", original_B, persistent=False)

    @property
    def A(self) -> Float[Tensor, "... d_out m"]:
        # New A is the transpose of the original B
        return einops.rearrange(self.original_B, "... m d_out -> ... d_out m")

    @property
    def B(self) -> Float[Tensor, "... d_in m"]:
        # New B is the transpose of the original A
        return einops.rearrange(self.original_A, "... d_in m -> ... m d_in")

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        """A @ B"""
        return einops.einsum(self.A, self.B, "... d_out m, ... m d_in -> ... d_out d_in")
