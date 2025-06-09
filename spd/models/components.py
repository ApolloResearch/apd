import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.module_utils import init_param_


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 0, x, alpha * x)
    # return F.leaky_relu(x, negative_slope=alpha)


def upper_leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Small slope in the positive and negative regions."""
    # TODO: Make more memory efficient
    return torch.where(x > 1, 1 + alpha * (x - 1), F.relu(x))


class Gate(nn.Module):
    """A gate that maps a single input to a single output."""

    def __init__(self, m: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((m,)))
        self.bias = nn.Parameter(torch.zeros((m,)))
        fan_val = 1  # Since each weight gets applied independently
        init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

    def forward(self, x: Float[Tensor, "batch m"]) -> Float[Tensor, "batch m"]:
        return leaky_relu(torch.clamp(x * self.weight + self.bias, max=1))

    def forward_unclamped(self, x: Float[Tensor, "batch m"]) -> Float[Tensor, "batch m"]:
        return upper_leaky_relu(x * self.weight + self.bias)


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, m: int, n_gate_hidden_neurons: int):
        super().__init__()
        self.n_gate_hidden_neurons = n_gate_hidden_neurons

        self.mlp_in = nn.Parameter(torch.empty((m, n_gate_hidden_neurons)))
        self.in_bias = nn.Parameter(torch.zeros((m, n_gate_hidden_neurons)))
        self.mlp_out = nn.Parameter(torch.empty((m, n_gate_hidden_neurons)))
        self.out_bias = nn.Parameter(torch.zeros((m,)))

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=n_gate_hidden_neurons, nonlinearity="linear")

    def _compute_pre_activation(self, x: Float[Tensor, "batch m"]) -> Float[Tensor, "batch m"]:
        """Compute the output before applying the final activation function."""
        # First layer with gelu activation
        hidden = einops.einsum(
            x,
            self.mlp_in,
            "... m, m n_gate_hidden_neurons -> ... m n_gate_hidden_neurons",
        )
        hidden = hidden + self.in_bias
        hidden = F.gelu(hidden)

        # Second layer
        out = einops.einsum(
            hidden,
            self.mlp_out,
            "... m n_gate_hidden_neurons, m n_gate_hidden_neurons -> ... m",
        )
        out = out + self.out_bias
        return out

    @torch.compile
    def forward(self, x: Float[Tensor, "batch m"]) -> Float[Tensor, "batch m"]:
        return leaky_relu(torch.clamp(self._compute_pre_activation(x), max=1))

    @torch.compile
    def forward_unclamped(self, x: Float[Tensor, "batch m"]) -> Float[Tensor, "batch m"]:
        return upper_leaky_relu(self._compute_pre_activation(x))


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, m: int, bias: Tensor | None):
        super().__init__()
        self.m = m

        self.A = nn.Parameter(torch.empty(d_in, m))
        self.B = nn.Parameter(torch.empty(m, d_out))
        self.bias = bias

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=m, nonlinearity="linear")

        self.mask: Float[Tensor, "... m"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """A @ B"""
        return einops.einsum(self.A, self.B, "d_in m, m d_out -> d_out d_in")

    @torch.compile
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.A, "... d_in, d_in m -> ... m")

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(component_acts, self.B, "... m, m d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        m: int,
    ):
        super().__init__()
        self.m = m

        self.A = nn.Parameter(torch.empty(vocab_size, m))
        self.B = nn.Parameter(torch.empty(m, embedding_dim))

        # init_param_(self.A, fan_val=d_in, nonlinearity="linear")
        init_param_(self.A, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.B, fan_val=m, nonlinearity="linear")

        # For sparse forward passes
        self.mask: Float[Tensor, "batch pos m"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """A @ B"""
        return einops.einsum(
            self.A, self.B, "vocab_size m, ... m embedding_dim -> vocab_size embedding_dim"
        )

    @torch.compile
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.A[x]  # (batch pos m)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.B, "batch pos m, ... m embedding_dim -> batch pos embedding_dim"
        )
        return out
