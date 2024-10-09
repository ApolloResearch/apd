import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.utils import init_param_


def initialize_embeds(
    W_E: nn.Linear,
    W_U: nn.Linear,
    n_inputs: int,
    d_embed: int,
    superposition: bool,
    torch_gen: torch.Generator | None = None,
):
    if torch_gen is None:
        torch_gen = torch.Generator()
    assert W_E.weight.shape == (d_embed, n_inputs), f"Shape of W_E: {W_E.weight.shape}"
    W_E.weight.data[:, :] = torch.zeros(d_embed, n_inputs)
    W_E.weight.data[0, 0] = 1.0
    num_functions = n_inputs - 1
    d_control = d_embed - 2

    if not superposition:
        W_E.weight.data[1:-1, 1:] = torch.eye(num_functions)
    else:
        random_matrix = torch.randn(d_control, num_functions, generator=torch_gen)
        random_normalised = random_matrix / torch.norm(random_matrix, dim=1, keepdim=True)
        W_E.weight.data[1:-1, 1:] = random_normalised

    W_U.weight.data = torch.zeros(1, d_embed)  # Assuming n_outputs is always 1
    W_U.weight.data[:, -1] = 1.0


class ParamComponents(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        init_scale: float,
        resid_component: nn.Parameter | None,
        resid_dim: int | None,
    ):
        """
        Args:
            in_dim: Input dimension of the parameter to be replaced with AB.
            out_dim: Output dimension of the parameter to be replaced with AB.
            k: Number of subnetworks.
            resid_component: Predefined component matrix of shape (d_embed, k) if A or (k, d_embed)
                if B.
            resid_dim: Dimension in which to use the predefined component.
        """
        super().__init__()

        if resid_component is not None:
            if resid_dim == 0:
                a = resid_component
                b = nn.Parameter(torch.empty(k, out_dim))
            elif resid_dim == 1:
                a = nn.Parameter(torch.empty(in_dim, k))
                b = resid_component
            else:
                raise ValueError("Invalid resid_dim value. Must be 0 or 1.")
        else:
            a = nn.Parameter(torch.empty(in_dim, k))
            b = nn.Parameter(torch.empty(k, out_dim))

        self.A = a
        self.B = b
        init_param_(self.A, init_scale)
        init_param_(self.B, init_scale)

    def forward(
        self,
        x: Float[Tensor, "batch dim1"],
        topk_mask: Bool[Tensor, "batch k"] | None = None,
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k"]]:
        inner_acts = einops.einsum(x, self.A, "batch dim1, dim1 k -> batch k")
        if topk_mask is not None:
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch k, batch k -> batch k")
        out = einops.einsum(inner_acts, self.B, "batch k, k dim2 -> batch dim2")
        return out, inner_acts


class ParamComponentsFullRank(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        bias: bool,
        init_scale: float,
    ):
        super().__init__()

        self.subnetwork_params = nn.Parameter(torch.empty(k, in_dim, out_dim))
        init_param_(self.subnetwork_params, init_scale)

        self.bias = nn.Parameter(torch.zeros(k, out_dim)) if bias else None

    def forward(
        self, x: Float[Tensor, "batch dim1"], topk_mask: Bool[Tensor, "batch k"] | None = None
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k dim2"]]:
        inner_acts = einops.einsum(
            x, self.subnetwork_params, "batch dim1, k dim1 dim2 -> batch k dim2"
        )
        if self.bias is not None:
            inner_acts += self.bias

        if topk_mask is not None:
            inner_acts = einops.einsum(
                inner_acts, topk_mask, "batch k dim2, batch k -> batch k dim2"
            )
        out = einops.einsum(inner_acts, "batch k dim2 -> batch dim2")
        return out, inner_acts


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: str = "relu"):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.input_layer = nn.Linear(d_model, d_mlp)
        self.output_layer = nn.Linear(d_mlp, d_model)
        if act_fn == "relu":
            self.act_fn = torch.nn.functional.relu
        elif act_fn == "gelu":
            self.act_fn = torch.nn.functional.gelu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> tuple[
        Float[Tensor, "... d_model"],
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"] | None],
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"]],
    ]:
        """Run a forward pass and cache pre and post activations for each parameter.

        Note that we don't need to cache pre activations for the biases. We also don't care about
        the output bias which is always zero.
        """
        out1_pre_act_fn = self.input_layer(x)
        out1 = self.act_fn(out1_pre_act_fn)
        out2 = self.output_layer(out1)

        pre_acts = {
            "input_layer.weight": x,
            "input_layer.bias": None,
            "output_layer.weight": out1,
        }
        post_acts = {
            "input_layer.weight": out1_pre_act_fn,
            "input_layer.bias": out1_pre_act_fn,
            "output_layer.weight": out2,
        }
        return out2, pre_acts, post_acts


class MLPComponents(nn.Module):
    """A module that contains two linear layers with a ReLU activation in between for full rank SPD.

    A bias gets added to the first layer but not the second. The bias does not have a subnetwork
    dimension in this rank 1 case.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        init_scale: float,
        input_bias: Float[Tensor, " d_mlp"] | None = None,
        input_component: nn.Parameter | None = None,
        output_component: nn.Parameter | None = None,
    ):
        super().__init__()

        self.linear1 = ParamComponents(
            in_dim=d_embed,
            out_dim=d_mlp,
            k=k,
            resid_component=input_component,
            resid_dim=0,
            init_scale=init_scale,
        )
        self.linear2 = ParamComponents(
            in_dim=d_mlp,
            out_dim=d_embed,
            k=k,
            resid_component=output_component,
            resid_dim=1,
            init_scale=init_scale,
        )

        self.bias1 = nn.Parameter(torch.zeros(d_mlp))
        if input_bias is not None:
            self.bias1.data[:] = input_bias.detach().clone()

    def forward(
        self, x: Float[Tensor, "... d_embed"], topk_mask: Bool[Tensor, "... k"] | None = None
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Note that "inner_acts" represents the activations after multiplcation by A in the rank 1
        case, and after multiplication by subnetwork_params (but before summing over k) in the
        full-rank case.

        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x, topk_mask)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)
        x, inner_acts_linear2 = self.linear2(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x, layer_acts, inner_acts


class MLPComponentsFullRank(nn.Module):
    """A module that contains two linear layers with a ReLU activation in between for full rank SPD.

    The biases are (optionally) part of the "linear" layers, and have a subnetwork dimension in this
    full rank case.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        init_scale: float,
        in_bias: bool = True,
        out_bias: bool = False,
    ):
        super().__init__()
        self.linear1 = ParamComponentsFullRank(
            in_dim=d_embed, out_dim=d_mlp, k=k, bias=in_bias, init_scale=init_scale
        )
        self.linear2 = ParamComponentsFullRank(
            in_dim=d_mlp, out_dim=d_embed, k=k, bias=out_bias, init_scale=init_scale
        )

    def forward(
        self, x: Float[Tensor, "... d_embed"], topk_mask: Bool[Tensor, "... k"] | None = None
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations at the output of each layer after summing over the
                subnetwork dimension.
            inner_acts: The activations at the output of each subnetwork before summing.
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x, topk_mask)
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)
        x, inner_acts_linear2 = self.linear2(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x, layer_acts, inner_acts
