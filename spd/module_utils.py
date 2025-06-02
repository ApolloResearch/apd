import math
from functools import reduce
from typing import Any

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn.init import calculate_gain


def get_nested_module_attr(module: nn.Module, access_string: str) -> Any:
    """Access a specific attribute by its full, path-like name.

    Taken from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8

    Args:
        module: The module to search through.
        access_string: The full name of the nested attribute to access, with each object separated
            by periods (e.g. "linear1.A").
    """
    names = access_string.split(".")
    try:
        mod = reduce(getattr, names, module)
    except AttributeError as err:
        raise AttributeError(f"{module} does not have nested attribute {access_string}") from err
    return mod


@torch.inference_mode()
def remove_grad_parallel_to_subnetwork_vecs(
    A: Float[Tensor, "... d_in m"], A_grad: Float[Tensor, "... d_in m"]
) -> None:
    """Modify the gradient by subtracting it's component parallel to the activation.

    I.e. subtract the projection of the gradient vector onto the activation vector.

    This is to stop Adam from changing the norm of A. Note that this will not completely prevent
    Adam from changing the norm due to Adam's (m/(sqrt(v) + eps)) term not preserving the norm
    direction.
    """
    parallel_component = einops.einsum(A_grad, A, "... d_in m, ... d_in m -> ... m")
    A_grad -= einops.einsum(parallel_component, A, "... m, ... d_in m -> ... d_in m")


def init_param_(
    param: torch.Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: str = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Fill in param with values sampled from a Kaiming normal distribution.

    Args:
        param: The parameter to initialize
        fan_val: The squared denominator of the std used for the kaiming normal distribution
        mean: The mean of the normal distribution
        nonlinearity: The nonlinearity of the activation function
        generator: The generator to sample from
    """
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)
