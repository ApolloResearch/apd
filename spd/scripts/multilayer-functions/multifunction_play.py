from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from piecewise_linear import ControlledPiecewiseLinear, ControlledResNet


# test
# make a list of 50 different cubic functions
def generate_cubics(num_cubics: int):
    def create_cubic(a, b, c, d):
        return lambda x: a * x**3 + b * x**2 + c * x + d

    cubics = []
    for _ in range(num_cubics):
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-4, 4)
        d = np.random.uniform(-8, 8)
        cubics.append(create_cubic(a, b, c, d))
    return cubics


def generate_trig_functions(num_trig_functions: int):
    def create_trig_function(a, b, c, d, e, f, g) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: a * torch.sin(b * x + c) + d * torch.cos(e * x + f) + g

    trig_functions = []
    for _ in range(num_trig_functions):
        a = np.random.uniform(-1, 1)
        b = np.exp(np.random.uniform(-1, 3))
        c = np.random.uniform(-np.pi, np.pi)
        d = np.random.uniform(-1, 1)
        e = np.exp(np.random.uniform(-1, 3))
        f = np.random.uniform(-np.pi, np.pi)
        g = np.random.uniform(-1, 1)
        trig_functions.append(create_trig_function(a, b, c, d, e, f, g))
    return trig_functions


def generate_regular_simplex(num_vertices):
    # Create the standard basis in num_vertices dimensions
    basis = torch.eye(num_vertices)

    # Create the (1,1,...,1) vector
    ones = torch.ones(num_vertices)

    # Compute the Householder transformation
    v = ones / torch.norm(ones)
    last_basis_vector = torch.zeros(num_vertices)
    last_basis_vector[-1] = 1
    u = v - last_basis_vector
    u = u / torch.norm(u)

    # Apply the Householder transformation
    H = torch.eye(num_vertices) - 2 * u.outer(u)
    rotated_basis = basis @ H

    # Remove the last coordinate
    simplex = rotated_basis[:, :-1]

    # Center the simplex at the origin
    centroid = simplex.mean(dim=0)
    simplex = simplex - centroid

    return simplex / simplex.norm(dim=1).unsqueeze(1)


num_functions = 50
dim = 49

trigs = generate_trig_functions(num_functions)
if num_functions == dim:
    control_W_E = torch.eye(num_functions)
elif num_functions == dim + 1:
    control_W_E = generate_regular_simplex(num_functions)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)
else:
    control_W_E = torch.randn(num_functions, dim)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)
test = ControlledPiecewiseLinear(trigs, 0, 5, 32, dim, control_W_E, negative_suppression=6)

if num_functions == dim:
    control_bits = torch.ones(num_functions, dtype=torch.float32)
else:
    control_bits = torch.zeros(num_functions, dtype=torch.float32)
    control_bits[4] = 1
    control_bits[9] = 1
    control_bits[17] = 1

test.plot(-0.1, 5.1, 1000, control_bits=control_bits)

num_functions = 50
dim = 50

trigs = generate_trig_functions(num_functions)
if num_functions == dim:
    control_W_E = torch.eye(num_functions)
elif num_functions == dim + 1:
    control_W_E = generate_regular_simplex(num_functions)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)
else:
    control_W_E = torch.randn(num_functions, dim)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)
# test = ControlledPiecewiseLinear(trigs, 0, 5, 32, control_W_E, negative_suppression=6)
test = ControlledResNet(trigs, 0, 5, 40, 5, dim, negative_suppression=100)

if num_functions == dim:
    control_bits = torch.ones(num_functions, dtype=torch.float32)
else:
    control_bits = torch.zeros(num_functions, dtype=torch.float32)
    control_bits[0] = 1
    control_bits[1] = 1
# control_bits = torch.zeros(num_functions, dtype=torch.float32)
# control_bits[torch.tensor([4, 9, 12])] = 1
# test.controlled_piecewise_linear.plot(-0.1, 5.1, 1000, control_bits=control_bits)

test.plot(-0.1, 5.1, 1000, control_bits=control_bits)
# test = ControlledResNet(
#     [lambda x: x**2 - 0.1 * x**4, lambda x: 10 * np.sin(5 * x)], 0, 5, 40, 5
# )

# test.plot(-2, 6, 1000, control_bits=torch.tensor([0, 0], dtype=torch.float32))


# %%
a = torch.randn((50, 20))
# normalise rows of a
a = a / a.norm(dim=1).unsqueeze(1)
plt.imshow(a @ a.T)
plt.colorbar()
