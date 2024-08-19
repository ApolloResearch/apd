# %%

from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# %%


class PiecewiseLinear(nn.Module):
    """
    this class is initialised with a function from real numbers to real numbers, a range
    (start,end)], and a num_neurons. It returns a neural network with one hidden layer of
    num_neurons relus with biases given by np.linspace(start, end, num_neurons) that computes a
    piecewise linear approximation to the function"""

    def __init__(self, f: Callable[[float], float], start: float, end: float, num_neurons: int):
        super().__init__()
        self.f = f
        self.start = start
        self.end = end
        self.num_neurons = num_neurons

        self.interval = (end - start) / (num_neurons - 1)
        self.input_layer = nn.Linear(1, num_neurons, bias=True)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.num_neurons, 1, bias=True)

        self.initialise_params()

    def initialise_params(self):
        biases = -np.linspace(self.start, self.end, self.num_neurons) + self.interval
        assert (
            len(biases) == self.num_neurons
        ), f"len(biases) = {len(biases)}, num_neurons = {self.num_neurons}, biases = {biases}"

        self.input_layer.bias.data = torch.tensor(biases, dtype=torch.float32)
        # -torch.tensor(
        #     np.linspace(start-self.interval, end, num_neurons), dtype=torch.float32
        # )[:-1] + self.interval
        # print("neuron bias", self.neurons.bias.data)
        self.input_layer.weight.data = torch.ones(self.num_neurons, 1, dtype=torch.float32)

        self.output_layer.bias.data = torch.tensor(0, dtype=torch.float32)

        xs = np.linspace(self.start, self.end, self.num_neurons)
        self.function_values = torch.tensor([self.f(x) for x in xs], dtype=torch.float32)
        self.function_values = torch.cat(
            [torch.tensor([0], dtype=torch.float32), self.function_values]
        )
        slopes = (self.function_values[1:] - self.function_values[:-1]) / self.interval
        slope_diffs = torch.zeros(self.num_neurons, dtype=torch.float32)
        slope_diffs[0] = slopes[0]
        slope_diffs[1:] = slopes[1:] - slopes[:-1]
        self.output_layer.weight.data = slope_diffs.view(-1, 1).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

    def plot(self, ax: plt.Axes, start: float, end: float, num_points: int):
        x = np.linspace(start, end, num_points)
        y = np.array([self.f(x) for x in x])
        ax.plot(x, y, label="f(x)")
        # print("input shape", torch.tensor(x, dtype=torch.float32).unsqueeze(1).shape)
        ax.plot(
            x,
            self.forward(torch.tensor(x, dtype=torch.float32).unsqueeze(1)).detach().numpy(),
            label="NN(x)",
        )
        ax.legend()
        ax.set_title("Piecewise Linear Approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # add vertical lines to show start and end
        ax.axvline(x=self.start, color="r", linestyle="--")
        ax.axvline(x=self.end, color="r", linestyle="--")



# %%
class ControlledPiecewiseLinear(nn.Module):
    """
    Takes in a list of num_functions functions, and a range (start, end), and a num_neurons. It
    creates a neural network which takes in x and a list of num_functions control bits. It outputs
    num_functions different real numbers. If the i-th control bit is 1, the i-th output is the
    output of the piecewise linear approximation to the i-th function. Otherwise, the i-th output is
    0. The piecewise linear approximation is done with num_neurons neurons.
    """

    def __init__(
        self,
        functions: list[Callable[[float], float]],
        start: float,
        end: float,
        num_neurons: int,
        d_control: int,
        control_W_E: torch.Tensor | None = None,
        negative_suppression: int = 100,
    ):
        super().__init__()
        self.functions = functions
        self.num_functions = len(functions)
        self.start = start
        self.end = end
        self.num_neurons = num_neurons
        self.negative_suppression = negative_suppression
        self.d_control = d_control
        self.control_W_E = control_W_E
        self.input_layer = nn.Linear(
            self.d_control + 1, self.num_functions * self.num_neurons, bias=True
        )
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(
            self.num_functions * self.num_neurons, self.num_functions, bias=True
        )
        self.initialise_params()

    def initialise_params(self):
        self.control_W_E = (
            torch.eye(self.num_functions) if self.control_W_E is None else self.control_W_E
        )
        assert (
            self.control_W_E.shape[0] <= self.num_functions
        ), "control_W_E should have at most num_functions rows"
        assert self.d_control == self.control_W_E.shape[1], "control_W_E should have d_control cols"
        self.piecewise_linears = [
            PiecewiseLinear(f, self.start, self.end, self.num_neurons) for f in self.functions
        ]
        # initialise all weights and biases to 0
        self.input_layer.weight.data = torch.zeros(
            self.num_functions * self.num_neurons, self.d_control + 1
        )
        for i in range(self.num_functions):
            piecewise_linear = self.piecewise_linears[i]
            self.input_layer.bias.data[
                i * self.num_neurons : (i + 1) * self.num_neurons
            ] = -self.negative_suppression

            self.input_layer.weight.data[
                i * self.num_neurons : (i + 1) * self.num_neurons, 0
            ] = piecewise_linear.input_layer.weight.data.squeeze()
            self.input_layer.weight.data[i * self.num_neurons : (i + 1) * self.num_neurons, 1:] += (
                self.control_W_E[i]
                * (self.negative_suppression + piecewise_linear.input_layer.bias.data.unsqueeze(1))
            )

        self.output_layer.weight.data = torch.zeros(
            self.num_functions, self.num_functions * self.num_neurons
        )
        for i in range(self.num_functions):
            piecewise_linear = self.piecewise_linears[i]
            self.output_layer.bias.data[i] = piecewise_linear.output_layer.bias.data
            self.output_layer.weight.data[
                i, i * self.num_neurons : (i + 1) * self.num_neurons
            ] = piecewise_linear.output_layer.weight.data.squeeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        control_bits = x[:, 1:]
        input_value = x[:, 0].unsqueeze(1)
        assert control_bits.shape[1] == self.num_functions
        assert torch.all((control_bits == 0) | (control_bits == 1))
        control_vectors = control_bits @ self.control_W_E
        if control_vectors.dim() == 1:
            control_vectors = control_vectors.unsqueeze(0).repeat(len(x), 1)
        control_vectors = control_vectors.to(torch.float32)
        x = torch.cat([input_value, control_vectors], dim=1)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

    def plot(
        self, start: float, end: float, num_points: int, control_bits: torch.Tensor | None = None
    ):
        # make a figure with self.num_functions subplots
        fig, axs = plt.subplots(self.num_functions, 1, figsize=(10, 5 * self.num_functions))
        assert isinstance(axs, Iterable)
        x = np.linspace(start, end, num_points)
        if control_bits is None:
            control_bits = torch.zeros(len(x), self.num_functions)
        if control_bits.dim() == 1:
            control_bits = control_bits.unsqueeze(0).repeat(len(x), 1)
        assert control_bits.shape[1] == self.num_functions
        assert control_bits.shape[0] == len(x)
        input = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        input_with_control = torch.cat([input, control_bits], dim=1)
        print(input_with_control.shape)
        outputs = self.forward(input_with_control).detach().numpy()
        for i in range(self.num_functions):
            target = np.array([self.functions[i](x) for x in x])
            axs[i].plot(x, target, label="f(x)")
            # print("input shape", torch.tensor(x, dtype=torch.float32).unsqueeze(1).shape)
            axs[i].plot(x, outputs[:, i], label="NN(x)")
            axs[i].legend()
            axs[i].set_title(f"Piecewise Linear Approximation of function {i}")
            axs[i].set_xlabel("x")
            axs[i].set_ylabel("y")
            # add vertical lines to show start and end
            axs[i].axvline(x=self.start, color="r", linestyle="--")
            axs[i].axvline(x=self.end, color="r", linestyle="--")
        plt.show()

# %%
class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, initialise_zero=True):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.input_layer = nn.Linear(d_model, d_mlp)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(d_mlp, d_model)
        if initialise_zero:
            self.initialise_zero()

    def initialise_zero(self):
        self.input_layer.weight.data = torch.zeros(self.d_mlp, self.d_model)
        self.input_layer.bias.data = torch.zeros(self.d_mlp)
        self.output_layer.weight.data = torch.zeros(self.d_model, self.d_mlp)
        self.output_layer.bias.data = torch.zeros(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class ControlledResNet(nn.Module):
    """
    Same inputs as ControlledPiecewiseLinear, but also takes in an input num_layers. Now it creates
    a network which has the same input and output as ControlledPiecewiseLinear, but the inputs and
    outputs lie in orthogonal parts of a single residual stream, and the neurons are randomly
    distributed across the layers.
    """

    def __init__(
        self,
        functions: list[Callable[[float], float]],
        start: float,
        end: float,
        neurons_per_function: int,
        num_layers: int,
        d_control: int,
        negative_suppression: int = 100,
    ):
        super().__init__()
        self.functions = functions
        self.d_control = d_control
        self.num_functions = len(functions)
        self.start = start
        self.end = end
        self.neurons_per_function = neurons_per_function
        self.num_layers = num_layers
        self.total_neurons = neurons_per_function * self.num_functions
        assert self.total_neurons % num_layers == 0, "num_neurons must be divisible by num layers"
        self.negative_suppression = negative_suppression

        self.d_mlp = self.total_neurons // num_layers
        # d_model: one for x, one for each control bit, and one for y (the output of the controlled
        # piecewise linear)
        self.d_model = self.d_control + 2
        self.mlps = nn.ModuleList([MLP(self.d_model, self.d_mlp) for _ in range(num_layers)])
        self.initialise_params()

    def initialise_params(self):
        if self.d_control == self.num_functions:
            print("control_W_E is identity")
            self.control_W_E = torch.eye(self.d_control)
        else:
            random_matrix = torch.randn(self.d_control, self.num_functions)
            # normalise rows
            self.control_W_E = random_matrix / random_matrix.norm(dim=1).unsqueeze(1)

        self.controlled_piecewise_linear = ControlledPiecewiseLinear(
            self.functions,
            self.start,
            self.end,
            self.neurons_per_function,
            self.d_control,
            self.control_W_E,
            self.negative_suppression,
        )

        # create a random permutation of the neurons
        self.neuron_permutation = torch.randperm(self.total_neurons)
        # split the neurons into num_layers parts
        self.neuron_permutations = torch.split(self.neuron_permutation, self.d_mlp)
        # create num_layers residual layers

        output_weights_summed = self.controlled_piecewise_linear.output_layer.weight.data.sum(dim=0)

        # set the weights of the residual layers to be the weights of the corresponding neurons in
        # the controlled piecewise linear
        for i in range(self.num_layers):
            self.mlps[i].input_layer.weight.data[
                :, :-1
            ] = self.controlled_piecewise_linear.input_layer.weight.data[
                self.neuron_permutations[i]
            ]
            self.mlps[
                i
            ].input_layer.bias.data = self.controlled_piecewise_linear.input_layer.bias.data[
                self.neuron_permutations[i]
            ]
            # print(self.residual_layers[i].output_layer.weight.data.shape)
            # print(self.neuron_permutations[i].shape)
            # print(output_weights_summed[self.neuron_permutations[i]].shape)
            self.mlps[i].output_layer.weight.data[-1] = output_weights_summed[
                self.neuron_permutations[i]
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concatenate x and control bits
        control_bits = x[:, 1:]
        input_value = x[:, 0].unsqueeze(1)

        # if control_bits is None:
        #     control_bits = torch.zeros(x.shape[0], self.num_functions)
        assert (
            control_bits.shape[1] == self.num_functions
        ), "control bits should have num_functions columns"
        assert torch.all((control_bits == 0) | (control_bits == 1)), "control bits should be 0 or 1"
        # map control bits to control vectors
        control_vectors = control_bits @ self.control_W_E
        x = torch.cat([input_value, control_vectors], dim=1)

        # add a zero for the output of the controlled piecewise linear
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)

        assert x.shape[1] == self.d_model
        for i in range(self.num_layers):
            x = x + self.mlps[i](x)
        return x

    def partial_forward(
        self, x: torch.Tensor, layer: int | None = None, control_bits: torch.Tensor | None = None
    ) -> torch.Tensor:
        # return the output of the network up to layer
        if control_bits is None:
            control_bits = torch.zeros(x.shape[0], self.num_functions)
        assert control_bits.shape[1] == self.num_functions
        assert control_bits.shape[0] == x.shape[0]
        if layer is None:
            layer = self.num_layers
        x = torch.cat([x, control_bits], dim=1)
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        for i in range(layer):
            x = x + self.mlps[i](x)
        return x

    def plot(
        self,
        start: float,
        end: float,
        num_points: int,
        layers: int | list[int] | None = None,
        control_bits: torch.Tensor | None = None,
    ):
        # make a figure
        fig, ax = plt.subplots(figsize=(10, 10))
        x = np.linspace(start, end, num_points)
        if control_bits is None:
            control_bits = torch.zeros(self.num_functions)
        assert control_bits.shape == (
            self.num_functions,
        ), "control_bits should be a 1D tensor for the plot function"
        input_control_bits = control_bits.unsqueeze(0).repeat(len(x), 1)

        target = np.zeros((self.num_functions, len(x)))
        for i in range(self.num_functions):
            target[i] = np.array([self.functions[i](x) for x in x])
        target = torch.einsum("fb,f -> b", torch.tensor(target, dtype=torch.float32), control_bits)
        ax.plot(x, target, label="f(x)", linewidth=8)

        if layers is None:
            layers = list(range(self.num_layers + 1))
        elif isinstance(layers, int):
            layers = [layers]
        for layer in layers:
            outputs = (
                self.partial_forward(
                    torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                    layer=layer,
                    control_bits=input_control_bits,
                )
                .detach()
                .numpy()
            )
            ax.plot(x, outputs[:, -1], label=f"layer {layer} NN(x)")

        # print("input shape", torch.tensor(x, dtype=torch.float32).unsqueeze(1).shape)
        ax.legend()
        # ax.set_title(
        #     "Piecewise Linear Approximation of functions "
        #     f"with control bits {control_bits.detach().numpy().tolist()}"
        # )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # add vertical lines to show start and end
        ax.axvline(x=self.start, color="r", linestyle="--")
        ax.axvline(x=self.end, color="r", linestyle="--")
        # set ylim between min and max value of target
        ax.set_ylim([target.min().item() - 2, target.max().item() + 2])  # type: ignore
        plt.show()
