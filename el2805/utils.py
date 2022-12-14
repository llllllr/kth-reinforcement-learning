import numpy as np
import torch
from typing import NamedTuple


class Experience(NamedTuple):
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool


class NeuralNetwork(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            n_hidden_layers: int,
            hidden_layer_size: int,
            activation: str
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation

        self._hidden_layers = []
        input_size = self.input_size
        for _ in range(n_hidden_layers):
            self._hidden_layers.append(torch.nn.Linear(input_size, self.hidden_layer_size))
            input_size = hidden_layer_size
        self._output_layer = torch.nn.Linear(input_size, output_size)

        if self.activation == "relu":
            self._activation_fn = torch.nn.functional.relu
        elif self.activation == "tanh":
            self._activation_fn = torch.nn.functional.tanh
        else:
            raise NotImplementedError

    def forward(self, x):
        for layer in self._hidden_layers:
            x = layer(x)
            x = self._activation_fn(x)
        x = self._output_layer(x)
        return x


def random_decide(rng, probability):
    return rng.binomial(n=1, p=probability) == 1


def running_average(data, window_length):
    if len(data) >= window_length:
        averages = np.copy(data)
        averages[window_length - 1:] = np.convolve(data, np.ones((window_length,)) / window_length, mode='valid')
    else:
        averages = np.zeros_like(data)
    return averages
