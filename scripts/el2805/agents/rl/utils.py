import numpy as np
import torch
from typing import NamedTuple


def get_epsilon(
        epsilon: float | str,
        *,
        episode: int | None = None,
        epsilon_max: float | None = None,
        epsilon_min: float | None = None,
        epsilon_decay_duration: int | None = None,
        delta: float | None = None
) -> float:
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = epsilon
    elif epsilon == "delta":
        assert delta is not None
        epsilon = 1 / (episode ** delta)
    elif epsilon == "linear":
        assert epsilon_max is not None and epsilon_min is not None and epsilon_decay_duration is not None
        epsilon = max(
            epsilon_min,
            epsilon_max - (epsilon_max - epsilon_min) * (episode - 1) / (epsilon_decay_duration - 1)
        )
    elif epsilon == "exponential":
        assert epsilon_max is not None and epsilon_min is not None and epsilon_decay_duration is not None
        epsilon = max(
            epsilon_min,
            epsilon_max * (epsilon_min / epsilon_max) ** ((episode - 1) / (epsilon_decay_duration - 1))
        )
    else:
        raise NotImplementedError
    return epsilon


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.has_mps:
    #     device = "mps"
    else:
        device = "cpu"
    return device


def normal_pdf(x, mean, var):
    pdf = 1 / torch.sqrt(2 * torch.pi * var) * torch.exp(-1/2 * (x - mean)**2 / var)
    return pdf


class Experience(NamedTuple):
    episode: int
    state: np.ndarray
    action: np.ndarray # action: int
    reward: float
    next_state: np.ndarray
    done: bool


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
            self,
            *,
            input_size: int,
            hidden_layer_sizes: list[int],
            hidden_layer_activation: str,
            output_size: int | None = None,
            output_layer_activation: str | None = None,
            include_top: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activation = hidden_layer_activation
        self.output_size = output_size
        self.output_layer_activation = output_layer_activation
        
        self.include_top = include_top

        # Hidden layers
        self._hidden_layers = []
        input_size = self.input_size
        for hidden_layer_size in self.hidden_layer_sizes:
            self._hidden_layers.append(torch.nn.Linear(input_size, hidden_layer_size))
            self._hidden_layers.append(torch.nn.BatchNorm1d(hidden_layer_size))
            if hidden_layer_activation == "relu":
                self._hidden_layers.append(torch.nn.ReLU())
            elif hidden_layer_activation == "tanh":
                self._hidden_layers.append(torch.nn.Tanh())
            else:
                raise NotImplementedError
            input_size = hidden_layer_size
        self._hidden_layers = torch.nn.Sequential(*self._hidden_layers)



        # Output layer
        assert not (include_top and output_size is None)
        if self.include_top:
            self._output_layer = torch.nn.Linear(input_size, self.output_size)

            if self.output_layer_activation == "sigmoid":
                output_activation = torch.nn.Sigmoid()
            elif self.output_layer_activation == "tanh":
                output_activation = torch.nn.Tanh()
            else:
                output_activation = None
                if self.output_layer_activation is not None:
                    raise NotImplementedError

            if output_activation is not None:
                self._output_layer = torch.nn.Sequential(self._output_layer, output_activation)
        else:
            self._output_layer = None

    def forward(self, x):
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
        if self.include_top:
            x = self._output_layer(x)
        return x
