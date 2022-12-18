import torch


class MultiLayerPerceptron(torch.nn.Module):
    """Fully-connected deep neural network."""

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
