import torch


class FCNetwork(torch.nn.Module):
    """Fully-connected deep neural network."""

    def __init__(
            self,
            *,
            input_size: int,
            n_hidden_layers: int,
            hidden_layer_size: int,
            activation: str,
            output_size: int | None = None,
            include_top: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.output_size = output_size
        self.include_top = include_top

        # Hidden layers
        self._hidden_layers = torch.nn.ModuleList()
        input_size = self.input_size
        for _ in range(n_hidden_layers):
            self._hidden_layers.append(torch.nn.Linear(input_size, self.hidden_layer_size))
            if activation == "relu":
                self._hidden_layers.append(torch.nn.ReLU())
            elif activation == "tanh":
                self._hidden_layers.append(torch.nn.Tanh())
            else:
                raise NotImplementedError
            input_size = hidden_layer_size

        # Output layer
        assert not (include_top and output_size is None)
        self._output_layer = torch.nn.Linear(input_size, self.output_size) if self.include_top else None

    def forward(self, x):
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
        if self.include_top:
            x = self._output_layer(x)
        return x
