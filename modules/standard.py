import torch
from typing import Sequence, Mapping
import utils


class MLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            hidden_layers: Sequence[int],
            d_out: int,
            activation: Mapping,
            bias: bool = True
    ):
        """
        A multi-layer perceptron.
        :param d_in: input dimension
        :param hidden_layers: sequence of integers giving hidden dimensions;
            number of layers = len(hidden_layers) + 1
        :parma d_out: output dimension
        :param activation: Activation function config
        :param bias: Whether to use bias in the linear layers. Default=True.
        """
        super(MLP, self).__init__()

        # Save parameters
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation

        # Construct layers
        layers = []

        architecture = (d_in,) + self.hidden_layers + (d_out,)
        for d_in, d_out in zip(architecture[:-1], architecture[1:]):
            layers.append(torch.nn.Linear(d_in, d_out, bias=bias))
            layers.append(utils.create_activation(activation))
        layers.pop()  # Remove the last activation function

        # Save layers as Sequential module
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
