from typing import Sequence, Mapping

import torch

import modules
import utils


def create_module(config: Mapping):
    """
    Create a model from its configuration.

    :param config: Dictionary of model configuration options. Must contain a
        'name' field to specify which model to load, which refers to the model's
        name within the `models` module.
    :return: An instance of the model specified by the given configuration
        settings.
    """
    config = dict(config)
    return getattr(modules, config.pop('name'))(**config)


class MLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            hidden_layers: Sequence[int],
            activation=(('name', 'ReLU'),),
            bias: bool | Sequence[bool] = True
    ):
        """
        :param d_in: Dimension of input
        :param d_out: Dimension of output
        :param hidden_layers: Sequence of hidden layer dimensions
        :param activation: List of activations or a single activation to apply
            to hidden layers. Should be given as configs that can be used with
            utils.activation.create_activation.
        :param bias: Whether to use bias in linear layers. Either a single bool
            to be applied to all hidden layers or a sequence to be applied to
            each layer in turn; should provide one extra value for whether bias
            should be used in the output linear layer.
        """
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)

        try:
            one_activation = dict(activation)
            self.activations = (one_activation,) * len(self.hidden_layers)
        except (ValueError, TypeError):
            self.activations = tuple(dict(a) for a in activation)

        try:
            self.bias = tuple(bias)
        except TypeError:
            self.bias = (bias,) * (len(self.hidden_layers) + 1)

        self.hidden_linear = torch.nn.ModuleList([
            torch.nn.Linear(in_feat, out_feat, bias=bias)
            for in_feat, out_feat, bias in
            zip((self.d_in,) + self.hidden_layers[:-1], self.hidden_layers, self.bias)
        ])
        self.activation_modules = torch.nn.ModuleList([
            utils.activation.create_activation(a) for a in self.activations
        ])
        last_hidden_dim = self.d_in if len(self.hidden_layers) == 0 else self.hidden_layers[-1]
        self.out_linear = torch.nn.Linear(last_hidden_dim, self.d_out, bias=self.bias[-1])

    def forward(self, x):
        """
        :param x: (*shape, d_in) input, shape is arbitrary
        :return: (*shape, d_out) output
        """
        for h, a in zip(self.hidden_linear, self.activation_modules):
            x = a(h(x))

        return self.out_linear(x)
