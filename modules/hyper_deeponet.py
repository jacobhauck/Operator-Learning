import torch
import modules
import utils
from typing import Sequence, Mapping


def prod(nums):
    p = 1
    for n in nums:
        p *= n
    return p


class LinearNoWeight(torch.nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        weight_shape = (d_out, d_in)
        if bias:
            bias_shape = (d_out,)
            self.hyperparams = {'weight': weight_shape, 'bias': bias_shape}
        else:
            self.hyperparams = {'weight': weight_shape}

    def forward(self, x, weight, bias=None):
        """
        :param x: (B, *shape, d_in) input tensor
        :param weight: (B, d_out, d_in) weight tensor
        :param bias: (B, *shape, d_out) optional bias tensor
        """
        linear = torch.einsum('b...i,boi->b...o', x, weight)
        if 'bias' in self.hyperparams:
            assert bias is not None
            broadcast_shape = (bias.shape[0],) + (1,) * (len(x.shape) - 2) + (bias.shape[1],)
            return linear + bias.reshape(broadcast_shape)
        
        return linear


class MainMLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            hidden_layers: Sequence[int],
            d_out: int,
            activation: Mapping,
            bias: bool = True
    ):
        """
        A multi-layer perceptron main network.
        :param d_in: input dimension
        :param hidden_layers: sequence of integers giving hidden dimensions;
            number of layers = len(hidden_layers) + 1
        :parma d_out: output dimension
        :param activation: Activation function config
        :param bias: Whether to use bias in the linear layers. Default=True.
        """
        super(MainMLP, self).__init__()

        # Save parameters
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation

        # Construct layers
        layers = []

        architecture = (d_in,) + self.hidden_layers + (d_out,)
        for d_in, d_out in zip(architecture[:-1], architecture[1:]):
            layers.append(LinearNoWeight(d_in, d_out, bias=bias))
            layers.append(utils.create_activation(activation))
        layers.pop()  # Remove the last activation function

        # Save layers as ModuleList
        self.layers = torch.nn.ModuleList(*layers)

        # Logic for routing hyperparameters
        self.offsets = [0]
        for layer in self.param_layers:
            for param_shape in layer.hyperparams.values():
                self.offsets.append(self.offsets[-1] + prod(param_shape))

    @property
    def param_layers(self):
        for layer in self.layers:
            try:
                _ = layer.hyperparams
                yield layer
            except AttributeError:
                pass

    @property
    def num_params(self):
        return self.offsets[-1]

    def forward(self, x, params):
        """
        :param x: (B, *shape, d_in) input value
        :param params: (B, self.num_params) network parameters
        :return: (B, *shape, d_out) result of applying the MLP with the given parameters
        """
        p_index = 0
        for layer in self.layers:
            layer_params = {}
            try:
                for p_name, p_shape in layer.hyperparams.items():
                    flat = params[:, self.offsets[p_index] : self.offsets[p_index + 1]]
                    layer_params[p_name] = flat.reshape((x.shape[0],) + p_shape)
                    p_index += 1
            except AttributeError:
                pass

            x = self.layer(x, **layer_params)
        
        return x


class MainMLPTrunk(torch.nn.Module):
    def __init__(self, v_d_in, v_d_out, mlp_config):
        """
        :param v_d_in: Dimension of output function domain
        :param v_d_out: Number of output function variables
        :param mlp_config: Config of underlying MLP (no need to provide d_in/d_out)
        """
        mlp_config['d_in'] = v_d_in
        mlp_config['d_out'] = v_d_out
        self.main_mlp = MainMLP(**mlp_config)
    
    def forward(self, x, params):
        """
        :param x: (B, *out_shape, v_d_in) Coordinates of points at which to evaluate
            the output function
        :param params: (B, num_params) Parameters of the main network
        :return: (B, *out_shape, v_d_out)
        """
        self.main_mlp(x, params)


class HypernetworkMLP(torch.nn.Module):
    def __init__(self, num_sensors, num_params, mlp_config):
        """
        MLP hypernetwork
        :param num_sensors: Number of sensors for the input
        :param num_params: Number of hyperparameters to pass to the main network.
            Note that num_params is passed automatically by HyperDeepONet.
        :param mlp_config: Config for the underlying MLP; d_in and d_out are
            inferred automatically from num_sensors and num_params.
        """
        super().__init__()
        mlp_config['name'] = 'MLP'
        mlp_config['d_in'] = num_sensors
        mlp_config['d_out'] = num_params
        self.mlp = modules.create_module(mlp_config)
    
    def forward(self, u):
        """
        :param u: (B, *in_shape, u_d_out) sample values of input function
        :return: (B, num_params) parameters to pass to the main network
        """
        return self.mlp(u.reshape(u.shape[0], -1))
