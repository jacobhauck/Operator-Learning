"""
Implementation of Fourier Neural Operator (FNO) in 1, 2 and 3 dimensions.

See:

"""
from typing import Mapping

import torch

import modules


class FNO(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            lift_config: Mapping,
            num_fourier_layers: int,
            fourier_layer_config: Mapping,
            project_config: Mapping,
    ):
        """
        A 1D FNO
        :param d_in: The dimension of the domain for both the input function
            and the output function. Only 1, 2 and 3 are supported.
        :param lift_config: Config for the lifting module, which should map
            (B, *shape, u_d_out) -> (B, *shape, d_latent), where d_latent
            is the model latent dimension provided in fourier_layer_config.
        :param num_fourier_layers: How many Fourier layers the model uses
        :param fourier_layer_config: Config for FourierLayer1d, used for each
            layer
        :param project_config: Config for the projection module, which should
            map (B, *shape, d_latent) -> (B, *shape, v_d_out), where d_latent
            is the model latent dimension provided in fourier_layer_config.
        """
        super(FNO, self).__init__()

        assert d_in in (1, 2, 3)

        layer_config = dict(fourier_layer_config)
        layer_config['name'] = f'fno.FourierLayer{d_in}d'
        self.layers = torch.nn.Sequential(*[
            modules.create_module(layer_config)
            for _ in range(num_fourier_layers)
        ])

        lift_config = dict(lift_config)
        lift_config['latent_dim'] = layer_config['latent_dim']
        self.lift = modules.create_module(lift_config)

        project_config = dict(project_config)
        project_config['latent_dim'] = layer_config['latent_dim']
        self.project = modules.create_module(project_config)

    def forward(self, u):
        """
        :param u: (B, d1, u_d_out) input function sampled uniformly d1 times on
            the interval domain
        :return: (B, d1, v_d_out) output function sampled at the same points as
            the input function
        """

        v = self.lift(u)
        v = self.layers(v)
        v = self.project(v)

        return v
