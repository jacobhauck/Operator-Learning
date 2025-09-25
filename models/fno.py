"""
Implementation of Fourier Neural Operator (FNO) in 1, 2 and 3 dimensions.
This implementation uses the improved architecture from [1].

See:
    [1] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
            Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
            TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.

"""
from typing import Mapping

import torch

import modules


class FNO(torch.nn.Module):
    def __init__(
            self,
            lift_config: Mapping,
            num_fourier_layers: int,
            d_model: int,
            fourier_layer_config: Mapping,
            project_config: Mapping,
    ):
        """
        A 1D FNO
        :param lift_config: Config for the lifting module, which should map
            (B, *shape, u_d_out) -> (B, *shape, d_latent), where d_latent
            is the model latent dimension provided in fourier_layer_config.
        :param num_fourier_layers: How many Fourier layers the model uses
        :param d_model: Output dimension of hidden functions
        :param fourier_layer_config: Config for FourierLayer, used for each
            layer
        :param project_config: Config for the projection module, which should
            map (B, *shape, d_latent) -> (B, *shape, v_d_out), where d_latent
            is the model latent dimension provided in fourier_layer_config.
        """
        super(FNO, self).__init__()

        layer_config = dict(fourier_layer_config)
        layer_config['name'] = f'fno.FourierLayer'
        conv_config = dict(layer_config['conv_config'])
        conv_config['in_channels'] = d_model
        conv_config['out_channels'] = d_model
        layer_config['conv_config'] = conv_config
        self.layers = torch.nn.Sequential(*[
            modules.create_module(layer_config)
            for _ in range(num_fourier_layers - 1)
        ])

        layer_config['is_last'] = True  # The last layer is treated separately
        self.layers.append(modules.create_module(layer_config))

        lift_config = dict(lift_config)
        self.lift = modules.create_module(lift_config)

        project_config = dict(project_config)
        self.project = modules.create_module(project_config)

    def forward(self, u):
        """
        :param u: (B, *shape, u_d_out) input function sampled uniformly on the
            domain
        :return: (B, *shape, v_d_out) output function sampled uniformly on the
            domain
        """

        v = self.lift(u).transpose(1, -1)  # (B, C, *shape)
        v = self.layers(v)  # (B, C, *shape)
        v = self.project(v.transpose(1, -1))  # (B, *shape, v_d_out)

        return v
