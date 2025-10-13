"""
Implementation of Fourier Neural Operator.

See:
    [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.

Adapted from https://github.com/neuraloperator/.
"""
from typing import Mapping

import mlx
import torch
from torch import nn

SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
        layer_config['name'] = f'operatorlearning.modules.fno.FourierLayer'
        conv_config = dict(layer_config['conv_config'])
        conv_config['in_channels'] = d_model
        conv_config['out_channels'] = d_model
        layer_config['conv_config'] = conv_config
        self.layers = torch.nn.Sequential(*[
            mlx.create_module(layer_config)
            for _ in range(num_fourier_layers - 1)
        ])

        layer_config['is_last'] = True  # The last layer is treated separately
        self.layers.append(mlx.create_module(layer_config))

        lift_config = dict(lift_config)
        self.lift = mlx.create_module(lift_config)

        project_config = dict(project_config)
        self.project = mlx.create_module(project_config)

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


def _contract_dense(x, weight):
    # batch-size, in_channels, x, y...
    x_syms = list(SYMBOLS[:x.ndim])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    weight_syms.insert(1, SYMBOLS[x.ndim])  # outputs
    out_syms = list(weight_syms)
    out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    return torch.einsum(eq, x, weight)


class SpectralConv(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            max_n_modes=None,
            bias=True,
            init_std="auto",
    ):
        """SpectralConv implements the Spectral Convolution component of a Fourier layer.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param n_modes: int or int tuple; number of modes to use for contraction in
            Fourier domain during training. Provided modes should be even integers.
            Odd numbers will be rounded to the closest even number. This can be
            updated dynamically during training.
        :param max_n_modes: int tuple or None, default is None
            * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
                The number of modes (`n_modes`) cannot be increased beyond that.
            * If None, all the n_modes are used.
        :param init_std: float or 'auto', default is 'auto'; std to use for the init
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std

        weight_shape = (in_channels, out_channels, *max_n_modes)
        weight = torch.empty(weight_shape, dtype=torch.cfloat)
        weight.normal_(0, init_std)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)

        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
            self, x: torch.Tensor
    ):
        """Generic forward pass for the Spectral Conv

        """
        batch_size, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm='forward', dim=fft_dims)
        # When x is real in spatial domain, the last half of the last dim is redundant.
        # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
        dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        out_fft = torch.zeros([batch_size, self.out_channels, *fft_size],
                              device=x.device, dtype=torch.cfloat)

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in
                  zip(fft_size, self.n_modes, self.max_n_modes)]

        slices_w = [slice(None), slice(None)]  # in_channels, out_channels

        # The last mode already has redundant half removed in real FFT
        slices_w += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        weight = self.weight[slices_w]

        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weight.shape[2:])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        out_fft[slices_x] = _contract_dense(x[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm='forward')

        if self.bias is not None:
            x = x + self.bias

        return x


class ChannelMLP(torch.nn.Module):
    def __init__(self, mlp_config):
        """MLP applied along channel dimension"""
        super(ChannelMLP, self).__init__()
        self.mlp = mlx.modules.MLP(**mlp_config)

    def forward(self, x):
        """
        :param x: (B, d_in, *shape) input
        :return: (B, d_out, *shape) MLP applied to x along dimension 1 (channel
            dimension)
        """
        result_t = self.mlp(torch.transpose(x, 1, -1))
        return torch.transpose(result_t, 1, -1)


class FourierLayer(torch.nn.Module):
    def __init__(
            self,
            conv_config: Mapping,
            nonlinearity: Mapping,
            mlp_config: Mapping | None = None,
            use_preactivation: bool = False,
            norm_config: Mapping | None = None,
            is_last=False,
    ):
        super(FourierLayer, self).__init__()

        self.conv = SpectralConv(**conv_config)

        # Skip connection is just a linear layer applied along channels,
        # which we implement with a channel-wise MLP with no hidden layers
        self.skip = ChannelMLP(dict(
            d_in=conv_config['in_channels'],
            d_out=conv_config['out_channels'],
            hidden_layers=[],
            activation={'name': 'Identity'},
            bias=False
        ))

        self.use_mlp = (mlp_config is not None)
        self.use_preactivation = use_preactivation

        self._nonlinearity = None
        self._norm = None
        self._mlp = None
        self._mlp_skip = None
        self._mlp_norm = None

        if not is_last:
            self._nonlinearity = mlx.create_module(nonlinearity)

        if self.use_mlp:
            self._mlp_skip = ChannelMLP(dict(
                d_in=conv_config['in_channels'],
                d_out=conv_config['out_channels'],
                hidden_layers=[],
                activation={'name': 'Identity'},
                bias=False
            ))

            mlp_config = dict(mlp_config)
            mlp_config['d_in'] = mlp_config['d_out'] = conv_config['out_channels']
            self._mlp = ChannelMLP(mlp_config)

        if norm_config is not None:
            self._norm = mlx.create_module(norm_config)
            if self.use_mlp:
                self._mlp_norm = mlx.create_module(norm_config)

    def forward(self, x):
        if self.use_preactivation:
            return self._forward_preactivation(x)
        else:
            return self._forward_normal(x)

    def nonlinearity(self, x):
        if self._nonlinearity is None:
            return x
        else:
            return self._nonlinearity(x)

    def norm(self, x):
        if self._norm is None:
            return x
        else:
            return self._norm(x)

    def mlp_skip(self, x):
        if self._mlp_skip is not None:
            return self._mlp_skip(x)

    def mlp_norm(self, x):
        if self._mlp_norm is None:
            return x
        else:
            return self._mlp_norm(x)

    def mlp(self, x):
        if self._mlp is None:
            return x
        else:
            # Make sure to move the channel dimension to the end
            # and then back to the channel position to apply MLP channel-wise
            return self._mlp(x)

    def _forward_normal(self, x):
        mlp_skip = self.mlp_skip(x)

        x = self.nonlinearity(self.norm(self.conv(x)) + self.skip(x))

        if self.use_mlp:
            x = self.nonlinearity(self.mlp_norm(self.mlp(x) + mlp_skip))

        return x

    def _forward_preactivation(self, x):
        x = self.norm(self.nonlinearity(x))
        mlp_skip = self.mlp_skip(x)

        x = self.nonlinearity(self.conv(x) + self.skip(x))

        if self.use_mlp:
            x = self.mlp(self.mlp_norm(x)) + mlp_skip

        return x
