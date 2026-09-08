import mlx
import torch
import neuralop.models
from typing import Mapping


class FNO(torch.nn.Module):
    def __init__(
            self,
            n_modes: tuple[int, ...],
            u_d_out: int,
            v_d_out: int,
            d_model: int,
            num_layers: int,
            activation: Mapping = (('name', 'GELU'),),
            **extra_args
    ):
        """
        Wrapper for neuraloperator FNO model
        :param n_modes: Number of modes to use in each direction
        :param u_d_out: Dimension of input function codomain
        :param v_d_out: Dimension of output function codomain
        :param d_model: Number of hidden channels in FNO model
        :param num_layers: Number of Fourier convolution layers
        :param activation: Pointwise nonlinearity config
        :param extra_args: Any other arguments; same as neuraloperator FNO
            implementation (see https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/fno.py)
            except for conv_module, which should be passed here as a config
        """
        super().__init__()

        if 'conv_module' in extra_args:
            extra_args['conv_module'] = mlx.create_module(extra_args['conv_module'])

        fno_args = dict(
            n_modes=n_modes,
            in_channels=u_d_out,
            out_channels=v_d_out,
            hidden_channels=d_model,
            n_layers=num_layers,
            non_linearity=mlx.create_module(activation),
            **extra_args
        )

        self._fno = neuralop.models.FNO(**fno_args)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if '_metadata' in state_dict:
            state_dict.pop('_metadata')
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, u):
        """
        :param u: (B, *shape, u_d_out) input function on a uniform grid
        :return: (B, *shape, v_d_out) output function on the same grid
        """
        u_ = torch.permute(u, (0, -1) + tuple(range(1, len(u.shape) - 1)))
        # (B, u_d_out, *shape)
        v_ = self._fno(u_)  # (B, v_d_out, *shape)
        return torch.permute(v_, (0,) + tuple(range(2, len(u.shape))) + (1,))
        # (B, *shape, v_d_out)
