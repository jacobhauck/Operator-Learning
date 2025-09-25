import torch

import modules


class ShiftTrunkNet(torch.nn.Module):
    def __init__(self, base_trunk_net, shift_net_config):
        """
        :param base_trunk_net: The base trunk net either as a Module or a
            config for a Module
        :param shift_net_config: Config for constructing the shift network.
            Should map (B, *out_shape, v_d_in) -> (B, *out_shape, v_d_in^2 + v_d_in),
            where the last dimension packs both the linear transformation matrix
            and the translation vector.
        """
        super().__init__()
        if isinstance(base_trunk_net, torch.nn.Module):
            self.base_trunk_net = base_trunk_net
        else:
            self.base_trunk_net = modules.create_module(base_trunk_net)

        self.shift_net = modules.create_module(shift_net_config)

    def forward(self, x_out):
        """
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the trunk network functions
        :return: (B, *out_shape, p, v_d_out) trunk net values at the given
            sample points
        """
        shift_params = self.shift_net(x_out)
        v_d_in = x_out.shape[-1]
        matrix_shape = x_out.shape[-1] + (v_d_in, v_d_in)
        matrix = shift_params[..., :-v_d_in].reshape(matrix_shape)
        # (B, *out_shape, v_d_in, v_d_in)

        shift = shift_params[..., -v_d_in:]  # (B, *out_shape, v_d_in)

        x_shift = torch.einsum('...d,...ed->...e', x_out, matrix) + shift

        return self.base_trunk_net(x_shift)
