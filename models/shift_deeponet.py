import torch.nn

import models
import modules.shift_deeponet


class ShiftDeepONet(torch.nn.Module):
    def __init__(
            self,
            deeponet_config,
            shift_net_config
    ):
        """
        This is a light wrapper of DeepONet that simply attaches a ShiftTrunkNet
        built on the main trunk net.

        :param deeponet_config: Config for constructing the underlying DeepONet
            model to be modified with Shift-DeepONet addition
        :param shift_net_config: Config for constructing the shift network.
            Should map (B, *out_shape, v_d_in) -> (B, *out_shape, v_d_in^2 + v_d_in),
            where the last dimension packs both the linear transformation matrix
            and the translation vector.
        """
        super().__init__()
        self.deeponet = models.create_model(deeponet_config)
        self.deeponet.trunk_net = modules.shift_deeponet.ShiftTrunkNet(
            self.deeponet.trunk_net,
            shift_net_config
        )

    def forward(self, u, x_out):
        """
        :param u: (B, *in_shape, u_d_out) sample values of a batch of input
            functions
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the output function.
        :return: (B, *out_shape, v_d_out)
        """
        return self.deeponet(u, x_out)
