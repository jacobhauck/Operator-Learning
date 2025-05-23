import torch

import modules


class MLPBranchNet(torch.nn.Module):
    def __init__(
            self,
            num_sensors: int,
            num_branches: int,
            mlp_config
    ):
        """
        A vanilla MLP branch network for DeepONet

        :param num_sensors: Total number m of sensors (obtained via `x.numel()`,
            where `x` is the tensor of sampling points for the DeepONet.
        :param num_branches: Number p of branch networks
        :param mlp_config: Config for the underlying MLP module. d_in and
            d_out will be overridden (and name will be added automatically)
        """
        super(MLPBranchNet, self).__init__()

        mlp_config['name'] = 'MLP'
        mlp_config['d_in'] = num_sensors
        mlp_config['d_out'] = num_branches
        self.mlp = modules.create_module(mlp_config)

    def forward(self, u):
        """
        :param u: (B, *in_shape, u_d_out) input sensor values
        :return: (B, p) weights for trunk net functions
        """
        return self.mlp(u.view(u.shape[0], -1))


class MLPTrunkNet(torch.nn.Module):
    def __init__(
            self,
            num_branches: int,
            v_d_in: int,
            v_d_out: int,
            mlp_config
    ):
        """
        A vanilla MLP trunk network for DeepONet

        :param num_branches: Number p of branch networks in the DeepONet
        :param v_d_in: Input dimension of the output function v
        :param v_d_out: Output dimension of the output function v
        :param mlp_config: Config for the underlying MLP module. d_in and
            d_out will be overridden (and name will be added automatically)
        """
        super(MLPTrunkNet, self).__init__()

        mlp_config['name'] = 'MLP'
        mlp_config['d_in'] = v_d_in
        mlp_config['d_out'] = num_branches * v_d_out
        self.mlp = modules.create_module(mlp_config)

        self.num_branches = num_branches
        self.v_d_out = v_d_out

    def forward(self, y):
        """
        :param y: (B, *out_shape, v_d_in) coordinates at which to query the
            output function
        :return: (B, *out_shape, p, v_d_out) output values at the coordinates
        """
        trunk = self.mlp(y)  # (B, *out_shape, p*v_d_out)
        return trunk.reshape(*y.shape[:-1], self.num_branches, self.v_d_out)
