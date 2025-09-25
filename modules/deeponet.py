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
            mlp_config,
            feat_expansion=None
    ):
        """
        A vanilla MLP trunk network for DeepONet

        :param num_branches: Number p of branch networks in the DeepONet
        :param v_d_in: Input dimension of the output function v
        :param v_d_out: Output dimension of the output function v
        :param mlp_config: Config for the underlying MLP module. d_in and
            d_out will be overridden (and name will be added automatically)
        :param feat_expansion: Optional feature expansion module config.
            Should map shape (B, *out_shape, v_d_in) to
            shape (B, *out_shape, feat_expansion.num_features)
        """
        super(MLPTrunkNet, self).__init__()

        if feat_expansion is not None:
            self.feat_expansion = modules.create_module(feat_expansion)
        else:
            self.feat_expansion = None

        mlp_config['name'] = 'MLP'
        mlp_config['d_in'] = v_d_in if self.feat_expansion is None else self.feat_expansion.num_features
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
        if self.feat_expansion is not None:
            y = self.feat_expansion(y)
        trunk = self.mlp(y)  # (B, *out_shape, p*v_d_out)
        return trunk.reshape(*y.shape[:-1], self.num_branches, self.v_d_out)


class FourierFeatureExpansion(torch.nn.Module):
    def __init__(self, origin, scale, features: int, mode: str = 'full', learnable=False):
        """
        :param origin: Origin of Fourier features, list of coordinates
            [x_1, ..., x_d]
        :param scale: Scale of the Fourier features in each direction, list of
            coordinates [s_1, ..., s_d]
        :param features: Number of Fourier features to use. Interpretation
            depends on value of mode
        :param mode: Either 'full' or 'random'. If 'full', then every mode
            corresponding to wave numbers [g_1, ..., g_d] with
            -features <= g_i <= features will be returned. If 'random' is given,
            then features random wave numbers will be generated following a
            normal distribution with mean [0, ..., 0] and standard deviations
            [3, ..., 3] and uncorrelated components
        :param learnable: If True, then the wave numbers will be learnable
            parameters (regardless of what value of mode is given)
        """
        super().__init__()
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float))

        self.dim = len(self.origin)
        self.mode = mode

        if mode == 'full':
            g_range = torch.arange(-features, features + 1)
            g = torch.stack(torch.meshgrid([g_range] * self.dim, indexing='ij'), dim=-1)
            # (2*features + 1, ..., 2*features + 1, dim)
            # |---------- x self.dim -------------|

            k = ((2 * torch.pi / self.scale) * g).reshape(-1, self.dim)
            # (num_features, dim)
        elif mode == 'random':
            g = 3 * torch.randn(features, self.dim)
            k = (2 * torch.pi / self.scale) * g
        else:
            raise ValueError('Invalid mode')

        if learnable:
            self.k = torch.nn.Parameter(k)
        else:
            self.register_buffer('k', k)

        self.num_features = 2 * self.k.shape[0]

    def forward(self, y):
        """
        :param y: (*out_shape, dim) input coordinates
        :return: (*out_shape, num_features) output features
        """
        angle = torch.einsum('...d,nd->...n', y, self.k)
        # (*out_shape, num_features/2)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        return torch.cat([sin, cos], dim=-1)  # (*out_shape, num_features)
