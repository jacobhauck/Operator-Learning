"""
Implementation of DeepONet
"""
import mlx
import torch
import operatorlearning as ol


class DeepONetOutputFunction(ol.Function):
    @staticmethod
    def _evaluate(deeponet, x, branch_vals):
        trunk_vals = deeponet.trunk_net(x)
        pre_scaled = torch.einsum('p,...po->...o', branch_vals, trunk_vals)
        # (*out_shape, v_d_out)

        scaled = deeponet.scale_output(pre_scaled, branch_vals.shape[0])

        return deeponet.add_bias(scaled, x)

    def __init__(self, deeponet, x, branch_vals):
        self.deeponet = deeponet
        branch_vals = branch_vals.detach().clone()
        super().__init__(
            DeepONetOutputFunction._evaluate(deeponet, x, branch_vals),
            x,
            ol.OracleInterpolator(self.oracle)
        )
        self.register_buffer('branch_vals', branch_vals)

    def oracle(self, x):
        return DeepONetOutputFunction._evaluate(self.deeponet, x, self.branch_vals)


class DeepONet(torch.nn.Module):
    def __init__(self, branch_config, trunk_config, scale=None, bias=None):
        """
        Implementation of a DeepONet with unspecified branch and trunk
        architecture.

        :param branch_config: (B, *in_shape, u_d_out) -> (B, p) module implementing the
            branch network. u_d_out is the number of components of u. Provide
            config for construction via modules.create_module().
        :param trunk_config: (B, *out_shape, v_d_in) -> (B, *out_shape, p, v_d_out) module
            implementing the trunk network. v_d_in is the dimension of the domain
            of the output function v, and v_d_out is the number of components
            of v. Provide config for construction via modules.create_module().
        :param scale: Scaling factor to apply. None <=> scale==1, 'linear' <=>
            scale==1/p, 'sqrt' <=> scale==1/p^.5, or one can supply any
            constant numerical value.
        :param bias: (B, *out_shape, v_d_in) -> (1, *out_shape, v_d_out) module
            computing optional bias function, or True to indicate a constant,
            learnable bias (which is initialized just-in-time with iid standard
            normal distribution).
        """
        super(DeepONet, self).__init__()
        self.branch_net = mlx.create_module(branch_config)
        self.trunk_net = mlx.create_module(trunk_config)
        self.scale = scale
        self._bias_is_module = False
        self._need_init = False
        if bias is True:
            self.bias = None
            self._need_init = True
        elif bias is None:
            self.bias = None
        else:
            self._bias_is_module = True
            self.bias = bias

    def _init_bias(self):
        torch.nn.init.normal_(self.bias)

    def scale_output(self, pre_scaled, num_branches):
        """
        :param pre_scaled: Pre-scaled output (B, *out_shape, v_d_out)
        :param num_branches: Number of branch outputs
        """
        if self.scale is None:
            return pre_scaled
        elif self.scale == 'linear':
            return pre_scaled / num_branches
        elif self.scale == 'sqrt':
            return pre_scaled / (num_branches ** .5)
        else:
            return pre_scaled * self.scale

    def add_bias(self, scaled, x_out=None):
        """
        :param scaled: Pre-bias scaled output (B, *out_shape, v_d_out)
        :param x_out: Optional coordinates of output sampling points (only
            needed if using functional bias)
        """
        if self.bias is None:
            return scaled
        elif self._bias_is_module:
            assert x_out is not None

            # noinspection PyCallingNonCallable
            return scaled + self.bias(x_out)
        else:
            if self._need_init:
                self.bias = torch.nn.Parameter(torch.empty(size=scaled.shape[-1]))
                self._init_bias()
                self._need_init = False

            return scaled + self.bias.view(*([1] * (len(scaled.shape)-1)), -1)

    def forward(self, u, x_out, return_functions=False):
        """
        :param u: (B, *in_shape, u_d_out) sample values of a batch of input
            functions
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the output function.
        :param return_functions: Whether the output should be returned as
            a list of Functions. Note that this detaches branch values, so
            this should not be used for training.
        :return: (B, *out_shape, v_d_out)
        """
        branch_vals = self.branch_net(u)  # (B, p)

        if return_functions:
            return [
                DeepONetOutputFunction(self, x_out[b], branch_vals[b])
                for b in range(len(x_out))
            ]

        trunk_vals = self.trunk_net(x_out)  # (B, *out_shape, p, v_d_out)

        pre_scaled = torch.einsum('bp,b...po->b...o', branch_vals, trunk_vals)
        # (B, *out_shape, v_d_out)

        scaled = self.scale_output(pre_scaled, branch_vals.shape[1])

        return self.add_bias(scaled, x_out)


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
        self.mlp = mlx.create_module(mlp_config)

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
            self.feat_expansion = mlx.create_module(feat_expansion)
        else:
            self.feat_expansion = None

        mlp_config['name'] = 'MLP'
        mlp_config['d_in'] = v_d_in if self.feat_expansion is None else self.feat_expansion.num_features
        mlp_config['d_out'] = num_branches * v_d_out
        self.mlp = mlx.create_module(mlp_config)

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
