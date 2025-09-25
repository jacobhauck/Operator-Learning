"""
Implementation of DeepONet.

See:

"""
import torch

import modules


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
        self.branch_net = modules.create_module(branch_config)
        self.trunk_net = modules.create_module(trunk_config)
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

    def forward(self, u, x_out):
        """
        :param u: (B, *in_shape, u_d_out) sample values of a batch of input
            functions
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the output function.
        :return: (B, *out_shape, v_d_out)
        """
        branch_vals = self.branch_net(u)  # (B, p)
        trunk_vals = self.trunk_net(x_out)  # (B, *out_shape, p, v_d_out)

        pre_scaled = torch.einsum('bp,b...po->b...o', branch_vals, trunk_vals)
        if self.scale is None:
            scaled = pre_scaled
        elif self.scale == 'linear':
            scaled = pre_scaled / branch_vals.shape[1]
        elif self.scale == 'sqrt':
            scaled = pre_scaled / (branch_vals.shape[1] ** .5)
        else:
            scaled = pre_scaled * self.scale

        if self.bias is None:
            return scaled
        elif self._bias_is_module:
            # noinspection PyCallingNonCallable
            return scaled + self.bias(x_out)
        else:
            if self._need_init:
                self.bias = torch.nn.Parameter(torch.empty(size=trunk_vals.shape[-1]))
                self._init_bias()
                self._need_init = False

            return scaled + self.bias.view(*([1] * (len(scaled.shape)-1)), -1)
