"""
Implementation of HyperDeepONet
"""
import torch
import modules


class HyperDeepONet(torch.nn.Module):
    def __init__(self, main_network_config, hypernetwork_config):
        super().__init__()
        self.main_network = modules.create_module(main_network_config)
        hypernetwork_config['num_params'] = self.main_network.num_params
        self.hypernetwork = modules.create_module(hypernetwork_config)
    
    def forward(self, u, x_out):
        """
        :param u: (B, *in_shape, u_d_out) input function samples
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which to sample
            the output function
        :return: (B, *out_shape, v_d_out) output function samples
        """
        params = self.hypernetwork(u)
        return self.main_network(x_out, params)
