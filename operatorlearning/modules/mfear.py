"""
Implementation of our multifidelity encode-approximate-reconstruct model (MFEAR)
"""
import torch
import mlx


class TrapezoidIntegrator(torch.nn.Module):
    """
    Integrates using the composite trapezoid rule.

    Underlying sampling points must lie on a tensor grid. Minimal
    and maximal points are used as bounds of integration domain.
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, f, x):
        """
        :param f: (B, *in_shape, d_out) function sample values
        :param x: (B, *in_shape, d_in) Points at which function is sampled. Must
            form a tensor grid (according to the conventions of GridFunction)
        :return: (B, d_out) integral of f over the domain using the composite
            trapezoid rule
        """
        d_in = x.shape[-1]
        assert len(x.shape) == 2 + d_in, 'x has wrong shape for tensor grid'

        xs = []
        extraction_tuple = [slice(None)] + [0] * d_in + [0]
        for i in range(d_in):
            if i > 0:
                extraction_tuple[i] = 0
            extraction_tuple[i + 1] = slice(None)
            extraction_tuple[-1] = i
            xs.append(x[*extraction_tuple].contiguous())
        # list of d_in tensors of shape (B, in_shape[i])

        w = torch.ones_like(x[..., 0:1])  # (B, *in_shape, 1)
        for i, x_i in enumerate(xs):
            w_i = torch.zeros_like(x_i)  # (B, in_shape[i])
            w_i[:, 1:-1] = (x_i[:, 2:] - x_i[:, :-2]) / 2
            w_i[:, 0] = (x_i[:, 1] - x_i[:, 0]) / 2
            w_i[:, -1] = (x_i[:, -1] - x_i[:, -2]) / 2

            broadcast_index = [slice(None)] + [None] * i + [slice(None)] + [None] * (d_in - i)
            w *= w_i[*broadcast_index]

        b, d_out = f.shape[0], f.shape[-1]
        return torch.sum((w * f).view(b, -1, d_out), dim=1)  # (B, d_out)


class CompactMLPBasis(torch.nn.Module):
    def __init__(self, mlp, p, d_in, d_out):
        """
        :param mlp: Config of underlying MLP
        :param p: Number of basis functions
        :param d_in: Input dimension
        :param d_out: Output dimension
        """
        super().__init__()
        self.p = p
        self.d_in = d_in
        self.d_out = d_out

        mlp = dict(mlp)
        mlp['name'] = 'MLP'
        mlp['d_in'] = d_in
        mlp['d_out'] = d_out * p
        self.mlp = mlx.create_module(mlp)

    def forward(self, x):
        """
        :param x: (B, *in_shape, d_in) Points at which to compute the basis
        :return: (b, *in_shape, p, d_out) Basis functions evaluated at x
        """
        packed = self.mlp(x)  # (B, *in_shape, p*d_out)
        return packed.reshape(*packed.shape[:-1], self.p, self.d_out)


class MFEAR(torch.nn.Module):
    def __init__(
            self,
            integrator,
            encoder_net,
            approximator_net,
            reconstructor_net
    ):
        """
        :param integrator: Encoder integration module; maps (f, x)
            to integral of f, where f is shape (B, *in_shape, d_out), x is
            shape (B, *in_shape, d_in), and resulting integral is shape
            (B, d_out)
        :param encoder_net: Config for encoder network;
            maps (B, *in_shape, u_d_in) -> (B, *in_shape, p, u_d_out)
        :param approximator_net: Config for approximator network;
            maps (B, p) -> (B, q)
        :param reconstructor_net: Config for reconstructor network;
            maps (B, *out_shape, v_d_in) -> (B, *out_shape, q, v_d_out)
        """
        super().__init__()
        self.integrator = mlx.create_module(integrator)
        self.encoder_net = mlx.create_module(encoder_net)
        self.approximator_net = mlx.create_module(approximator_net)
        self.reconstructor_net = mlx.create_module(reconstructor_net)

    def forward(self, u, x_in, x_out):
        """
        :param u: (B, *in_shape, u_d_out) Input function samples
        :param x_in: (B, *in_shape, u_d_in) Coordinates of input function sample
            points
        :param x_out: (B, *out_shape, v_d_in) Coordinates at which to sample output
            function
        :return: (B, *out_shape, v_d_out) Output function sampled at the points
            given by x_out
        """
        encoder_basis = self.encoder_net(x_in)  # (B, *in_shape, p, u_d_out)
        prod = (u[..., None, :] * encoder_basis).sum(dim=-1)
        # (B, *in_shape, p)

        z = self.integrator(prod, x_in)  # (B, p)
        w = self.approximator_net(z)  # (B, q)

        reconstruct_basis = self.reconstructor_net(x_out)
        # (B, *out_shape, q, v_d_out)

        v = torch.einsum('bq,b...qd->b...d', w, reconstruct_basis)
        # (B, *out_shape, v_d_out)

        return v
