import torch
from math import floor


class SplineGridIntegrator(torch.nn.Module):
    """
    Integrates using spline approximation on a uniform tensor grid.

    Underlying sampling points must lie on a uniform tensor grid with
    endpoint convention matching this rule (open or closed).

    If using 'closed' mode, then this is a closed Newton-Cotes formula.

    If using 'open' mode with n = 0, then this is the one-point, open
    Newton-Cotes formula. For n > 0, it is a bit different from Newton-Cotes
    because the endpoints in this library's open grid convention are a distance
    h/2 from the first interior points, which are subsequently spaced by h. In
    standard open Newton-Cotes formulas, the first interior point and endpoint
    are separated by a distance h, the same as all subsequent points.
    """

    def __init__(self, n, mode='closed'):
        """
        :param n: Number of points minus one per spline
        :param mode: Either 'open' or 'closed'. 'closed' cannot be used with n = 0.
        """
        super().__init__()

        assert n >= 0, 'Must use at least one point per spline'
        assert n >= 1 or mode == 'open', 'closed spline requires at least 2 points'

        self.n = n
        self.mode = mode

    @staticmethod
    def coefficients(n, mode, device=None, dtype=None):
        """
        :return: (n + 1) tensor of quadrature weights up to factor of step size
        """
        kw = {'device': device, 'dtype': dtype}
        if mode == 'closed':
            if n == 1:  # Trapezoidal rule
                return torch.ones(2, **kw) / 2
            elif n == 2:  # Simpson's rule
                return torch.tensor([1.0, 4.0, 1.0], **kw) / 3.0
            elif n == 3:  # Simpson's 3/8 rule
                return torch.tensor([3.0, 9.0, 9.0, 3.0], **kw) / 8.0
            elif n == 4:  # Boole's rule
                return torch.tensor([14.0, 64.0, 24.0, 64.0, 14.0], **kw) / 45.0
            else:
                raise NotImplementedError('Closed rules not implemented for n >= 5')
        else:
            if n == 0:  # One-point, open Newton-Cotes formula
                return torch.ones(1, **kw)
            elif n == 1:
                return torch.ones(2, **kw)
            elif n == 2:
                return torch.tensor([9.0, 6.0, 9.0], **kw) / 8.0
            elif n == 3:
                return torch.tensor([13.0, 11.0, 11.0, 13.0], **kw) / 12.0
            elif n == 4:
                    return torch.tensor([1375.0, 500.0, 2010.0, 500.0, 1375.0], **kw) / 1152.0

    @staticmethod
    def closed_composite(base_weights, num_reps):
        """
        Generates composite weights by repeating the closed base rule.

        :param base_weights: (n + 1) Base weights for n + 1 points
        :param num_reps: Number of times to repeat the base rule
        :return: composite_weights (n * num_reps + 1) composite weights
        """
        inner = base_weights.clone()[:-1]
        # In closed mode, boundaries of the unit add, so add them before tiling
        inner[0] += base_weights[-1]
        # Tile the inner unit, adding one extra to allow for the last point
        comp = torch.tile(inner, (num_reps + 1,))  # (n * num_reps + n)
        # Cut off all but one point of the last unit
        target_length = len(inner) * num_reps + 1
        comp = comp[:target_length]  # (n * num_reps + 1)

        # Fix boundary points
        comp[0] = base_weights[0]
        comp[-1] = base_weights[-1]

        return comp

    @staticmethod
    def open_composite(base_weights, num_reps):
        """
        Generates composite weights by repeating the open base rule.

        :param base_weights: (n + 1) Base weights for n + 1 points
        :param num_reps: Number of times to repeat the base rule
        :return: composite_weights ((n + 1) * num_reps) composite weights
        """
        return torch.tile(base_weights, (num_reps + 1,))  # easy

    def num_reps(self, size):
        """
        Calculates number of repetitions of "unit interval" and number of
        leftover points to cover a grid with the given size.

        :param size: number of points
        :return: num_reps, leftover
        """
        if self.mode == 'closed':
            num_reps = floor((size - 1) / self.n)
            leftover = size - (num_reps * self.n + 1)
        elif self.mode == 'open':
            num_reps = floor(size / (self.n + 1))
            leftover = size - num_reps * (self.n + 1)
        else:
            raise ValueError('Invalid interval mode!')

        return num_reps, leftover

    def weights(self, x):
        d_in = x.shape[-1]
        assert len(x.shape) == 2 + d_in, 'x has wrong shape for tensor grid'

        in_shape = x.shape[1:-1]
        c = self.coefficients(self.n, self.mode, device=x.device, dtype=x.dtype)

        w = torch.ones_like(x[..., 0:1])  # (1, *in_shape, 1)
        for i in range(d_in):
            cur_i = (0,) * d_in
            next_i = [0] * d_in
            next_i[i] = 1
            dx_i = x[:, *tuple(next_i), i] - x[:, *cur_i, i]  # (B)

            w_i = torch.zeros((x.shape[0], in_shape[i]), device=x.device, dtype=x.dtype)
            num_reps, left_over = self.num_reps(in_shape[i])

            if self.mode == 'closed':
                comp = self.closed_composite(c, num_reps)
            elif self.mode == 'open':
                comp = self.open_composite(c, num_reps)
            else:
                raise ValueError('Invalid interval mode!')

            w_i[:, :len(comp)] = dx_i[:, None] * comp[None]
            # Use highest-order rule possible on the left-over points
            if left_over > 0:
                if self.mode == 'closed':
                    c_lo = self.coefficients(
                        left_over, self.mode,
                        device=x.device, dtype=x.dtype
                    )
                    # (left_over + 1)
                    w_i[:, len(comp) - 1:] += dx_i[:, None] * c_lo[None]
                elif self.mode == 'open':
                    c_lo = self.coefficients(
                        left_over - 1, self.mode,
                        device=x.device, dtype=x.dtype
                    )
                    # (left_over)
                    w_i[:, len(comp):] = dx_i[:, None] * c_lo[None]

            broadcast_index = [slice(None)] + [None] * i + [slice(None)] + [None] * (d_in - i)
            w *= w_i[*broadcast_index]

        return w

    def forward(self, f, x):
        """
        :param f: (B, *in_shape, d_out) function sample values
        :param x: (B, *in_shape, d_in) Points at which function is sampled. Must
            form a tensor grid (according to the conventions of GridFunction)
        :return: (B, d_out) integral of f over the domain using the composite
            Newton-Cotes rule
        """
        w = self.weights(x)
        b, d_out = f.shape[0], f.shape[-1]
        return torch.sum((w * f).view(b, -1, d_out), dim=1)  # (B, d_out)


class TrapezoidIntegrator(torch.nn.Module):
    """
    Integrates using the composite trapezoidal rule.

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
            trapezoidal rule
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
