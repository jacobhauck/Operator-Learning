import itertools
from abc import ABC, abstractmethod

import torch


class Interpolator(ABC):
    @abstractmethod
    def interpolate(self, function: 'Function', x):
        """
        Interpolates a function to a given set of points
        :param function: The Function object to interpolate
        :param x: (..., d_in) the points at which to interpolate the function
        :return: The interpolated values of the function at the given points
        """
        raise NotImplemented


class Function(torch.nn.Module):
    def __init__(self, y, x, interpolator: Interpolator):
        """
        Create a sampled function object
        :param y: (..., d_out) or (...) the sampled function values
        :param x: (..., d_in) the sampling points
        :param interpolator: interpolator to use for evaluation
        """
        super(Function, self).__init__()

        self.x = x
        self.y = y
        self.is_scalar = (len(self.y.shape) == len(self.x.shape) - 1)

        assert self.y.shape[:len(self.x.shape) - 1] == self.x.shape[:-1], \
            ('Sample values and sample points must have same shape in all but '
             'last dimension')

        self.interpolator = interpolator

    @property
    def d_in(self):
        """Dimension of domain"""
        return self.x.shape[-1]

    @property
    def d_out(self):
        """Dimension of codomain"""
        return self.y.shape[-1] if not self.is_scalar else 1

    def __call__(self, x_query):
        """
        Evaluate the function at given points.
        :param x_query: (..., d_in) The points at which to evaluate the function
            (may require interpolation using the current interpolator)
        :return: (..., d_out) or (...) The function value at the given points
        """

        assert x_query.shape[-1] == self.d_in, 'Invalid input dimension'

        return self.interpolator.interpolate(self, x_query)


def _grid_interpolate(function: 'GridFunction', x, method, extend):
    """
    Interpolate a GridFunction to a new set of points
    :param function: The GridFunction to interpolate
    :param x: (..., d_in) the set of points on which to interpolate the function
    :param method: Method to use for interpolation. One of 'linear' or 'nearest'.
    :param extend: How to extend the function when interpolation points are
        outside the grid. One of 'clamped' to clamp to the nearest boundary
        value, 'periodic' to extend periodically (this is done in a closed-open
        manner, so the function values at the upper limits will be ignored),
        or a given constant value to pad with.
    """
    assert isinstance(function, GridFunction), \
        '`GridInterpolator` only works for `GridFunction`s'

    assert function.d_in == x.shape[-1], 'Invalid input dimension'

    query_shape = x.shape[:-1]
    x = x.reshape(-1, function.d_in)  # (N, d_in)

    min_tensor = function.x_min[None]  # (1, d_in)
    max_tensor = function.x_max[None]  # (1, d_in)

    extend_constant = False
    if extend == 'clamped':
        x = torch.clamp(x, min_tensor, max_tensor)
    elif extend == 'periodic':
        x = (x - min_tensor) % (max_tensor - min_tensor) + min_tensor
    else:
        extend_constant = True

    ts = []
    left_indices = []
    is_valid = []
    for i in range(function.d_in):
        left_indices_i = torch.searchsorted(function.xs[i], x[:, i].contiguous(), side='right') - 1
        left_indices_i[x[:, i] == function.xs[i][-1]] -= 1

        if extend == 'periodic':
            left_indices_i[left_indices_i == (len(function.xs[i]) - 1)] = -1
            deltas = function.xs[i][left_indices_i + 1] - function.xs[i][left_indices_i]
            ts_i = (x[:, i] - function.xs[i][left_indices_i]) / deltas
        else:
            # noinspection PyTypeChecker
            valid = torch.bitwise_and(x[:, i] >= min_tensor[:, i], x[:, i] <= max_tensor[:, i])
            is_valid.append(valid)

            internal = torch.bitwise_and(
                left_indices_i >= 0,
                left_indices_i < (len(function.xs[i]) - 1)
            )

            left_indices_int = left_indices_i[internal]
            deltas_int = function.xs[i][left_indices_int + 1] - function.xs[i][left_indices_int]

            ts_i = torch.zeros_like(x[:, i])
            ts_i[internal] = (x[internal, i] - function.xs[i][left_indices_int]) / deltas_int
            ts_i[left_indices_i == -1] = 1

        left_indices.append(left_indices_i)
        ts.append(ts_i)

    is_oob = None
    if extend_constant:
        is_oob = torch.bitwise_not(torch.stack(is_valid)).sum(dim=0).to(bool)

    if method == 'linear':
        result = 0
        for exponent in itertools.product(*([(0, 1)] * function.d_in)):
            c = 1
            for i in range(function.d_in):
                c = ((1 - ts[i]) if exponent[i] == 0 else ts[i]) * c
            # c has shape (N)

            index = [
                torch.minimum(left_indices[i] + exponent[i], torch.tensor(len(function.xs[i]) - 1))
                for i in range(function.d_in)
            ]

            if extend_constant:
                for i in range(function.d_in):
                    index[i][is_oob] = 0

            if not function.is_scalar:
                index = index + [slice(None)]
            val = function.y[*index]  # (N) or (N, d_out)

            if extend_constant:
                val[is_oob] = extend

            if function.is_scalar:
                result = c * val + result
            else:
                result = c[:, None] * val + result

        if function.is_scalar:
            return result.reshape(query_shape)
        else:
            return result.reshape(*query_shape, -1)

    elif method == 'nearest':
        shift = [ts[i] > 0.5 for i in range(function.d_in)]
        index = [left_indices[i] + shift[i] for i in range(function.d_in)]

        if extend_constant:
            for i in range(function.d_in):
                index[i][is_oob] = 0

        if not function.is_scalar:
            index = index + [slice(None)]
        result = function.y[*index]  # (N) or (N, d_out)

        if extend_constant:
            result[is_oob] = extend

        if function.is_scalar:
            return result.reshape(query_shape)
        else:
            return result.reshape(*query_shape, -1)


class GridInterpolator(Interpolator):
    def __init__(self, method='linear', extend='clamped'):
        """
        Interpolator for functions defined on a tensor grid.
        :param method: Interpolation method to use. One of 'nearest' or 'linear'
        :param extend: How to handle samples that are out of bounds. One of
            'clamped' to clamp to nearest boundary value, 'periodic' to extend
            periodically, or a constant value, for constant extension.
            Default: 'clamped'.
        """
        self.method = method
        self.extend = extend

    def interpolate(self, function: 'GridFunction', x):
        return _grid_interpolate(function, x, self.method, self.extend)


class OracleInterpolator(Interpolator):
    def __init__(self, oracle):
        """
        Interpolate a function by using a known oracle for it.
        """
        self.oracle = oracle

    def interpolate(self, function: 'Function', x):
        """
        Compute the value of the function using the oracle.
        :param function: The function; this argument is actually ignored,
            and the stored oracle is used for interpolation; be careful not to
            use this as an interpolator unless you are sure that the function
            it is attached to is actually the same as the oracle!
        :param x: The values at which to evaluate the function.
        """
        return self.oracle(x)


class GridFunction(Function):
    def __init__(
            self,
            y, x=None,
            interpolator: Interpolator | None = None,
            xs=None,
            is_sorted=False,
            x_min=None, x_max=None
    ):
        """
        Create a tensor function with samples at the tensor product of the
        given sampling points.
        :param y: (..., d_out) or (...) the sample values
        :param x: (x_1, ..., x_{d_in}, d_in) tensor of coordinate values. Can
            be constructed from xs if not given (one or both of x or xs *must*
            be given; do not give invalid combinations). Coordinates must be
            sorted.
        :param interpolator: interpolates the function. Defaults to
            GridInterpolator
        :param xs: List [x_1, ..., x_{d_in}] of sampling coordinates in each
            dimension. These will be sorted for normalization (which simplifies
            interpolation, for example). Can be inferred from x if not given.
        :param is_sorted: If xs is given, whether it is already sorted (do not
            lie about this!)
        :param x_min: Indicates the minimum point of the rectangular domain.
            Taken to be the minimum of the given sampling points if not given.
        :param x_max: Indicates the maximum point of the rectangular domain.
            Taken to be the maximum of the given sampling points if not given.
        """

        if x is not None:
            if xs is None:
                self.xs = []
                extraction_tuple = [0] * len(x.shape)
                for i in range(len(x.shape) - 1):
                    extraction_tuple[i-1] = 0
                    # noinspection PyTypeChecker
                    extraction_tuple[i] = slice(None)
                    extraction_tuple[-1] = i
                    self.xs.append(x[*extraction_tuple])
            else:
                self.xs = xs

        elif xs is not None:
            if is_sorted:
                self.xs = xs
            else:
                self.xs = [torch.sort(x_i)[0] for x_i in xs]

            x = torch.stack(torch.meshgrid(self.xs, indexing='ij'), dim=-1)
        else:
            raise ValueError('Must provide one or both of x and xs')


        if interpolator is None:
            interpolator = GridInterpolator()

        if x_min is None:
            x_min = torch.tensor(
                [self.xs[i][0] for i in range(len(self.xs))],
                dtype=x.dtype, device=x.device
            )
        self.x_min = x_min

        if x_max is None:
            x_max = torch.tensor(
                [self.xs[i][-1] for i in range(len(self.xs))],
                dtype=x.dtype, device=x.device
            )
        self.x_max = x_max

        super(GridFunction, self).__init__(y, x, interpolator)

    @staticmethod
    def from_oracle(f, x=None, interpolator: Interpolator | None = None, xs=None, is_sorted=False):
        """
        Constructs a GridFunction from an oracle
        :param f: The oracle, maps (..., d_in) to (..., d_out)
        :param x: (..., d_in), optional grid sampling points
        :param interpolator: optional interpolator; if not given, an
            OracleInterpolator is used with the given oracle
        :param xs: list of length d_in of sampling points in each direction. At
            least one of x or xs must be provided
        :param is_sorted: Whether the sampling points are given in sorted order
            (if xs is provided); points will be sorted for interpolation
            efficiency if False is provided. Default: False.
        """
        if x is None:
            if is_sorted:
                xs = xs
            else:
                xs = [torch.sort(x_i)[0] for x_i in xs]


            x = torch.stack(torch.meshgrid(xs, indexing='ij'), dim=-1)
        else:
            raise ValueError('Must provide one or both of x and xs')

        y = f(x)

        if interpolator is None:
            interpolator = OracleInterpolator(f)

        return GridFunction(y, x=x, interpolator=interpolator, xs=xs, is_sorted=True)

    @staticmethod
    def uniform_xs(min_point, max_point, num):
        """
        Constructs the grid points for a uniform sampling grid in each direction.
        :param min_point: (d_in) Minimum coordinate values of input domain
        :param max_point: (d_in) Maximum coordinate values of input domain
        :param num: (d) or int, number of sampling points in each direction,
            optionally one number to be used for all directions.
        """
        assert len(min_point) == len(max_point), \
            'Min and max points must have same dimension'

        try:
            num = int(num)
            num = num * torch.ones_like(min_point)
        except ValueError:
            assert len(num) == len(min_point), \
                'Numbers of sampling points must have same dimension as min and max points'

            num = num.to(int)

        xs = []
        for min_i, max_i, num_i in zip(min_point, max_point, num):
            xs.append(torch.linspace(min_i, max_i, num_i + 1))

        return xs

    @staticmethod
    def uniform_from_oracle(f, min_point, max_point, num, interpolator=None):
        """
        Create a GridFunction on a uniform grid from an oracle.
        :param f: The oracle, maps (..., d_in) to (..., d_out)
        :param min_point: (d_in) The minimum coordinate values
        :param max_point: (d_in) The maximum coordinate values
        :param num: (d_in) or int. The number of samples in each
            direction, optionally one number to be used for all directions
        :param interpolator: optional interpolator; if not given, an
            OracleInterpolator is used with the given oracle
        """
        return GridFunction.from_oracle(
            f,
            xs=GridFunction.uniform_xs(min_point, max_point, num),
            is_sorted=True,
            interpolator=interpolator
        )
