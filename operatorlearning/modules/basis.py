import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Basis(ABC):
    @property
    @abstractmethod
    def dimension(self):
        raise NotImplemented

    @abstractmethod
    def coefficients(self, u, x):
        """
        Gets the coefficients of a given function in this basis
        :param u: (B, *shape, d_out) function sample values
        :param x: (B, *shape, d_in) sample points
        :return: (B, dimension) coefficients
        """
        raise NotImplemented

    def eval(self, x, coef):
        """
        Evaluates the function with the given coefficients in this basis
        :param x: (B, *shape, d_in) sample points
        :param coef: (B, dimension) coefficients of the function in the basis
        :return: (B, *shape, d_out)
        """
        basis = self.eval_basis(x)  # (B, dimension, *shape, d_out)
        return torch.einsum('Bd,Bd...D->B...D', coef, basis)

    @abstractmethod
    def eval_basis(self, x, i=None):
        """
        Evaluates the basis functions at the given points
        :param x: (B, *shape, d_in) sample points
        :param i: Optional tensor of indices of basis functions to evaluate; if
            not given, then i = range(self.dimension) is used.
        :return: (B, d, *shape, d_out) basis functions evaluated at the given
            points; d dimension indices are taken from i
        """
        raise NotImplemented


class OrthonormalBasis(Basis, ABC):
    @staticmethod
    def inner_product(u1, u2, x, integrator=None):
        """
        Computes the inner product between two functions.

        Default implementation uses L^2([0,1]^d_in) inner product
        :param u1: (B, *shape, d_out) First input function values
        :param u2: (B, *shape, d_out) Second input function values
        :param x: (B, *shape, d_in) points at which both functions are sampled
        :param integrator: optional module to use for integration
        :return: (B) inner product of u1 and u2
        """
        prod = torch.sum(u1 * u2, dim=-1, keepdim=True)  # (B, *shape, 1)
        if integrator is None:
            return torch.mean(prod.reshape(prod.shape[0], -1), dim=1)
        else:
            return integrator(prod, x)[:, 0]

    def coefficients(self, u, x, integrator=None):
        """
        Gets the coefficients of a given function in this basis
        :param u: (B, *shape, d_out) function sample values
        :param x: (B, *shape, d_in) sample points
        :param integrator: Optional integrator module to use for inner product
        :return: (B, dimension) coefficients
        """
        basis = self.eval_basis(x)  # (B, dimension, *shape, d_out)

        if integrator is None:
            c = self.inner_product(u, basis.reshape(-1, *basis.shape[2:]), x)
        else:
            c = self.inner_product(u, basis.reshape(-1, *basis.shape[2:]), x, integrator=integrator)
        # (B*dimension)

        return c.reshape(u.shape[0], -1)  # (B, dimension)


class FullFourierBasis2d(OrthonormalBasis, torch.nn.Module):
    def __init__(self, num_modes, x_min, x_max):
        """
        :param num_modes: Number of Fourier modes (of each type and in each
            direction, so total number of basis functions is roughly
            4*num_modes^2)
        :param x_min: (2) tuple giving lower point of rectangular domain
        :param x_max: (2) tuple giving upper point of rectangular domain
        """
        torch.nn.Module.__init__(self)
        self.num_modes = num_modes
        self.x_min = x_min
        self.x_max = x_max

        g_sin, g_cos = torch.arange(1, num_modes +  1), torch.arange(num_modes + 1)
        g_x, g_y = torch.meshgrid(g_sin, g_sin, indexing='ij')
        # (num_modes, num_modes)
        self.register_buffer('g_x', g_x.reshape(-1))  # (num_modes^2)
        self.register_buffer('g_y', g_y.reshape(-1))  # (num_modes^2)

        scale = self.volume ** .5

        # (num_modes^2)
        n_all = torch.full(self.g_x.shape, scale / 2, device=self.g_x.device)
        # (num_modes^2)
        n_cos = torch.cat([
            torch.tensor([scale], device=self.g_x.device),
            torch.full(g_sin.shape, scale/2**.5, device=self.g_x.device)
        ])
        # (num_modes + 1)
        n_sin = torch.full(g_sin.shape, scale/2**.5, device=self.g_x.device)
        # (num_modes)

        self.register_buffer(
            'akx',
            (2*torch.pi) * torch.cat([self.g_x, g_cos, torch.zeros_like(g_sin)])
        )
        # ((num_modes + 1)^2)
        self.register_buffer(
            'aky',
            (2*torch.pi) * torch.cat([self.g_y, torch.zeros_like(g_cos), g_sin])
        )
        # ((num_modes + 1)^2)
        self.register_buffer('a_n', torch.cat([n_all, n_cos, n_sin]))
        # ((num_modes + 1)^2)

        self.register_buffer(
            'bkx',
            (2*torch.pi) * torch.cat([self.g_x, torch.zeros_like(g_sin)])
        )
        # (num_modes^2 + num_modes)
        self.register_buffer(
            'bky',
            (2*torch.pi) * torch.cat([self.g_y, g_sin])
        )
        # (num_modes^2 + num_modes)
        self.register_buffer('b_n', torch.cat([n_all, n_sin]))
        # (num_modes^2 + num_modes)

        self.register_buffer(
            'ckx',
            (2*torch.pi) * torch.cat([self.g_x, g_sin])
        )
        # (num_modes^2 + num_modes)
        self.register_buffer(
            'cky',
            (2*torch.pi) * torch.cat([self.g_y, torch.zeros_like(g_sin)])
        )
        # (num_modes^2 + num_modes)
        self.register_buffer('c_n', torch.cat([n_all, n_sin]))
        # (num_modes^2 + num_modes)

        self.register_buffer('dkx', (2*torch.pi) * self.g_x)  # (num_modes^2)
        self.register_buffer('dky', (2*torch.pi) * self.g_y) # (num_modes^2)
        self.register_buffer('d_n', n_all)  # (num_modes^2)

    @property
    def dimension(self):
        return len(self.akx) + len(self.bkx) + len(self.ckx) + len(self.dkx)

    @property
    def volume(self):
        return (self.x_max[0] - self.x_min[0]) * (self.x_max[1] - self.x_min[1])

    def inner_product(self, u1, u2, x, integrator=None):
        """
        Computes the L^2 (on the box from x_min to x_max) inner product between
        two functions.
        :param u1: (B, *shape, 1) First input function values
        :param u2: (B, *shape, 1) Second input function values
        :param x: (B, *shape, 2) points at which both functions are sampled
        :param integrator: optional module to use for integration
        :return: (B) inner product of u1 and u2
        """
        unscaled = OrthonormalBasis.inner_product(u1, u2, x, integrator=integrator)
        # (B)

        return unscaled * self.volume

    def b_offset(self):
        return len(self.akx)

    def c_offset(self):
        return self.b_offset() + len(self.bkx)

    def d_offset(self):
        return self.c_offset() + len(self.ckx)

    def eval_basis(self, x, i=None):
        """
        Evaluates basis functions at given points
        :param x: (B, *shape, 2) Points at which to evaluate basis functions
        :param i: Optional (d) tensor of indices of basis functions to evaluate
        :return: (B, d, *shape, 1) Basis functions evaluated at the given points
        """
        if i is None:
            i = torch.arange(self.dimension, device=x.device, dtype=torch.long)

        a_i = i[i < self.b_offset()]
        b_i = i[(self.b_offset() <= i) & (i < self.c_offset())] - self.b_offset()
        c_i = i[(self.c_offset() <= i) & (i < self.d_offset())] - self.c_offset()
        d_i = i[self.d_offset() <= i] - self.d_offset()

        x_ = (x[..., 0] - self.x_min[0]) / (self.x_max[0] - self.x_min[0])
        y_ = (x[..., 1] - self.x_min[1]) / (self.x_max[1] - self.x_min[1])
        # each (B, *shape)

        ax = torch.cos(torch.einsum('N,B...->NB...', self.akx[a_i], x_))
        ay = torch.cos(torch.einsum('N,B...->NB...', self.aky[a_i], y_))
        add = (slice(None),) + (None,) * len(x_.shape)
        a = ax * ay / self.a_n[*add]
        # each (n_a, B, *shape)

        bx = torch.cos(torch.einsum('N,B...->NB...', self.bkx[b_i], x_))
        by = torch.sin(torch.einsum('N,B...->NB...', self.bky[b_i], y_))
        b = bx * by / self.b_n[*add]
        # each (n_b, B, *shape)

        cx = torch.sin(torch.einsum('N,B...->NB...', self.ckx[c_i], x_))
        cy = torch.cos(torch.einsum('N,B...->NB...', self.cky[c_i], y_))
        c = cx * cy / self.c_n[*add]
        # each (n_c, B, *shape)

        dx = torch.sin(torch.einsum('N,B...->NB...', self.dkx[d_i], x_))
        dy = torch.sin(torch.einsum('N,B...->NB...', self.dky[d_i], y_))
        d = dx * dy / self.d_n[*add]
        # each (n_d, B, *shape)

        basis = torch.empty((x.shape[0], i.shape[0], *x.shape[1:-1], 1), device=x.device)
        # (B, d, *shape, 1)

        basis[:, i < self.b_offset(), ..., 0] = torch.transpose(a, 1, 0)
        basis[:, (self.b_offset() <= i) & (i < self.c_offset()), ..., 0] = torch.transpose(b, 1, 0)
        basis[:, (self.c_offset() <= i) & (i < self.d_offset()), ..., 0] = torch.transpose(c, 1, 0)
        basis[:, self.d_offset() <= i, ..., 0] = torch.transpose(d, 1, 0)

        return basis

    def validate(self):
        """Validates that the basis is orthonormal"""
        x = torch.stack(
            torch.meshgrid(
                torch.linspace(self.x_min[0], self.x_max[0], 512),
                torch.linspace(self.x_min[1], self.x_max[1], 512),
                indexing='ij'
            ),
            dim=-1
        ).to(self.akx.device)  # (128, 128, 2)

        basis = self.eval_basis(x[None])[0, ...]  # (n, 128, 128, 1)

        gram = torch.zeros((len(basis), len(basis)), device=self.akx.device)
        for i in range(len(basis)):
            for j in range(len(basis)):
                gram[i, j] = self.inner_product(basis[i:i+1], basis[j:j+1], x[None])[0]
        print('Gram matrix', gram)
        print(f'Max deviation from identity: {(gram - torch.eye(self.dimension)).abs().max()}')

    def show(self):
        """Visualizes the basis functions"""
        x = torch.stack(
            torch.meshgrid(
                torch.linspace(self.x_min[0], self.x_max[0], 128),
                torch.linspace(self.x_min[1], self.x_max[1], 128),
                indexing='ij'
            ),
            dim=-1
        ).to(self.akx.device)  # (128, 128, 2)

        basis = self.eval_basis(x[None])[0, ..., 0]  # (n, 128, 128)
        im_kwargs = {
            'extent': [self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1]],
            'origin': 'lower',
            'cmap': 'seismic'
        }
        for i in range(len(basis)):
            im = plt.imshow(basis[i].T, **im_kwargs)
            plt.colorbar(im)
            plt.xlabel('x')
            plt.ylabel('y')
            if i < self.b_offset():
                gx = int(round(self.akx[i].item() / (2 * torch.pi)))
                gy = int(round(self.aky[i].item() / (2 * torch.pi)))
                plt.title(f'a (cos(x)cos(y)) basis gx = {gx}, gy = {gy} ({i+1}/{self.dimension})')
            elif i < self.c_offset():
                gx = int(round(self.bkx[i - self.b_offset()].item() / (2 * torch.pi)))
                gy = int(round(self.bky[i - self.b_offset()].item() / (2 * torch.pi)))
                plt.title(f'b (cos(x)sin(y)) basis gx = {gx}, gy = {gy} ({i+1}/{self.dimension})')
            elif i < self.d_offset():
                gx = int(round(self.ckx[i - self.c_offset()].item() / (2 * torch.pi)))
                gy = int(round(self.cky[i - self.c_offset()].item() / (2 * torch.pi)))
                plt.title(f'c (sin(x)cos(y)) basis gx = {gx}, gy = {gy} ({i+1}/{self.dimension})')
            else:
                gx = int(round(self.dkx[i - self.d_offset()].item() / (2 * torch.pi)))
                gy = int(round(self.dky[i - self.d_offset()].item() / (2 * torch.pi)))
                plt.title(f'd (sin(x)sin(y)) basis gx = {gx}, gy = {gy} ({i+1}/{self.dimension})')
            plt.show()
