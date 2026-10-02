"""
Implements a Gaussian Random Field generator using truncated Karhunen-Loeve
basis techniques.
"""

import torch
import mlx


class GRF(torch.nn.Module):
    def __init__(self, basis, variances):
        """
        Implements a Gaussian Random Field generator using truncated Karhunen-Loeve
        basis techniques.

        :param basis: OrthonormalBasis object representing the truncated
            Karhunen-Loeve basis that orthogonalizes this field's covariance
            operator
        :param variances: Corresponding covariance eigenvalues; either an
            explicit list of eigenvalues or a config for a module that evaluates
            the eigenvalues as a function of the basis object
        """
        super().__init__()
        self.basis = mlx.create_module(basis)

        if isinstance(variances, dict):
            variance_computer = mlx.create_module(variances)
            self.variances = variance_computer(self.basis)
        else:
            self.variances = torch.tensor(variances)

    def _make_fn(self, c):
        def f(x):
            basis_val = self.basis.eval_basis(x[None])[0]  # (b, *shape, 1)
            # (n, b)

            return torch.einsum('b...D,b->...D', basis_val, c)

        return f

    def forward(self, n, x):
        """
        Generates samples from the random field
        :param n: Number of samples to generate
        :param x: (*shape, d) Points at which to evaluate the field, or None,
            in which case a list of n callables that evaluate the functions
            at any points is returned
        :return: (n, *shape, 1) Basis evaluated at x, or list of n callables,
            each mapping (*shape, d) -> (*shape, 1)
        """
        if x is None:
            coef = torch.sqrt(self.variances) * torch.randn((n, self.basis.dimension), device=self.variances.device)
            # (n, b)

            return [self._make_fn(coef[i]) for i in range(n)]
        else:
            basis_val = self.basis.eval_basis(x[None])[0]  # (b, *shape, 1)
            coef = torch.sqrt(self.variances) * torch.randn((n, self.basis.dimension), device=self.variances.device)
            # (n, b)

            return torch.einsum('b...D,Nb->N...D', basis_val, coef)


class Fourier1dRationalDecayVariances:
    def __init__(self, alpha, beta, gamma, allow_offset=True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.allow_offset = allow_offset

    def __call__(self, basis):
        k = basis.k()
        g = k / (2 * torch.pi)
        base = self.alpha**2 / (self.beta + g**2) ** self.gamma
        if self.allow_offset:
            return base
        else:
            z = torch.zeros_like(g)
            return base * (~torch.isclose(g, z))


class Fourier2dRationalDecayVariances:
    def __init__(self, alpha, beta, gamma, allow_offset=True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.allow_offset = allow_offset

    def __call__(self, basis):
        kx, ky = basis.kx(), basis.ky()
        gx, gy = kx / (2 * torch.pi), ky / (2 * torch.pi)
        base = self.alpha**2 / (self.beta + gx**2 + gy**2) ** self.gamma
        if self.allow_offset:
            return base
        else:
            z = torch.zeros_like(gx)
            return base * (~(torch.isclose(gx, z) & torch.isclose(gy, z)))
