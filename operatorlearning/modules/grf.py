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

    def forward(self, n, x):
        """
        Generates samples from the random field
        :param n: Number of samples to generate
        :param x: (*shape, d) Points at which to evaluate the field
        :return: (n, *shape, 1) Basis evaluated at x
        """
        basis_val = self.basis.eval_basis(x[None])[0]  # (b, *shape, 1)
        coef = torch.sqrt(self.variances) * torch.randn((n, self.basis.dimension), device=basis_val.device)
        # (n, b)

        return torch.einsum('b...D,Nb->N...D', basis_val, coef)


class Fourier1dRationalDecayVariances:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, basis):
        k = basis.k()
        g = k / (2 * torch.pi)
        return self.alpha**2 / (self.beta + g**2) ** self.gamma


class Fourier2dRationalDecayVariances:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, basis):
        kx, ky = basis.kx(), basis.ky()
        gx, gy = kx / (2 * torch.pi), ky / (2 * torch.pi)
        return self.alpha**2 / (self.beta + gx**2 + gy**2) ** self.gamma
