import torch


class ForwardEuler1dDifferentiator(torch.nn.Module):
    """
    Differentiates a 1D function using the forward Euler method
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, f, x):
        """
        :param f: (B, n, d_out) function samples
        :param x: (B, n, 1) sample points
        :return: (B, n, d_out, 1) derivative of f at samples
        """
        f_prime = torch.zeros_like(f)
        f_prime[:, :-1] = (f[:, 1:] - f[:, :-1]) / (x[:, 1:] - x[:, :-1])
        f_prime[:, -1] = (f[:, -1] - f[:, -2]) / (x[:, -1] - x[:, -2])
        return f_prime[..., None]
