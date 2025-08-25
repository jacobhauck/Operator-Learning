"""
Supporting modules for Fourier Neural Operator (see models.fno.base).
"""
import torch


class FourierLayer1d(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int
    ):
        super(FourierLayer1d, self).__init__()