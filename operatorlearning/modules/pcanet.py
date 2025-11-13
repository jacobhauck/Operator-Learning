import mlx
import torch


def pca_basis(dataset, x, num_modes, mean=None):
    """
    Calculate a PCA basis (mean + basis functions) on a fixed set of
    sampling points

    :param dataset: Dataset of functions (sequence of Function objects)
    :param x: (*shape, d_in) Sampling points
    :param num_modes: Number of PCA modes to keep
    :param mean: Optional (*shape, d_out) Sample values of a mean function
        to force the PCA to use instead of the calculated mean
    :return: Tuple mean, basis. mean (*shape, d_out) is the mean function of
        the dataset, and basis (num_modes, *shape, d_out), the samples of the
        PCA basis functions
    """
    d_out = dataset[0].d_out
    data = torch.empty(
        (len(dataset), *x.shape[:-1], d_out),
        dtype=x.dtype,
        device=x.device
    )  # (n, *shape, d_out)

    for i, f in enumerate(dataset):
        data[i] = f(x)

    if mean is None:
        mean = data.mean(dim=0)  # (*shape, d_out)

    centered = (data - mean[None]).reshape(len(dataset), -1)  # (n, D)
    assert min(centered.shape) >= num_modes, \
        'Data cannot provide requested number of modes'

    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    # (n, r), (r, r), (r, D), r = min(n, D)

    basis = vt[:num_modes].reshape(num_modes, *mean.shape)
    # (num_modes, *shape, d_out)

    # NOTE: basis is l^2 normalized along (*shape, d_out), so we need to
    # divide by a quadrature weight that will make it approximately L^2
    # normalized. Without knowing the distribution of x, we choose the
    # simplest option (and also the option that matches the L^2 inner product
    # approximation in PCANet), 1/sqrt(prod(shape)), which assigns equal weight
    # to all points while remaining a consistent approximation of the L^2
    # norm of the basis function. In short, when we compute the approximate
    # L^2 norm of the basis functions via something like
    # norm = torch.sqrt((basis**2).sum(dim=-1).mean(dim=(1, 2, ..., len(shape))))
    # we should get 1 for all basis functions.
    basis *= ((torch.prod(torch.tensor(mean.shape[:-1]))) ** .5).item()

    return mean, basis


class PCANet(torch.nn.Module):
    def __init__(
            self,
            u_num_modes,
            u_sample_shape,
            v_num_modes,
            v_sample_shape,
            approximator
    ):
        """
        :param u_num_modes: Number of PCA modes to keep for the input function u
        :param u_sample_shape: Shape of sampled input function u (*u_shape, u_d_out)
        :param v_num_modes: Number of PCA modes to keep for the output function v
        :param v_sample_shape: Shape of sampled output function v (*v_shape, v_d_out)
        :param approximator: Config of approximation module. Should map
            (B, u_num_modes) -> (B, v_num_modes)
        """
        super().__init__()
        self.u_num_modes = u_num_modes
        self.v_num_modes = v_num_modes
        self.register_buffer('u_mean', torch.empty(u_sample_shape))
        self.register_buffer('u_basis', torch.empty((u_num_modes, *u_sample_shape)))
        self.register_buffer('v_mean', torch.empty(v_sample_shape))
        self.register_buffer('v_basis', torch.empty((v_num_modes, *v_sample_shape)))
        self.approximator = mlx.create_module(approximator)
    
    def fit_pca(self, dataset, x, y):
        """
        Fit the model's PCA bases
        :param dataset: The dataset to fit to. Should be an iterable whose
            elements are pairs (u, v) of input and output functions
        :param x: (*shape, u_d_in) Points at which to sample input functions
        :param y: (*shape, v_d_in) Points at which to sample output functions
        """
        u_dataset, v_dataset = [], []
        for u, v in dataset:
            u_dataset.append(u)
            v_dataset.append(v)
        
        u_mean, u_basis = pca_basis(u_dataset, x, self.u_num_modes)
        v_mean, v_basis = pca_basis(v_dataset, y, self.v_num_modes)
        self.u_mean[:] = u_mean
        self.u_basis[:] = u_basis
        self.v_mean[:] = v_mean
        self.v_basis[:] = v_basis
    
    def forward(self, u):
        """
        :param u: (B, *input_shape, u_d_out) Input function sampled on the
            same points used for the input PCA basis
        :return: (B, *output_shape, v_d_out) Output function sampled on the
            same points used for the output PCA basis
        """

        # L^2 projection of u onto u_basis
        u_centered = u - self.u_mean[None]
        # (B, *input_shape, u_d_out)

        h = 1 / torch.prod(torch.tensor(u_centered.shape[1:-1])).item()
        u_proj = h * torch.einsum('B...d,N...d->BN', u_centered, self.u_basis)
        # (B, u_num_modes)

        v_proj = self.approximator(u_proj)  # (B, v_num_modes)

        v = torch.einsum('BN,N...d->B...d', v_proj, self.v_basis)
        # (B, *output_shape, v_d_out)

        return v + self.v_mean[None]
