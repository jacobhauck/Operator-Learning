import torch
import mlx


class FunctionalL2Loss(torch.nn.Module):
    """
    L^2 loss, either squared or not, relative or not. Allows for custom integrator
    (default just computes mean over spatial dimensions)
    """

    def __init__(self, relative=True, squared=False, integrator=None):
        """
        :param relative: Whether to use relative loss (norm of error/norm of target)
        :param squared: Whether to return the squared L^2 norm squared (i.e., MSE)
        :param integrator: Optional integrator module that maps function sample
            array (B, *shape, d_out) -> integral of function (B, d_out). If not
            given, then the integral is approximated by the mean over shape (i.e.,
            spatial points).
        """
        super().__init__()
        self.relative = relative
        self.squared = squared
        if integrator is not None:
            self.integrator = mlx.create_module(integrator)
        else:
            self.integrator = None

    def __repr__(self):
        return f'L2Loss(relative={self.relative}, squared={self.squared})'

    def forward(self, prediction, target, x=None):
        """
        :param prediction: (B, *shape, d_out) prediction function samples
        :param target: (B, *shape, d_out) target function samples
        :param x: (B, *shape, d_in) shared coordinates (or None, if using
            mean integration)
        """
        diff2 = torch.sum((prediction - target) ** 2, dim=-1)  # (B, *shape)

        if self.relative:
            magnitude2 = torch.sum(target**2, dim=-1)  # (B, *shape)
            if self.integrator is not None and x is not None:
                i_magnitude2 = self.integrator(magnitude2[..., None], x)[0]  # (B,)
            else:
                i_magnitude2 = torch.mean(magnitude2.view(diff2.shape[0], -1), dim=1)
                # (B,)
        else:
            i_magnitude2 = 1.0

        if self.integrator is not None and x is not None:
            i_diff2 = self.integrator(diff2[..., None], x)[0]  # (B,)
        else:
            i_diff2 = torch.mean(diff2.view(diff2.shape[0], -1), dim=1)
            # (B,)

        if self.squared:
            error = i_diff2
            norm = i_magnitude2
        else:
            error = i_diff2 ** .5
            norm = i_magnitude2 ** .5

        return torch.mean(error / norm)


class FunctionalL1Loss(torch.nn.Module):
    """
    L^1 loss, either relative or not. Allows for custom integrator
    (default just computes mean over spatial dimensions)
    """

    def __init__(self, relative=True, integrator=None):
        """
        :param relative: Whether to use relative loss (norm of error/norm of target)
        :param integrator: Optional integrator module that maps function sample
            array (B, *shape, d_out) -> integral of function (B, d_out). If not
            given, then the integral is approximated by the mean over shape (i.e.,
            spatial points).
        """
        super().__init__()
        self.relative = relative
        if integrator is not None:
            self.integrator = mlx.create_module(integrator)
        else:
            self.integrator = None

    def __repr__(self):
        return f'L1Loss(relative={self.relative})'

    def forward(self, prediction, target, x=None):
        """
        :param prediction: (B, *shape, d_out) prediction function samples
        :param target: (B, *shape, d_out) target function samples
        :param x: (B, *shape, d_in) shared coordinates (or None, if using
            mean integration)
        """
        diff = torch.sum((prediction - target).abs(), dim=-1)  # (B, *shape)

        if self.relative:
            magnitude = torch.sum(target.abs(), dim=-1)  # (B, *shape)
            if self.integrator is not None and x is not None:
                i_magnitude = self.integrator(magnitude[..., None], x)[0]  # (B,)
            else:
                i_magnitude = torch.mean(magnitude.view(diff.shape[0], -1), dim=1)
                # (B,)
        else:
            i_magnitude = 1.0

        if self.integrator is not None and x is not None:
            i_diff = self.integrator(diff[..., None], x)[0]  # (B,)
        else:
            i_diff = torch.mean(diff.view(diff.shape[0], -1), dim=1)
            # (B,)

        return torch.mean(i_diff / i_magnitude)


class FunctionalH1Loss(torch.nn.Module):
    """
    H^1 loss, either relative or not, squared or not. Allows for custom
    integrator (default just computes mean over spatial dimensions).
    """
    def __init__(self, differentiator, relative=True, squared=False, integrator=None):
        """
        :param differentiator: Module that maps function samples (B, *shape, d_out)
            and sample points (B, *shape, d_in) to the Jacobian at the sample
            points (B, *shape, d_out, d_in)
        :param relative: Whether to use relative loss (norm of error/norm of target)
        :param squared: Whether to return the squared H^1 norm
        :param integrator: Optional integrator module that maps function sample
            array (B, *shape, d_out) -> integral of function (B, d_out). If not
            given, then the integral is approximated by the mean over shape (i.e.,
            spatial points).
        """
        super().__init__()
        self.differentiator = mlx.create_module(differentiator)
        self.relative = relative
        self.squared = squared
        if integrator is not None:
            self.integrator = mlx.create_module(integrator)
        else:
            self.integrator = None

    def __repr__(self):
        return f'H1Loss(relative={self.relative}, squared={self.squared})'

    def forward(self, prediction, target, x):
        """
        :param prediction: (B, *shape, d_out) predicted function samples
        :param target: (B, *shape, d_out) target function samples
        :param x: (B, *shape, d_in) sampling points for prediction and target
        """
        j_diff = self.differentiator(prediction - target, x)
        # (B, *shape, d_out, d_in)
        j_diff2 = torch.sum(j_diff**2, dim=[-1, -2])  # (B, *shape)

        if self.relative:
            j_target = self.differentiator(target, x)
            # (B, *shape, d_out, d_in)

            magnitude2 = torch.sum(j_target**2, dim=[-1, -2])  # (B, *shape)
            if self.integrator is not None and x is not None:
                i_magnitude2 = self.integrator(magnitude2[..., None], x)[0]  # (B,)
            else:
                i_magnitude2 = torch.mean(magnitude2.view(j_diff2.shape[0], -1), dim=1)
                # (B,)
        else:
            i_magnitude2 = 1.0

        if self.integrator is not None and x is not None:
            i_diff2 = self.integrator(j_diff2[..., None], x)[0]  # (B,)
        else:
            i_diff2 = torch.mean(j_diff2.view(j_diff2.shape[0], -1), dim=1)
            # (B,)

        if self.squared:
            error = i_diff2
            norm = i_magnitude2
        else:
            error = i_diff2 ** .5
            norm = i_magnitude2 ** .5

        return torch.mean(error / norm)


class FunctionalTVLoss(torch.nn.Module):
    """
    Total variation loss, either relative or not. Allows for custom
    integrator (default just computes mean over spatial dimensions).
    """
    def __init__(self, differentiator, relative=True, integrator=None):
        """
        :param differentiator: Module that maps function samples (B, *shape, d_out)
            and sample points (B, *shape, d_in) to the Jacobian at the sample
            points (B, *shape, d_out, d_in)
        :param relative: Whether to use relative loss (norm of error/norm of target)
        :param integrator: Optional integrator module that maps function sample
            array (B, *shape, d_out) -> integral of function (B, d_out). If not
            given, then the integral is approximated by the mean over shape (i.e.,
            spatial points).
        """
        super().__init__()
        self.differentiator = mlx.create_module(differentiator)
        self.relative = relative
        if integrator is not None:
            self.integrator = mlx.create_module(integrator)
        else:
            self.integrator = None

    def __repr__(self):
        return f'TVLoss(relative={self.relative})'

    def forward(self, prediction, target, x):
        """
        :param prediction: (B, *shape, d_out) predicted function samples
        :param target: (B, *shape, d_out) target function samples
        :param x: (B, *shape, d_in) sampling points for prediction and target
        """
        j_diff = self.differentiator(prediction - target, x)
        # (B, *shape, d_out, d_in)

        # Use Frobenius norm of Jacobian
        j_diff = torch.sum(j_diff**2, dim=[-1, -2]) ** .5  # (B, *shape)

        if self.relative:
            j_target = self.differentiator(target, x)
            # (B, *shape, d_out, d_in)

            magnitude = torch.sum(j_target**2, dim=[-1, -2]) ** .5  # (B, *shape)
            if self.integrator is not None and x is not None:
                i_magnitude = self.integrator(magnitude[..., None], x)[0]  # (B,)
            else:
                i_magnitude = torch.mean(magnitude.view(j_diff.shape[0], -1), dim=1)
                # (B,)
        else:
            i_magnitude = 1.0

        if self.integrator is not None and x is not None:
            i_diff = self.integrator(j_diff[..., None], x)[0]  # (B,)
        else:
            i_diff = torch.mean(j_diff.view(j_diff.shape[0], -1), dim=1)
            # (B,)

        return torch.mean(i_diff / i_magnitude)
