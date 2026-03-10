import mlx
import torch
import operatorlearning as ol
from operatorlearning.modules import (
    FunctionalL1Loss,
    FunctionalL2Loss,
    FunctionalH1Loss,
    FunctionalTVLoss
)


class TestLossExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        integrator = config['integrator']
        differentiator = config['differentiator']
        x = ol.GridFunction.uniform_x(
            torch.tensor([0.0]),
            torch.tensor([2*torch.pi]),
            config['num_points']
        )[None]

        f = torch.sin(x)
        g = torch.ones_like(f)

        losses = [
            FunctionalL1Loss(relative=False, integrator=integrator),
            FunctionalL1Loss(relative=True, integrator=integrator),
            FunctionalL2Loss(relative=True, squared=True, integrator=integrator),
            FunctionalL2Loss(relative=True, squared=False, integrator=integrator),
            FunctionalL2Loss(relative=False, squared=True, integrator=integrator),
            FunctionalL2Loss(relative=False, squared=False, integrator=integrator),
            FunctionalH1Loss(differentiator, relative=False, squared=False, integrator=integrator),
            FunctionalH1Loss(differentiator, relative=False, squared=True, integrator=integrator),
            FunctionalH1Loss(differentiator, relative=True, squared=False, integrator=integrator),
            FunctionalH1Loss(differentiator, relative=True, squared=False, integrator=integrator),
            FunctionalTVLoss(differentiator, relative=False, integrator=integrator),
            FunctionalTVLoss(differentiator, relative=True, integrator=integrator),
        ]

        expected = [
            2 * torch.pi,
            torch.pi / 2,
            3,
            3**0.5,
            3 * torch.pi,
            (3 * torch.pi) ** 0.5,
            torch.pi ** 0.5,
            torch.pi,
            1,
            1,
            4,
            1
        ]

        for loss_fn, actual in zip(losses, expected):
            loss = loss_fn(g, f, x).item()
            print(f'{loss_fn}: {loss}, expected: {actual}')
