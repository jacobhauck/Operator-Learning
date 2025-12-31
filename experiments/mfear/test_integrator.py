from typing import Mapping

import torch
import mlx
from operatorlearning import GridFunction


class TestIntegratorExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        integrator = mlx.create_module(config['integrator'])

        x_min = torch.zeros(config['dimension'])
        x_max = torch.ones(config['dimension'])
        xs1 = GridFunction.uniform_xs(x_min, x_max, config['resolution'])
        xs2 = []
        for x_i in xs1:
            xs2.append(x_i + (torch.rand_like(x_i) - 0.5) / config['resolution'])
            xs2[-1][0] = 0.0
            xs2[-1][-1] = 1.0

        x1 = GridFunction.build_x(xs1)
        x2 = GridFunction.build_x(xs2)
        # (*in_shape, dimension)

        x = torch.stack([x1, x2])
        # (2, *in_shape, dimension)

        f = torch.stack([
            torch.sin(x).sum(dim=-1),
            torch.cos(x).sum(dim=-1)
        ], dim=-1)
        # (2, *in_shape, 2)

        approx = integrator(f, x)  # (2, 2)
        actual = torch.stack([1 - torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))])
        actual *= config['dimension']

        print('Actual integral')
        print(actual)
        print('Approximations')
        for approx_i in approx:
            print('Val', approx_i, 'Err', (approx_i - actual).abs())
