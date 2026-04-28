import mlx
from operatorlearning.modules.integration import OpenSimpsonIntegrator
import torch
from math import log


class TestSimpsonExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        torch.set_default_dtype(torch.float64)
        i1 = OpenSimpsonIntegrator([0.], [1.])

        true_value = torch.sin(torch.tensor(3.0)) / 3

        def make_at(n):
            dx = 1/n
            x = torch.linspace(dx/2, 1-dx/2, n)[None, :, None]
            f = torch.cos(3*x)
            return f, x

        def true_simpson(n):
            dx = 1 / (n - 1)
            w = torch.full((n,), dx * 2/3)
            w[0] = dx / 3
            w[-1] = dx / 3
            w[1:-1:2] = dx * 4/3
            x = torch.linspace(0, 1, n)
            return torch.sum(torch.cos(3*x) * w)

        errors = [abs(float(true_value - i1(*make_at(n)))) for n in [5, 10, 20, 40, 80, 160]]
        rates = [log(errors[i+1] / errors[i]) / log(1/2) for i in range(4)]
        print('Open Simpson errors', errors)
        print('Empirical convergence order', rates)

        errors = [abs(float(true_value - true_simpson(n))) for n in [5, 11, 21, 41, 81, 161]]
        rates = [log(errors[i+1] / errors[i]) / log(1/2) for i in range(4)]
        print('Typical Simpson errors', errors)
        print('Empirical convergence order', rates)

