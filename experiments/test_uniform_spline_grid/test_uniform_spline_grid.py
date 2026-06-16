import torch
import mlx
from operatorlearning import GridFunction


torch.set_default_dtype(torch.float64)


@mlx.experiment
def run(config, name, group=None):
    integrator = mlx.create_module(config['integrator'])
    x1 = torch.linspace(0, 1, config['resolution'] + 1)[:-1]
    x1 = x1 + x1[1] / 2
    x = torch.stack(torch.meshgrid(x1, x1, indexing='ij'), dim=-1)
    # (*in_shape, 2)

    f = torch.stack([
        torch.sin(x[None]).sum(dim=-1),
        torch.cos(x[None]).sum(dim=-1)
    ], dim=-1)[None]
    # (1, *in_shape, 2)

    approx = integrator(f, x[None])  # (2, 2)
    actual = torch.stack([1 - torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))])
    actual *= 2

    print('Actual integral')
    print(actual)
    print('Approximations')
    for approx_i in approx:
        print('Val', approx_i, 'Err', (approx_i - actual).abs())
