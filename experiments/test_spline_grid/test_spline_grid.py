import mlx
import torch
from operatorlearning import GridFunction

torch.set_default_dtype(torch.float64)


@mlx.experiment
def run_test(config, name, group=None):
    integrator = mlx.create_module(config['integrator'])
    x = GridFunction.uniform_x(torch.zeros(2), torch.ones(2), config['base_resolution'])
    # (r, r, 2)
    x2 = x[::2, ::2]  # (r/2, r/2, 2)
    x4 = x[::4, ::4]  # (r/4, r/4, 2)

    f = torch.stack([
        torch.sin(x[None]).sum(dim=-1),
        torch.cos(x[None]).sum(dim=-1)
    ], dim=-1)[None]
    # (1, r, r, 2)
    f2 = torch.stack([
        torch.sin(x2[None]).sum(dim=-1),
        torch.cos(x2[None]).sum(dim=-1)
    ], dim=-1)[None]
    # (1, r/2, r/2, 2)
    f4 = torch.stack([
        torch.sin(x4[None]).sum(dim=-1),
        torch.cos(x4[None]).sum(dim=-1)
    ], dim=-1)[None]
    # (1, r/4, r/4, 2)

    print('Integrating 0')
    approx = integrator(f, x[None])
    print('Integrating 2')
    approx2 = integrator(f2, x2[None])
    print('Integrating 4')
    approx4 = integrator(f4, x4[None])

    actual = torch.stack([1 - torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))])
    actual *= 2

    print('Actual integral')
    print(actual)
    print('Approximations')
    for approx_i in [approx, approx2, approx4]:
        print('Val', approx_i, 'Err', (approx_i - actual).abs())
