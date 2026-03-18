import mlx
import torch

import matplotlib.pyplot as plt


class TestDifferentiator(mlx.Experiment):
    def run(self, config, name, group=None):
        diff = mlx.create_module(config['differentiator'])

        x = torch.linspace(0, 1, 200)[None, :, None]
        f = torch.sin(3*x)
        f_p = 3*torch.cos(3*x)
        f_p_a = diff(f, x)

        plt.plot(x.flatten(), f.flatten(), label='f')
        plt.plot(x.flatten(), f_p.flatten(), label="f'")
        plt.plot(x.flatten(), f_p_a.flatten(), label="f' numeric")
        plt.legend()
        plt.show()
