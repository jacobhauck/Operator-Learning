import mlx
import torch
import matplotlib.pyplot as plt


@mlx.experiment
def grf_test(config, name, group=None):
    grf = mlx.create_module(config['grf'])
    x = torch.linspace(config['x_min'], config['x_max'], 512)[:, None]
    values = grf(config['num_samples'], x)

    for i in range(config['num_samples']):
        plt.plot(x[:, 0], values[i, :, 0])
        plt.show()
        plt.close()
