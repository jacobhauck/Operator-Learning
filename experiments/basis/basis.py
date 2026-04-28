import mlx
from operatorlearning.modules.basis import FullFourierBasis2d
from operatorlearning import GridFunction
import torch
import matplotlib.pyplot as plt
import numpy as np


class BasisExperiment(mlx.Experiment):
    def  run(self, config, name, group=None):
        basis = FullFourierBasis2d(config['num_modes'], config['x_min'], config['x_max'])
        if config.get('show', False):
            basis.show()

        if config.get('validate', False):
            basis.validate()

        x = GridFunction.uniform_x(torch.tensor([0, 0]), torch.tensor([5, 5]), 512)
        x_, y_ = x[..., 0], x[..., 1]
        x_m = 2.15
        y_m = 2.72
        a = 0.025
        sigma = 1/30
        mu = 6*sigma
        r = ((x_ - x_m)**2 + (y_ - y_m)**2) ** 0.5
        uo = ((r > mu - 6*sigma) & (r < mu + 6*sigma)) * a * torch.exp(-(r-mu)**2/(2*sigma)**2)
        vo = uo * (x_ - x_m) / (r + 1e-15)
        c = basis.coefficients(vo[None, ..., None], x[None])[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        v_min = float(vo.min())
        v_max = float(vo.max())
        im_kwargs = {
            'cmap': 'seismic',
            'vmin': v_min,
            'vmax': v_max,
            'extent': [config['x_min'][0], config['x_max'][0], config['x_min'][1], config['x_max'][1]],
            'origin': 'lower'
        }
        axes[0].imshow(vo.T, **im_kwargs)
        axes[0].set_title('Original')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        v_recon = basis.eval(x[None], c[None])[0, ..., 0]
        axes[1].imshow(v_recon.T, **im_kwargs)
        error = (torch.mean((vo - v_recon) ** 2) / torch.mean(vo**2))
        axes[1].set_title(f'Projection (d = {basis.dimension}, error = {100*float(error):.1f}%)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.show()

        c_sort = np.cumsum(sorted((c**2 / (25*torch.mean(vo**2))).numpy().tolist(), reverse=True))
        plt.plot(c_sort)
        plt.hlines([1.0], 0, len(c_sort), color='red')
        plt.show()
