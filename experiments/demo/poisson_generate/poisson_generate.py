import torch

import data.synthetic.poisson
import experiments


class PoissonGenerateExperiment(experiments.Experiment):
    def run(self, config, name, group=None):
        source_gen = data.synthetic.poisson.DenseSourceGenerator(
            config['n_modes'], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )
        gen = data.synthetic.poisson.PoissonDataGenerator(
            torch.tensor([config['x_min'], config['y_min']]),
            torch.tensor([config['x_max'], config['y_max']]),
            source_gen
        )

        source, solution = gen(1)
        source[0].quick_visualize()
        solution[0].quick_visualize()
