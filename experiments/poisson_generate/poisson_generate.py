import torch
import operatorlearning.data.synthetic.poisson as poisson
import mlx


class PoissonGenerateExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        source_gen = poisson.DenseSourceGenerator(
            config['n_modes'], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )
        gen = poisson.PoissonDataGenerator(
            torch.tensor(config['x_min']),
            torch.tensor(config['x_max']),
            source_gen
        )

        source, solution = gen(1)
        source[0].quick_visualize()
        solution[0].quick_visualize()
