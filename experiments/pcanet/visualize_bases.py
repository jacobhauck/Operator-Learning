import mlx
import operatorlearning as ol
from operatorlearning.modules import pcanet
from operatorlearning.data.synthetic import poisson
import matplotlib.pyplot as plt
import torch


class VisualizePCABasesExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        source_gen = poisson.DenseSourceGenerator(
            config['n_modes'], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )
        gen = poisson.PoissonDataGenerator(
            torch.tensor(config['x_min']),
            torch.tensor(config['x_max']),
            source_gen
        )

        sources = torch.nn.ModuleList()
        solutions = torch.nn.ModuleList()

        batch_start = 0
        while batch_start < config['num_samples']:
            batch_size = min(config['batch_size'], config['num_samples'] - batch_start)
            batch_sources, batch_solutions = gen(batch_size)
            sources.extend(batch_sources)
            solutions.extend(batch_solutions)
            batch_start += config['batch_size']

        x_pca = ol.GridFunction.uniform_x(gen.a, gen.b, 128)
        dataset = poisson.PoissonDataset(sources, solutions)

        source_dataset = [src for src, sol in dataset]
        solution_dataset = [sol for src, sol in dataset]

        src_mean, src_basis = pcanet.pca_basis(source_dataset, x_pca, config['num_pca_modes'])
        sol_mean, sol_basis = pcanet.pca_basis(solution_dataset, x_pca, config['num_pca_modes'])

        print('Source')
        ol.GridFunction(src_mean, x_pca).quick_visualize()
        for f in src_basis:
            ol.GridFunction(f, x_pca).quick_visualize()
            print(((f**2).sum(dim=-1).mean().item())**.5)

        print('Solution')
        ol.GridFunction(sol_mean, x_pca).quick_visualize()
        for f in sol_basis:
            ol.GridFunction(f, x_pca).quick_visualize()
            print(((f**2).sum(dim=-1).mean().item())**.5)
