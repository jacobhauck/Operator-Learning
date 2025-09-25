import torch

import data.synthetic.poisson
import experiments


class PoissonGenerateExperiment(experiments.Experiment):
    def run(self, config, name, group=None):
        source_gen = data.synthetic.poisson.DenseSourceGenerator(
            config['n_modes'], lambda k: 3/(1.0 + (k**2).sum(dim=-1, keepdim=True))
        )
        gen = data.synthetic.poisson.PoissonDataGenerator(
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

        dataset = data.synthetic.poisson.PoissonDataset(sources, solutions)
        dataset.save(config['output_folder'])
