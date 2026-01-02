from operatorlearning.data.synthetic.poisson import (
    DenseSourceGenerator,
    PoissonDataGenerator,
    PoissonDataset
)
import operatorlearning as ol

import mlx
import torch.utils.data

from abc import ABC, abstractmethod


class BasePoissonTrainer(mlx.training.BaseTrainer, ABC):
    grid = None
    grid_batch = None
    loss_fn = None

    def load_datasets(self, config):
        self.loss_fn = mlx.modules.RelativeL2Loss()

        source_gen = DenseSourceGenerator(
            [6, 6], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )

        gen = PoissonDataGenerator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 10.0]),
            source_gen
        )

        self.grid = ol.GridFunction.uniform_x(gen.a, gen.b, num=31)  # (H, W, 2)
        self.grid_batch = torch.tile(
            self.grid[None],
            (config['training']['batch_size'],) + (1,) * len(self.grid.shape)
        )  # (B, H, W, 2)

        sources, solutions = [], []
        for i in range(config['training']['dataset_size']):
            source, solution = gen(1)
            sources.append(source[0])
            solutions.append(solution[0])

        train_dataset = PoissonDataset(sources, solutions)
        train_dataset.x = self.grid
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )

        sources, solutions = [], []
        for i in range(config['testing']['dataset_size']):
            source, solution = gen(1)
            sources.append(source[0])
            solutions.append(solution[0])

        test_dataset = PoissonDataset(sources, solutions)
        test_dataset.x = self.grid
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        return (
            {'train': train_dataset, 'test': test_dataset},
            {'train': train_loader, 'test': test_loader}
        )

    @abstractmethod
    def apply_model(self, source):
        raise NotImplemented

    def loss(self, data):
        source, solution = data
        pred = self.apply_model(source)

        return pred, {'objective': self.loss_fn(pred, solution)}
