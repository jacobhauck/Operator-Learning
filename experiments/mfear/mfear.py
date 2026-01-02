import mlx
import torch
import operatorlearning as ol
from experiments.poisson import BasePoissonTrainer


class MFEARPoissonTrainer(BasePoissonTrainer):
    def apply_model(self, source):
        return self.model(u=source, x_in=self.grid_batch, x_out=self.grid_batch)


class MFEARDemoExperiment(mlx.WandBExperiment):
    def wandb_run(self, config, run):
        trainer = MFEARPoissonTrainer(config, run)
        trainer.train(config['training']['epochs'])
        losses, _ = trainer.evaluate(datasets=('train', 'test'))

        for dataset, dataset_losses in losses.items():
            print(f'===== Losses for {dataset} =====')
            for loss_name, loss_vals in dataset_losses.items():
                print(f'Loss {loss_name}')
                print(f'Mean: {loss_vals.mean()}')
                print(f'Median: {loss_vals.median()}')
                print(f'Standard deviation: {loss_vals.std()}')
                print()
            print()

        source, solution = trainer.datasets['test'][0]

        solution_fn = ol.GridFunction(
            solution, x=trainer.grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=torch.tensor([0.0, 0.0]),
            x_max=torch.tensor([10.0, 10.0])
        )

        solution_pred = ol.GridFunction(
            trainer.model(u=source[None], x_in=trainer.grid_batch, x_out=trainer.grid_batch)[0].detach(),
            x=trainer.grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=torch.tensor([0.0, 0.0]),
            x_max=torch.tensor([10.0, 10.0])
        )
        solution_fn.quick_visualize()
        solution_pred.quick_visualize()
