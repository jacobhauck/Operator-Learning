import mlx
import torch
from operatorlearning.modules import two_step_deeponet
import operatorlearning as ol
from experiments.poisson import BasePoissonTrainer


class TwoStepDeepONetPoissonTrainer(BasePoissonTrainer):
    helper = None
    current_step = None

    def load_datasets(self, config):
        datasets, data_loaders = super().load_datasets(config)
        for name in datasets:
            datasets[name] = two_step_deeponet.IndexedDataset(datasets[name])
            data_loaders[name] = torch.utils.data.DataLoader(
                datasets[name],
                batch_size=data_loaders[name].batch_size,
                shuffle=True,
            )
        return datasets, data_loaders

    def apply_model(self, source):
        return self.model(u=source, x_out=self.grid_batch)

    def num_step_1_steps(self):
        return self.config['training']['epochs'] * len(self.data_loaders['train'])

    def set_step_1(self):
        trunk_params = list(self.model.deeponet.trunk_net.parameters())
        self.helper = two_step_deeponet.TrainingHelper(len(self.datasets['train']), self.model)
        a_matrix = list(self.helper.parameters())
        self.optim = mlx.create_optimizer(trunk_params + a_matrix, self.config['optim'])
        self.current_step = 1

    def set_step_2(self):
        self.loss_fn = torch.nn.MSELoss()
        branch_params = self.model.deeponet.branch_net.parameters()
        self.helper.set_target_matrix(self.datasets['train'].base_dataset.x)
        self.optim = mlx.create_optimizer(branch_params, self.config['optim'])
        self.current_step = 2

    def loss(self, batch):
        indices, initial, final = batch

        if self.run.step == self.num_step_1_steps():
            self.save_checkpoint()
            print('Starting step 2 training')
            self.set_step_2()

        if self.current_step == 0:  # Evaluation mode
            pred = self.model(initial, x_out=self.grid_batch[0:1])
            loss = self.loss_fn(pred, final)
            return pred, {'objective': loss}
        elif self.current_step == 1:
            a = self.helper(indices, step=1)
            pred_final = self.model.step_one(a, self.grid_batch)
            loss = self.loss_fn(pred_final, final)
            return pred_final, {'objective': loss}
        else:
            target = self.helper(indices, step=2)
            pred = self.model.step_two(initial)
            loss = self.loss_fn(pred, target.T)
            return pred, {'objective': loss}

    def evaluate(self, datasets=('train',)):
        old_step = self.current_step
        self.current_step = 0
        result = super().evaluate(datasets)
        self.current_step = old_step
        return result

    def dump_additional_state(self):
        return {'helper': self.helper.state_dict()}

    def load_additional_state(self, state):
        if self.helper is None:
            self.helper = two_step_deeponet.TrainingHelper(len(self.datasets['train']), self.model)
            self.helper.to(self.device)
        self.helper.load_state_dict(state['helper'])


class TwoStepDeepONetDemoExperiment(mlx.WandBExperiment):
    def wandb_run(self, config, run):
        trainer = TwoStepDeepONetPoissonTrainer(config, run)

        if run.step is None or run.step == 0:
            trainer.set_step_1()
        step_1_epochs = config['training']['epochs']
        step_2_epochs = config['training'].get('epochs2', step_1_epochs)
        trainer.train(step_1_epochs + step_2_epochs)

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

        _, source, solution = trainer.datasets['test'][0]

        solution_fn = ol.GridFunction(
            solution, x=trainer.grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=torch.tensor([0.0, 0.0]),
            x_max=torch.tensor([10.0, 10.0])
        )

        solution_pred = ol.GridFunction(
            trainer.model(u=source[None], x_out=trainer.grid_batch)[0].detach(),
            x=trainer.grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=torch.tensor([0.0, 0.0]),
            x_max=torch.tensor([10.0, 10.0])
        )
        solution_fn.quick_visualize()
        solution_pred.quick_visualize()
