import mlx
import torch
import torch.utils.data
from operatorlearning.modules import two_step_deeponet
import operatorlearning.data.synthetic.poisson as poisson
import operatorlearning as ol


class TwoStepDeepONetDemoExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        model = mlx.create_model(config['model'])
        model.to(config['device'])
        helper = two_step_deeponet.TrainingHelper(config['training']['dataset_size'], model)
        helper.to(config['device'])

        source_gen = poisson.DenseSourceGenerator(
            [6, 6], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        ).to(config['device'])
        gen = poisson.PoissonDataGenerator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 10.0]),
            source_gen
        ).to(config['device'])

        dataset_sources, dataset_solutions = [], []
        for _ in range(config['training']['dataset_size']):
            sources, solutions = gen(1)
            dataset_sources.append(sources[0])
            dataset_solutions.append(solutions[0])
        dataset_base = poisson.PoissonDataset(dataset_sources, dataset_solutions)
        dataset = two_step_deeponet.IndexedDataset(dataset_base)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )

        grid = ol.GridFunction.uniform_x(gen.a, gen.b, num=31).to(config['device'])
        # (H, W, 2)
        grid_batch = torch.tile(
            grid[None],
            (config['training']['batch_size'],) + (1,) * len(grid.shape)
        )  # (B, H, W, 2)
        dataset_base.x = grid

        # Step 1 training
        optim1 = torch.optim.Adam(
            list(model.deeponet.trunk_net.parameters()) + list(helper.parameters()),
            lr=config['training']['step_1']['lr']
        )
        losses = []
        loss_fn = mlx.modules.RelativeL2Loss()
        for epoch in range(config['training']['step_1']['epochs']):
            for batch, (indices, _, solutions) in enumerate(data_loader):
                optim1.zero_grad()
                a = helper(indices, step=1)
                pred_solutions = model.step_one(a, grid_batch)
                loss = loss_fn(pred_solutions, solutions)
                loss.backward()
                optim1.step()
                losses.append(loss.item())
                print(f'Step 1 epoch {epoch}, batch {batch}: loss = {sum(losses[-500:]) / min(500, len(losses)):.05f}')

        helper.set_target_matrix(grid)

        # Step 2 training
        optim2 = torch.optim.Adam(
            model.deeponet.branch_net.parameters(),
            lr=config['training']['step_2']['lr']
        )
        loss_fn = torch.nn.MSELoss()
        losses = []
        for epoch in range(config['training']['step_2']['epochs']):
            for batch, (indices, sources, _) in enumerate(data_loader):
                optim2.zero_grad()
                target = helper(indices, step=2)
                pred = model.step_two(sources)
                loss = loss_fn(pred, target.T)
                loss.backward()
                optim2.step()
                losses.append(loss.item())
                print(f'Step 2 epoch {epoch}, batch {batch}: loss = {sum(losses[-500:]) / min(500, len(losses)):.05f}')

        # Testing
        model.train(False)
        losses = []
        for i in range(100):
            sources, solutions = gen(1)
            source_disc = torch.stack([source(grid) for source in sources])  # (1, H, W, 1)
            sol_disc = torch.stack([sol(grid) for sol in solutions])  # (1, H, W, 1)
            pred = model(u=source_disc, x_out=grid[None])
            loss = loss_fn(pred, sol_disc) / loss_fn(sol_disc, torch.zeros_like(sol_disc))
            losses.append(loss.item())
        losses = torch.tensor(losses)

        print('Average loss:', losses.mean().item())
        print('Standard deviation:', losses.std().item())

        source, solution = gen(1)
        solution_pred = ol.GridFunction(
            model(u=source[0](grid)[None], x_out=grid[None])[0].detach().cpu(),
            x=grid.cpu(),
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=gen.a.cpu(), x_max=gen.b.cpu()
        )
        source[0].cpu().quick_visualize()
        solution[0].cpu().quick_visualize()
        solution_pred.cpu().quick_visualize()
