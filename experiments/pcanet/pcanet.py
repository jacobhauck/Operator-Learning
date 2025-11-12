import mlx
import torch
import operatorlearning.data.synthetic.poisson as poisson
import operatorlearning as ol
from operatorlearning.modules import pcanet


class PCANetDemoExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        source_gen = poisson.DenseSourceGenerator(
            [6, 6], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )
        gen = poisson.PoissonDataGenerator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 10.0]),
            source_gen
        )
        loss_fn = mlx.modules.RelativeL2Loss()

        grid = ol.GridFunction.uniform_x(gen.a, gen.b, num=128)[:-1, :-1]

        source_dataset = []
        solution_dataset = []
        for i in range(config['training']['dataset_size']):
            source, solution = gen(1)
            source_dataset.append(source[0])
            solution_dataset.append(solution[0])

        model = mlx.create_model(config['model'])
        model.fit_pca(zip(source_dataset, solution_dataset), grid, grid)
        optim = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

        for i in range(config['training']['iterations']):
            sources, solutions = gen(config['training']['batch_size'])
            in_batch = torch.stack([source(grid) for source in sources])  # (B, H, W, 1)
            out_batch = torch.stack([sol(grid) for sol in solutions])  # (B, H, W, 1)

            optim.zero_grad()
            pred_batch = model(in_batch)
            loss = loss_fn(pred_batch, out_batch)
            loss.backward()
            optim.step()

            print(f'Batch {i}: loss = {loss.item():.06f}')

        model.train(False)
        losses = []
        for i in range(100):
            sources, solutions = gen(1)
            source_disc = torch.stack([source(grid) for source in sources])  # (B, H, W, 1)
            sol_disc = torch.stack([sol(grid) for sol in solutions])  # (B, H, W, 1)
            pred = model(source_disc)
            loss = loss_fn(pred, sol_disc)
            losses.append(loss.item())
        losses = torch.tensor(losses)

        print('Average loss:', losses.mean().item())
        print('Standard deviation:', losses.std().item())

        source, solution = gen(1)
        solution_pred = ol.GridFunction(
            model(source[0](grid)[None])[0].detach(), x=grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=gen.a, x_max=gen.b
        )
        source[0].quick_visualize()
        solution[0].quick_visualize()
        solution_pred.quick_visualize()
