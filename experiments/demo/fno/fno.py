import experiments
import models
import torch
import data.synthetic.poisson
import operatorlearning as ol


class FNODemoExperiment(experiments.Experiment):
    def run(self, config, name, group=None):
        model = models.create_model(config['model'])
        optim = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

        source_gen = data.synthetic.poisson.DenseSourceGenerator(
            [6, 6], lambda k: 3/(1.0 + k[:, 0:1]**2 + k[:, 1:2]**2)
        )
        gen = data.synthetic.poisson.PoissonDataGenerator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            source_gen
        )
        loss_fn = torch.nn.MSELoss()

        grid = ol.GridFunction.uniform_x(gen.a, gen.b, num=128)[:-1, :-1]

        for i in range(config['training']['iterations']):
            sources, solutions = gen(config['training']['batch_size'])
            in_batch = torch.stack([source(grid) for source in sources])  # (B, H, W, 1)
            out_batch = torch.stack([sol(grid) for sol in solutions])  # (B, H, W, 1)

            optim.zero_grad()
            pred_batch = model(in_batch)
            loss = loss_fn(pred_batch, out_batch) / loss_fn(out_batch, torch.zeros_like(out_batch))
            loss.backward()
            optim.step()

            print(f'Batch {i}: loss = {loss.item():.06f}')

        source, solution = gen(1)
        solution_pred = ol.GridFunction(
            model(source[0](grid)[None])[0].detach(), x=grid,
            interpolator=ol.GridInterpolator(extend='periodic'),
            x_min=gen.a, x_max=gen.b
        )
        source[0].quick_visualize()
        solution[0].quick_visualize()
        solution_pred.quick_visualize()
