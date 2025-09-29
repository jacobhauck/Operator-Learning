import mlx
import operatorlearning.modules.deeponet
import torch
import matplotlib.pyplot as plt


class FourierFeatureTestExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        expansion1d = operatorlearning.modules.deeponet.FourierFeatureExpansion(
            origin=config['origin_1d'],
            scale=config['scale_1d'],
            features=config['features_1d'],
            mode='full'
        )

        x = torch.linspace(
            config['origin_1d'][0],
            config['origin_1d'][0] + config['scale_1d'][0],
            500
        )

        features = expansion1d(x[:, None])
        for i in range(features.shape[-1]):
            plt.plot(x, features[:, i])
        plt.show()

        expansion1d_rand = operatorlearning.modules.deeponet.FourierFeatureExpansion(
            origin=config['origin_1d'],
            scale=config['scale_1d'],
            features=config['features_rand_1d'],
            mode='random'
        )

        features = expansion1d_rand(x[:, None])
        for i in range(features.shape[-1]):
            plt.plot(x, features[:, i])
        plt.show()

        x = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1)
        expansion2d = operatorlearning.modules.deeponet.FourierFeatureExpansion(
            origin=config['origin_2d'],
            scale=config['scale_2d'],
            features=config['features_2d'],
            mode='full'
        )

        features = expansion2d(x)
        rows = 4
        cols = features.shape[-1] // 4
        fig, axes = plt.subplots(rows + 1, cols)
        i = 0
        for row in range(rows + 1):
            for col in range(cols):
                if i < features.shape[-1]:
                    axes[row][col].imshow(features[:, :, i])
                    i += 1
        plt.show()

        expansion2d_rand = operatorlearning.modules.deeponet.FourierFeatureExpansion(
            origin=config['origin_2d'],
            scale=config['scale_2d'],
            features=config['features_rand_2d'],
            mode='random'
        )

        features = expansion2d_rand(x)
        rows = 4
        cols = features.shape[-1] // 4
        fig, axes = plt.subplots(rows + 1, cols)
        i = 0
        for row in range(rows + 1):
            for col in range(cols):
                if i < features.shape[-1]:
                    axes[row][col].imshow(features[:, :, i])
                    i += 1
        plt.show()
