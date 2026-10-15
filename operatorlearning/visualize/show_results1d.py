import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mlx

from .visualize1d import add_gaps


class VisualizationInterface(ABC):
    @abstractmethod
    def get_data(self):
        raise NotImplemented

    @abstractmethod
    def predict(self, u, x, y):
        raise NotImplemented


def show_results1d(config, name, run_id, data, data_library, dataset_id, predictor):
    for i in mlx.subset_indices(config, data):
        fig, axes = plt.subplots(
            1, 3,
            figsize=config.get('figure_size', [8, 4]),
            sharey=config.get('share_y_axis', False)
        )

        u, x, v, y = data[i]
        v_pred = predictor(u, x, y)

        axes[0].set_title(config.get('input_title', 'Input function $u$'))
        for j in range(u.shape[1]):
            if 'input_style' in config:
                style = config.get('input_style')[j]
            else:
                style = {}

            if 'min_gap' in config:
                u_, x_ = add_gaps(u[:, j], x[:, 0], config['min_gap'])
            else:
                u_, x_ = x[:, 0], u[:, j]

            axes[0].plot(x_, u_, **style, label=f'$u_{{{j + 1}}}$')

        if config.get('input_legend', True) and u.shape[1] > 1:
            axes[0].legend()

        axes[0].set_xlabel(config.get('input_x_label', '$x$'))
        axes[0].set_ylabel(config.get('input_y_label', '$y$'))

        axes[1].set_title(config.get('output_title', 'Output function $v$'))
        for j in range(v.shape[1]):
            if 'output_style' in config:
                style = config.get('output_style')[j]
            else:
                style = {}

            if 'min_gap' in config:
                v_, y_ = add_gaps(v[:, j], y[:, 0], config['min_gap'])
            else:
                v_, y_ = y[:, 0], v[:, j]

            axes[1].plot(y_, v_, **style, label=f'$v_{{{j + 1}}}$')

        if config.get('output_legend', True) and v.shape[1] > 1:
            axes[1].legend()

        axes[1].set_xlabel(config.get('output_x_label', '$x$'))
        axes[1].set_ylabel(config.get('output_y_label', '$y$'))

        axes[2].set_title(config.get('pred_output_title', r'Predicted output $\hat{v}$'))
        for j in range(v_pred.shape[1]):
            if 'output_style' in config:
                style = config.get('output_style')[j]
            else:
                style = {}

            if 'min_gap' in config:
                v_pred_, y_ = add_gaps(v_pred[:, j], y[:, 0], config['min_gap'])
            else:
                v_pred_, y_ = y[:, 0], v_pred[:, j]

            axes[2].plot(y_, v_pred_, **style, label=f'$\\hat{{v}}_{{{j + 1}}}$')

        if config.get('output_legend', True) and v_pred.shape[1] > 1:
            axes[2].legend()

        axes[2].set_xlabel(config.get('pred_output_x_label', '$x$'))
        axes[2].set_ylabel(config.get('pred_output_y_label', '$y$'))

        if 'resolution' in config:
            file_name = f'{config["split"]}@{config["resolution"]}-{i}'
        else:
            file_name = f'{config["split"]}-{i}'

        run_folder = config['model']['name'] + '-' + run_id
        sub_path = os.path.join(data_library, str(dataset_id), run_folder)
        mlx.show_and_save(fig, file_name, config, name, sub_path)
        plt.close(fig)

