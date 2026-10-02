import mlx
import matplotlib.pyplot as plt
import os
from operatorlearning.data import OLDatasetLibrary, OLDataset


@mlx.experiment
def visualize1d(config, name, group=None):
    mlx.configure_plotting(config)
    lib = OLDatasetLibrary(config.get('library', name))
    data = OLDataset(lib.dataset_path(config['split'], config['dataset_id'], config.get('resolution')))

    for i in mlx.subset_indices(config, data):
        fig, axes = plt.subplots(1, 2, figsize=config.get('figure_size', [8, 4]))

        u, x, v, y = data[i]
        axes[0].set_title(config.get('input_title', 'Input function $u$'))
        for j in range(u.shape[1]):
            if 'input_style' in config:
                style = config.get('input_style')[j]
            else:
                style = {}
            axes[0].plot(x[:, 0], u[:, j], **style, label=f'$u_{{{j + 1}}}$')

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
            axes[1].plot(y[:, 0], v[:, j], **style, label=f'$v_{{{j + 1}}}$')

        if config.get('output_legend', True) and v.shape[1] > 1:
            axes[1].legend()

        if 'resolution' in config:
            file_name = f'{config["split"]}@{config["resolution"]}-{i}'
        else:
            file_name = f'{config["split"]}-{i}'
            
        sub_path = os.path.join(config.get('library', ''), str(config['dataset_id']))
        mlx.show_and_save(fig, file_name, config, name, sub_path)
