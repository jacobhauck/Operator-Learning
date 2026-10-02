import mlx
import matplotlib.pyplot as plt
import os
from operatorlearning.data import OLDatasetLibrary, OLDataset


@mlx.experiment
def visualize2d(config, name, group=None):
    mlx.configure_plotting(config)
    lib = OLDatasetLibrary(config.get('library', name))
    data = OLDataset(lib.dataset_path(config['split'], config['dataset_id'], config.get('resolution')))

    for i in mlx.subset_indices(config, data):
        u, x, v, y = data[i]
        fig, axes = plt.subplots(2, max(u.shape[2], v.shape[2]), figsize=config.get('figure_size', [8, 4]), squeeze=False)

        for j in range(axes.shape[1]):
            axes[0][j].set_axis_off()
            axes[1][j].set_axis_off()

        for j in range(u.shape[2]):
            style = {
                'cmap': 'seismic',
                'origin': 'lower',
                'vmin': -u[:, j].abs().max(),
                'vmax': u[:, j].abs().max()
            }
            if 'input_style' in config:
                style.update(config.get('input_style')[j])

            axes[0][j].set_title(config.get('input_title', f'$u_{{{j + 1}}}$'))
            axes[0][j].imshow(u[:, :, j].T, **style)

        for j in range(v.shape[2]):
            style = {
                'cmap': 'seismic',
                'origin': 'lower',
                'vmin': -v[:, j].abs().max(),
                'vmax': v[:, j].abs().max()
            }
            if 'output_style' in config:
                style.update(config.get('output_style')[j])

            axes[1][j].set_title(config.get('output_title', f'$v_{{{j + 1}}}$'))
            axes[1][j].imshow(v[:, :, j].T, **style)

        if 'resolution' in config:
            file_name = f'{config["split"]}@{config["resolution"]}-{i}'
        else:
            file_name = f'{config["split"]}-{i}'

        sub_path = os.path.join(config.get('library', ''), str(config['dataset_id']))
        mlx.show_and_save(fig, file_name, config, name, sub_path)
