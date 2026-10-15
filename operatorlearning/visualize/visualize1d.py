import mlx
import matplotlib.pyplot as plt
import os
from operatorlearning.data import OLDatasetLibrary, OLDataset


def add_gaps(u, x, min_gap):
    u_new, x_new = [], []
    for i in range(len(u) - 1):
        u_new.append(float(u[i]))
        x_new.append(float(x[i]))
        if abs(u[i + 1] - u[i]) >= min_gap:
            u_new.append(float('nan'))
            x_new.append(float('nan'))
    
    u_new.append(float(u[-1]))
    x_new.append(float(x[-1]))
    
    return u_new, x_new


def visualize_dataset(config, name, lib, dataset_id):
    data = OLDataset(lib.dataset_path(config['split'], dataset_id, config.get('resolution')))

    for i in mlx.subset_indices(config, data):
        fig, axes = plt.subplots(
            1, 2,
            figsize=config.get('figure_size', [8, 4]),
            sharey=config.get('share_y_axis', False)
        )

        u, x, v, y = data[i]
        axes[0].set_title(config.get('input_title', 'Input function $u$'))
        for j in range(u.shape[1]):
            if 'input_style' in config:
                style = config.get('input_style')[j]
            else:
                style = {}
            
            if 'min_gap' in config:
                u_, x_ = add_gaps(u[:, j], x[:, 0], config['min_gap'])
            else:
                u_, x_ = u[:, j], x[:, 0]
            
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
                v_, y_ = v[:, j], y[:, 0]
            
            axes[1].plot(y_, v_, **style, label=f'$v_{{{j + 1}}}$')

        if config.get('output_legend', True) and v.shape[1] > 1:
            axes[1].legend()

        if 'resolution' in config:
            file_name = f'{config["split"]}@{config["resolution"]}-{i}'
        else:
            file_name = f'{config["split"]}-{i}'
            
        sub_path = os.path.join(config.get('library', ''), str(dataset_id))
        mlx.show_and_save(fig, file_name, config, name, sub_path)
        plt.close(fig)


@mlx.experiment
def visualize1d(config, name, group=None):
    mlx.configure_plotting(config)
    if 'libraries' in config:
        print(f'Visualizing {sum(map(len, config["dataset_ids"]))} datasets in {len(config["libraries"])} libraries')
        for library, dataset_ids in zip(config['libraries'], config['dataset_ids']):
            print(f'Visualizing {len(dataset_ids)} datasets in library {config["library"]}')
            lib = OLDatasetLibrary(library)
            for dataset_id in dataset_ids:
                print(f'Visualizing dataset {dataset_id}')
                visualize_dataset(config, name, lib, dataset_id)
                
    elif 'dataset_ids' in config:
        print(f'Visualizing {len(config["dataset_ids"])} datasets in one library ({config["library"]})')
        lib = OLDatasetLibrary(config['library'])
        for dataset_id in config['dataset_ids']:
            print(f'Visualizing dataset {dataset_id}')
            visualize_dataset(config, name, lib, dataset_id)
            
    else:
        print(f'Visualizing one dataset (id={config["dataset_id"]}) in one library ({config["library"]})')
        visualize_dataset(config, name, OLDatasetLibrary(config['library']), config['dataset_id'])
