import os

import mlx
import numpy as np
import torch
import operatorlearning as ol
from operatorlearning.data import OLDataset


class TestSubsampleExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        print('Test single-discretization dataset')
        x = ol.GridFunction.uniform_x(
            min_point=torch.tensor([0.0, 0.0]),
            max_point=torch.tensor([1.0, 1.0]),
            num=21
        )
        print('x shape (original):', x.shape)
        y = ol.GridFunction.uniform_x(
            min_point=torch.tensor([0.0]),
            max_point=torch.tensor([1.0]),
            num=9
        )
        print('y shape (original):', y.shape)
        u = torch.randn_like(x)
        print('u shape (original):', u.shape)
        v = torch.randn_like(y)
        print('v shape (original):', v.shape)

        print('x (original)')
        print(x)
        print('y (original)')
        print(y)
        print('u (original)')
        print(u)
        print('v (original)')
        print(v)

        OLDataset.write([u], [x], [v], [y], 'test.ol.h5')
        dataset = OLDataset('test.ol.h5')
        x_indices = torch.from_numpy(np.indices(x.shape[:-1]).transpose(1, 2, 0))
        x_indices = x_indices[::4, ::5]
        print('x_indices shape:', x_indices.shape)
        y_indices = torch.from_numpy(np.indices(y.shape[:-1]).T)
        y_indices = y_indices[::2]
        print('y_indices shape:', y_indices.shape)
        dataset.save_subsampled(x_indices, y_indices, 'test_sub.ol.h5')

        dataset_sub = OLDataset('test_sub.ol.h5')
        u_sub, x_sub, v_sub, y_sub = dataset_sub[0]
        print('x shape (sub):', x_sub.shape)
        print('y shape (sub):', y_sub.shape)
        print('u shape (sub):', u_sub.shape)
        print('v shape (sub):', v_sub.shape)
        print('x (sub)')
        print(x_sub)
        print('y (sub)')
        print(y_sub)
        print('u (sub)')
        print(u_sub)
        print('v (sub)')
        print(v_sub)

        dataset.file.close()
        dataset_sub.file.close()
        os.remove('test.ol.h5')
        os.remove('test_sub.ol.h5')

        print('Test multi-discretization dataset')
        x2 = ol.GridFunction.uniform_x(
            min_point=torch.tensor([0.0, 0.0]),
            max_point=torch.tensor([1.0, 1.0]),
            num=16
        )
        print('x2 shape (original):', x2.shape)
        y2 = ol.GridFunction.uniform_x(
            min_point=torch.tensor([0.0]),
            max_point=torch.tensor([1.0]),
            num=7
        )
        print('y2 shape (original):', y2.shape)
        u2 = torch.randn_like(x2)
        print('u shape (original):', u2.shape)
        v2 = torch.randn_like(y2)
        print('v shape (original):', v2.shape)

        print('x2 (original)')
        print(x2)
        print('y2 (original)')
        print(y2)
        print('u2 (original)')
        print(u2)
        print('v2 (original)')
        print(v2)

        OLDataset.write([u, u2], [x, x2], [v, v2], [y, y2], 'test.ol.h5')
        dataset = OLDataset('test.ol.h5')
        x2_indices = torch.from_numpy(np.indices(x2.shape[:-1]).transpose(1, 2, 0))
        x2_indices = x2_indices[::3, ::5]
        print('x2_indices shape:', x_indices.shape)
        y2_indices = torch.from_numpy(np.indices(y2.shape[:-1]).T)
        y2_indices = y2_indices[::3]
        print('y2_indices shape:', y2_indices.shape)
        x_indices = {0: x_indices, 1: x2_indices}
        y_indices = {0: y_indices, 1: y2_indices}
        dataset.save_subsampled(x_indices, y_indices, 'test_sub.ol.h5')

        dataset_sub = OLDataset('test_sub.ol.h5')
        u_sub, x_sub, v_sub, y_sub = dataset_sub[0]
        print('x shape (sub):', x_sub.shape)
        print('y shape (sub):', y_sub.shape)
        print('u shape (sub):', u_sub.shape)
        print('v shape (sub):', v_sub.shape)
        print('x (sub)')
        print(x_sub)
        print('y (sub)')
        print(y_sub)
        print('u (sub)')
        print(u_sub)
        print('v (sub)')
        print(v_sub)

        u2_sub, x2_sub, v2_sub, y2_sub = dataset_sub[1]
        print('x2 shape (sub):', x2_sub.shape)
        print('y2 shape (sub):', y2_sub.shape)
        print('u2 shape (sub):', u2_sub.shape)
        print('v2 shape (sub):', v2_sub.shape)
        print('x2 (sub)')
        print(x2_sub)
        print('y2 (sub)')
        print(y2_sub)
        print('u2 (sub)')
        print(u2_sub)
        print('v2 (sub)')
        print(v2_sub)

        dataset.file.close()
        dataset_sub.file.close()
        os.remove('test.ol.h5')
        os.remove('test_sub.ol.h5')
