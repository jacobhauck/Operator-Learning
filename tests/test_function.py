import torch
import pytest
import operatorlearning as ol
import matplotlib.pyplot as plt


@pytest.mark.parametrize('d', [1, 2])
def test_scalar(d):
    if d == 1:
        x = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])
        y = torch.sin(x)
        base_f = ol.GridFunction(y, x[:, None])

        x_rearrange = x[torch.randperm(len(x))]
        test_sort = ol.GridFunction(y, xs=[x_rearrange], is_sorted=False)

        # noinspection PyTypeChecker
        assert torch.all(test_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.xs[0] == base_f.xs[0])

        test_no_sort = ol.GridFunction(y, xs=[x], is_sorted=True)

        # noinspection PyTypeChecker
        assert torch.all(test_no_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_no_sort.xs[0] == base_f.xs[0])

        test_both = ol.GridFunction(y, x[:, None], xs=[x])

        # noinspection PyTypeChecker
        assert torch.all(test_both.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_both.xs[0] == base_f.xs[0])

        x_inter = torch.linspace(-.5, 1.5, 200)

        for method in ('linear', 'nearest'):
            for extend in ('clamped', 'periodic', 0.5):
                base_f.interpolator.method = method
                base_f.interpolator.extend = extend
                y_inter = base_f(x_inter[:, None])
                plt.plot(x_inter, y_inter)
                plt.title(f'{extend} {method} interpolation')
                plt.show()

    elif d == 2:
        x1 = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])
        x2 = torch.tensor([0.0, 0.1, 0.6, 0.7, 0.8, 1.0])
        x = torch.stack(torch.meshgrid([x1, x2], indexing='ij'), dim=-1)
        y = torch.sin(x[..., 0]) * torch.cos(x[..., 1])
        base_f = ol.GridFunction(y, xs=[x1, x2], is_sorted=True)

        x1_rearrange = x1[torch.randperm(len(x1))]
        x2_rearrange = x2[torch.randperm(len(x2))]
        test_sort = ol.GridFunction(y, xs=[x1_rearrange, x2_rearrange], is_sorted=False)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.xs[0] == base_f.xs[0])

        test_meshgrid = ol.GridFunction(y, x=x)
        # noinspection PyTypeChecker
        assert torch.all(test_meshgrid.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_meshgrid.xs[0] == base_f.xs[0])

        test_both = ol.GridFunction(y, x=x, xs=[x1, x2], is_sorted=True)
        # noinspection PyTypeChecker
        assert torch.all(test_both.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_both.xs[0] == base_f.xs[0])

        x1_inter = torch.linspace(-.5, 1.5, 200)
        x2_inter = torch.linspace(-.4, 1.6, 300)
        im_extent = ( -.4 - 0.5/300, 1.6 + 0.5/300, -.5 - 0.5/200, 1.5 + 0.5/200)
        x_inter = torch.stack(torch.meshgrid([x1_inter, x2_inter], indexing='ij'), dim=-1)

        for method in ('linear', 'nearest'):
            for extend in ('clamped', 'periodic', 0.5):
                base_f.interpolator.method = method
                base_f.interpolator.extend = extend
                y_inter = base_f(x_inter)
                plt.imshow(y_inter, origin='lower', extent=im_extent)
                plt.title(f'{extend} {method} interpolation')
                plt.show()


@pytest.mark.parametrize('d', [1, 2])
def test_vector(d):
    if d == 1:
        x = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])
        y = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        base_f = ol.GridFunction(y, x[:, None])

        x_rearrange = x[torch.randperm(len(x))]
        test_sort = ol.GridFunction(y, xs=[x_rearrange], is_sorted=False)

        # noinspection PyTypeChecker
        assert torch.all(test_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.xs[0] == base_f.xs[0])

        test_no_sort = ol.GridFunction(y, xs=[x], is_sorted=True)

        # noinspection PyTypeChecker
        assert torch.all(test_no_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_no_sort.xs[0] == base_f.xs[0])

        test_both = ol.GridFunction(y, x[:, None], xs=[x])

        # noinspection PyTypeChecker
        assert torch.all(test_both.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_both.xs[0] == base_f.xs[0])

        x_inter = torch.linspace(-.5, 1.5, 200)

        for method in ('linear', 'nearest'):
            for extend in ('clamped', 'periodic', torch.tensor([[0.2, 0.7]])):
                base_f.interpolator.method = method
                base_f.interpolator.extend = extend
                y_inter = base_f(x_inter[:, None])
                plt.plot(x_inter, y_inter[..., 0])
                plt.plot(x_inter, y_inter[..., 1])
                plt.title(f'{extend} {method} interpolation')
                plt.show()

    elif d == 2:
        x1 = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])
        x2 = torch.tensor([0.0, 0.1, 0.6, 0.7, 0.8, 1.0])
        x = torch.stack(torch.meshgrid([x1, x2], indexing='ij'), dim=-1)
        y = torch.stack([
            torch.sin(x[..., 0]) * torch.cos(x[..., 1]),
            torch.cos(x[..., 0]) * torch.sin(x[..., 1])
        ], dim=-1)
        base_f = ol.GridFunction(y, xs=[x1, x2], is_sorted=True)

        x1_rearrange = x1[torch.randperm(len(x1))]
        x2_rearrange = x2[torch.randperm(len(x2))]
        test_sort = ol.GridFunction(y, xs=[x1_rearrange, x2_rearrange], is_sorted=False)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_sort.xs[0] == base_f.xs[0])

        test_meshgrid = ol.GridFunction(y, x=x)
        # noinspection PyTypeChecker
        assert torch.all(test_meshgrid.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_meshgrid.xs[0] == base_f.xs[0])

        test_both = ol.GridFunction(y, x=x, xs=[x1, x2], is_sorted=True)
        # noinspection PyTypeChecker
        assert torch.all(test_both.x == base_f.x)
        # noinspection PyTypeChecker
        assert torch.all(test_both.xs[0] == base_f.xs[0])

        x1_inter = torch.linspace(-.5, 1.5, 200)
        x2_inter = torch.linspace(-.4, 1.6, 300)
        im_extent = ( -.4 - 0.5/300, 1.6 + 0.5/300, -.5 - 0.5/200, 1.5 + 0.5/200)
        x_inter = torch.stack(torch.meshgrid([x1_inter, x2_inter], indexing='ij'), dim=-1)

        for method in ('linear', 'nearest'):
            for extend in ('clamped', 'periodic', torch.tensor([[[0.2, 0.7]]])):
                base_f.interpolator.method = method
                base_f.interpolator.extend = extend
                y_inter = base_f(x_inter)
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(y_inter[..., 0], origin='lower', extent=im_extent)
                ax[1].imshow(y_inter[..., 1], origin='lower', extent=im_extent)
                ax[0].set_title(f'{extend} {method} interpolation')
                plt.show()
