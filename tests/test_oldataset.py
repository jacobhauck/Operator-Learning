import torch
import operatorlearning.data as old


def test_write():
    u = torch.randn(5, 3, 3, 1)
    x1 = torch.randn(3, 3, 2)
    x2 = torch.randn(3, 3, 2)

    v = torch.randn(5, 4, 2, 3)
    y1 = torch.randn(4, 2, 4)
    y2 = torch.randn(4, 2, 4)
    y3 = torch.randn(4, 2, 4)

    old.OLDataset.write(
        u, [x1, x2],
        v, [y1, y2, y3],
        'test.ol.hdf5',
        u_disc=torch.tensor([0, 1, 0, 0, 1]),
        v_disc=torch.tensor([2, 0, 2, 1, 1])
    )


def test_read():
    u = torch.randn(5, 3, 3, 1)
    x1 = torch.randn(3, 3, 2)
    x2 = torch.randn(4, 4, 2)

    v = torch.randn(5, 4, 2, 3)
    y1 = torch.randn(4, 2, 4)
    y2 = torch.randn(4, 2, 4)
    y3 = torch.randn(4, 2, 4)

    x = [x1, x2]
    y = [y1, y2, y3]
    u_disc = torch.tensor([0, 1, 0, 0, 1])
    v_disc = torch.tensor([2, 0, 2, 1, 1])
    old.OLDataset.write(
        u, x,
        v, y,
        'test.ol.hdf5',
        u_disc=u_disc,
        v_disc=v_disc
    )

    read_in = old.OLDataset('test.ol.hdf5')

    for i, (ui, xi, vi, yi) in enumerate(read_in):
        assert (u[i] == ui).all()
        assert (v[i] == vi).all()
        assert (x[u_disc[i]] == xi).all()
        assert (y[v_disc[i]] == yi).all()
