import pytest
import data.operatorlearning as ol_data
import numpy as np
import torch


@pytest.mark.parametrize('num_reps', [1, 2, 4, 8])
@pytest.mark.parametrize('size', [10, 20, 100])
@pytest.mark.parametrize('batch_size', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('drop_last', [False, True])
def test(num_reps, size, batch_size, drop_last):
    representations = torch.from_numpy(
        np.random.choice(np.arange(num_reps, dtype=int), size=size, replace=True)
    )
    batch_sampler = ol_data.SameRepresentationBatchSampler(
        representations, batch_size=batch_size, drop_last=drop_last
    )
    seen = set()
    for batch in batch_sampler:
        # noinspection PyTypeChecker
        assert torch.all(representations[batch] == representations[batch[0]])

        seen.update(map(int, batch))

        if drop_last:  # drop_last should force all batches to have the desired size
            assert len(batch) == batch_size

    if not drop_last:
        assert seen == set(range(len(representations)))
    else:  # at least make sure we saw as many samples as possible
        assert len(seen) == len(batch_sampler) * batch_size
