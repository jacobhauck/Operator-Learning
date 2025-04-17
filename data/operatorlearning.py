import torch.utils.data


class SameRepresentationBatchSampler:
    def __init__(self, representations, batch_size: int, drop_last: bool = False):
        """
        :param representations: (N) tensor of discrete representation IDs,
            parallel to the dataset to be sampled
        :param batch_size: desired size of batches
        :param drop_last: whether to drop batches for which not enough samples
            are available to achieve the desired batch size
        """
        self.batch_size = batch_size

        _, i_bins, self.bin_sizes = torch.unique(
            representations, return_inverse=True, return_counts=True
        )
        self.original_indices = torch.argsort(i_bins)

        self.offsets = torch.cumsum(self.bin_sizes, 0)
        self.offsets[1:] = self.offsets[:-1].clone()
        self.offsets[0] = 0

        if drop_last:
            self.num_batches = self.bin_sizes // batch_size
        else:
            self.num_batches = (self.bin_sizes + batch_size - 1) // batch_size

    def __len__(self):
        return int(self.num_batches.sum())

    def __iter__(self):
        if len(self) == 0:
            return

        batch_bin_indices = torch.repeat_interleave(
            torch.arange(len(self.bin_sizes), device=self.bin_sizes.device),
            self.num_batches
        )[torch.randperm(len(self))]

        bin_reorders = [
            torch.randperm(size, device=self.bin_sizes.device)
            for size in self.bin_sizes
        ]

        current_bin_offsets = torch.zeros_like(self.offsets)
        for i_bin in batch_bin_indices:
            end = min(current_bin_offsets[i_bin] + self.batch_size, self.bin_sizes[i_bin])
            bin_indices = bin_reorders[i_bin][current_bin_offsets[i_bin]: end]
            current_bin_offsets[i_bin] = end

            yield self.original_indices[self.offsets[i_bin] + bin_indices]
