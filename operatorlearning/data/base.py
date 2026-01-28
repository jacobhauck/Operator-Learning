import torch
import torch.utils.data
import numpy as np
import h5py


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


class OLDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, stream_uv=True, stream_xy=False):
        self.file_name = file_name
        file = h5py.File(file_name)

        assert 'x' in file.keys(), 'Invalid dataset file: no group named "x"'
        self.x = None if stream_xy else {}
        self.x_keys = {}
        x_group = file['x']
        for key in x_group.keys():
            x_id = int(x_group[key].attrs['id'])
            self.x_keys[x_id] = key
            if not stream_xy:
                self.x[x_id] = torch.from_numpy(x_group[key][()])

        assert 'y' in file.keys(), 'Invalid dataset file: no group named "y"'
        self.y = None if stream_xy else {}
        self.y_keys = {}
        y_group = file['y']
        for key in y_group.keys():
            y_id = int(y_group[key].attrs['id'])
            self.y_keys[y_id] = key
            if not stream_xy:
                self.y[y_id] = torch.from_numpy(y_group[key][()])

        assert 'u' in file.keys(), 'Invalid dataset file: no group named "u"'
        self.u = None if stream_uv else {}
        self.u_disc_ids = {}
        u_keys = []
        u_indices = []
        u_indices_rel = []
        u_group = file['u']
        for key in u_group.keys():
            self.u_disc_ids[key] = int(u_group[key].attrs['disc_id'])
            u_subgroup = u_group[key]
            indices = torch.from_numpy(u_subgroup['indices'][()])
            u_indices.append(indices)
            u_indices_rel.extend(list(range(len(indices))))
            u_keys.extend([key] * len(indices))

            if not stream_uv:
                self.u[key] = torch.from_numpy(u_subgroup['u'][()])

        u_indices = torch.cat(u_indices)
        u_indices_rel = torch.tensor(u_indices_rel).to(torch.long)
        u_order = u_indices.argsort()

        assert 'v' in file.keys(), 'Invalid dataset file: no group named "v"'
        self.v = None if stream_uv else {}
        self.v_disc_ids = {}
        v_keys = []
        v_indices = []
        v_indices_rel = []
        v_group = file['v']
        for key in v_group.keys():
            self.v_disc_ids[key] = int(v_group[key].attrs['disc_id'])
            v_subgroup = v_group[key]
            indices = torch.from_numpy(v_subgroup['indices'][()])
            v_indices.append(indices)
            v_indices_rel.extend(list(range(len(indices))))
            v_keys.extend([key] * len(indices))

            if not stream_uv:
                self.v[key] = torch.from_numpy(v_subgroup['v'][()])

        v_indices = torch.cat(v_indices)
        v_indices_rel = torch.tensor(v_indices_rel).to(torch.long)
        v_order = v_indices.argsort()

        assert (u_indices[u_order] == v_indices[v_order]).all(), \
            'Invalid dataset file: indices of u and v could not be matched'

        self.u_keys = [u_keys[int(i)] for i in u_order]
        self.u_indices = u_indices_rel[u_order].numpy().tolist()
        self.v_keys = [v_keys[int(i)] for i in v_order]
        self.v_indices = v_indices_rel[v_order].numpy().tolist()
        
        file.close()
        self.file = None
    
    def __len__(self):
        return len(self.u_indices)

    def __getitem__(self, index):
        u_key, u_index = self.u_keys[index], self.u_indices[index]
        v_key, v_index = self.v_keys[index], self.v_indices[index]

        u_disc_id = self.u_disc_ids[u_key]
        v_disc_id = self.v_disc_ids[v_key]

        if self.x is None:
            if self.file is None:
                self.file = h5py.File(self.file_name)
            
            x = self.file['x'][self.x_keys[u_disc_id]][:].copy()
            x = torch.from_numpy(x)
            
            y = self.file['y'][self.y_keys[v_disc_id]][:].copy()
            y = torch.from_numpy(y)
        else:
            x = self.x[u_disc_id]
            y = self.y[v_disc_id]

        if self.u is None:
            if self.file is None:
                self.file = h5py.File(self.file_name)
            
            u = self.file['u'][u_key]['u'][u_index].copy()
            u = torch.from_numpy(u)
            
            v = self.file['v'][v_key]['v'][v_index].copy()
            v = torch.from_numpy(v)
        else:
            u = self.u[u_key][u_index]
            v = self.v[v_key][v_index]

        return u, x, v, y

    @staticmethod
    def write(u, x, v, y, file_name, u_disc=None, v_disc=None):
        """
        Writes a single-I/O dataset to an OLDataset-compatible HDF5 file.
        :param u: Length-N list of (*, u_d_out) values of input functions at
            input sampling points
        :param x: List of (*, u_d_in) arrays of input sample point coordinates.
            Interpretation depends on whether u_disc is provided
        :param v: Length-N list of (*, v_d_out) values of output functions at
            output sampling points. Parallel list with u
        :param y: List of (*, v_d_in) arrays of output sample point coordinates.
            Interpretation depends on whether v_disc is provided
        :param u_disc: Optional length-N list of indices into x of the sampling
            points used for each input function in u. Parallel list with u. If
            not provided, then x must be length-N and is assumed to be parallel
            with u. Default: None (same as unspecified)
        :param file_name: Name of output file to write to
        :param v_disc: Optional length-N list of indices into y of the sampling
            points used for each output function in v. Parallel list with v. If
            not provided, then y must be length-N and is assumed to be parallel
            with v. Default: None (same as unspecified)
        """
        # Infer data type for all data arrays (not indices)
        dtype = u[0].numpy().dtype

        if u_disc is None:
            assert len(u) == len(x), 'Invalid size of x or missing u_disc'
            u_disc = torch.arange(len(x))

        u_group_indices, u_groups, u_group_sizes = torch.unique(
            u_disc,
            return_inverse=True,
            return_counts=True
        )

        if v_disc is None:
            assert len(v) == len(y), 'Invalid size of y or missing v_disc'
            v_disc = torch.arange(len(y))

        v_group_indices, v_groups, v_group_sizes = torch.unique(
            v_disc,
            return_inverse=True,
            return_counts=True
        )

        with h5py.File(file_name, 'w') as f:
            x_group = f.create_group('x')
            for i_group in range(len(x)):
                x_group[str(i_group)] = x[i_group].numpy().astype(dtype)
                x_group[str(i_group)].attrs['id'] = i_group

            y_group = f.create_group('y')
            for i_group in range(len(y)):
                y_group[str(i_group)] = y[i_group].numpy().astype(dtype)
                y_group[str(i_group)].attrs['id'] = i_group

            u_group = f.create_group('u')
            for i_group, group_size in zip(u_group_indices, u_group_sizes):
                u_subgroup = u_group.create_group(str(int(i_group)))
                u_subgroup.attrs['disc_id'] = int(i_group)
                u_subgroup.create_dataset('indices', shape=(group_size,), dtype=int)

            added_by_subgroup = {int(i_group): 0 for i_group in u_group_indices}
            for i, i_group in enumerate(u_groups):
                u_subgroup = u_group[str(int(i_group))]
                if 'u' not in u_subgroup.keys():
                    shape = (int(u_group_sizes[i_group]),) + u[i].shape
                    u_subgroup.create_dataset('u', shape=shape, dtype=dtype)

                rel_index = added_by_subgroup[int(i_group)]
                u_subgroup['indices'][rel_index] = i
                u_subgroup['u'][rel_index] = u[i].numpy()
                added_by_subgroup[int(i_group)] += 1

            v_group = f.create_group('v')
            for i_group, group_size in zip(v_group_indices, v_group_sizes):
                v_subgroup = v_group.create_group(str(int(i_group)))
                v_subgroup.attrs['disc_id'] = int(i_group)
                v_subgroup.create_dataset('indices', shape=(group_size,), dtype=int)

            added_by_subgroup = {int(i_group): 0 for i_group in v_group_indices}
            for i, i_group in enumerate(v_groups):
                v_subgroup = v_group[str(int(i_group))]
                if 'v' not in v_subgroup.keys():
                    shape = (int(v_group_sizes[i_group]),) + v[i].shape
                    v_subgroup.create_dataset('v', shape=shape, dtype=dtype)

                rel_index = added_by_subgroup[int(i_group)]
                v_subgroup['indices'][rel_index] = i
                v_subgroup['v'][rel_index] = v[i].numpy()
                added_by_subgroup[int(i_group)] += 1
