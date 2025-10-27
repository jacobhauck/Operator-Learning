import torch
import torch.utils.data
import mlx


class TwoStepDeepONet(torch.nn.Module):
    def __init__(self, num_branches, deeponet):
        """
        :param num_branches: Number of basis functions in the DeepONet (p)
        :param deeponet: Config for underlying DeepONet
        """
        super().__init__()
        self.deeponet = mlx.create_module(deeponet)

        mat_size = (num_branches, num_branches)
        self.register_buffer('t_matrix', torch.randn(mat_size))

    def forward(self, u, x_out):
        """
        :param u: (B, *in_shape, u_d_out) sample values of a batch of input
            functions
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the output function.
        :return: (B, *out_shape, v_d_out)
        """
        branch = self.deeponet.branch_net(u) # (B, p)
        trunk = self.deeponet.trunk_net(x_out)  # (B, *out_shape, p, v_d_out)

        branch_proj = torch.einsum('qp,bp->bq', self.t_matrix, branch)
        # (B, p)

        pre_bias = torch.einsum('b...pd,bp->b...d', trunk, branch_proj)
        # (B, *out_shape, v_d_out)

        return self.deeponet.add_bias(pre_bias)

    def step_one(self, a, x_out):
        """
        Prediction for step 1 training (trunk)

        :param a: (p, B) columns of the A matrix corresponding to the given
            batch of inputs
        :param x_out: (B, *out_shape, v_d_in) coordinates of points at which
            to sample the output function.
        :return: (B, *out_shape, v_d_out)
        """
        trunk = self.deeponet.trunk_net(x_out)  # (B, *out_shape, p, v_d_out)

        pre_bias = torch.einsum('pb,b...pd->b...d', a, trunk)
        # (B, *out_shape, v_d_out)

        result = self.deeponet.add_bias(pre_bias)
        # (B, *out_shape, v_d_out)

        return result

    def step_two(self, u):
        """
        Prediction for step 2 of training (branch)

        :param u: (B, *in_shape, v_d_in) Sample values of a batch of input
            functions
        :return: (B, p) Branch net prediction for the given input function
        """
        return self.deeponet.branch_net(u)  # (B, p)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        """
        Dataset that returns sample indices as the first variable in addition
        to all other variables.
        :param base_dataset: Dataset to wrap with index return
        """
        self.base_dataset = base_dataset

    def __getitem__(self, item):
        base_data = self.base_dataset[item]
        if isinstance(base_data, tuple):
            return item, *self.base_dataset[item]
        else:
            return item, base_data

    def __len__(self):
        return len(self.base_dataset)


class TrainingHelper(torch.nn.Module):
    def __init__(self, dataset_size, two_step_deeponet):
        """
        Class for storing temporary training state used in the two-step
        training process.
        :param dataset_size: Size of training dataset
        :param two_step_deeponet: TwoStepDeepONet module to be trained
        """
        super().__init__()
        # Use a list to store a reference without registering its parameters
        # We can then use a property to pretend like this hack isn't here...
        self._two_step_deeponet = [two_step_deeponet]

        num_branches = self.two_step_deeponet.t_matrix.shape[0]
        self.a_matrix = torch.nn.Parameter(torch.randn((num_branches, dataset_size)))
        self.register_buffer('target_matrix', torch.randn_like(self.a_matrix))

    @property
    def two_step_deeponet(self):
        return self._two_step_deeponet[0]

    def set_target_matrix(self, x_in):
        """
        Using the current trunk net and A matrix, set the target matrix for
        step 2 of training, and set the T matrix of the two-step DeepONet for
        use in inference.

        :param x_in: (*in_shape, v_d_in) Coordinates of sample points to use
            for orthogonalization
        """
        with torch.no_grad():
            x_in = x_in.view(-1, x_in.shape[-1])  # (m, v_d_in)

            # Approximate trunk basis on the given sample points
            trunk = self.two_step_deeponet.deeponet.trunk_net(x_in)  # (m, p, v_d_out)
            big_matrix = torch.permute(trunk, (0, 2, 1)).reshape(-1, trunk.shape[1])
            # (m*v_d_out, p)

            # Orthogonalize trunk basis approximately using sample values
            _, r = torch.linalg.qr(big_matrix)  # (p, p)
            r_inv = torch.linalg.inv(r)  # (p, p)
            self.target_matrix[:] = r @ self.a_matrix
            self.two_step_deeponet.t_matrix[:] = r_inv

    def forward(self, indices, step=1):
        """
        Get columns of A matrix for input to TwoStepDeepONet.step_one or
        the columns of the target matrix for training in step 2.

        :param indices: (B) indices of data in a batch
        :param step: Which step to get values for
        :return: (p, B) corresponding columns of A matrix if step == 1 or
            corresponding columns of the target matrix if step == 2
        """
        if step == 1:
            return self.a_matrix[:, indices]
        elif step == 2:
            return self.target_matrix[:, indices]
        else:
            raise ValueError("Invalid step. May only be 1 or 2.")
