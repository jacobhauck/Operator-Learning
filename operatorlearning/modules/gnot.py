"""
Implementation of GNOT
"""
import torch
import mlx


class GNOT(torch.nn.Module):
    def __init__(
            self,
            d_model,
            v_d_in,
            v_d_out,
            query_mlp_config,
            output_mlp_config,
            input_encoders,
            num_layers,
            layer_config
    ):
        """
        Implementation of General Neural Operator Transformer (GNOT).

        For now, geometric gating mechanism is not included.

        :param d_model: Model dimension
        :param v_d_in: Output function domain dimension
        :param v_d_out: Output function codomain dimension
        :param query_mlp_config: Config of MLP for query embedding
        :param output_mlp_config: Config of MLP for decoding output
        :param input_encoders: List of input encoder configs. d_model
            will be overridden.
        :param num_layers: Number of GNOT layers
        :param layer_config: Config for GNOT layers (GNOTLayer module)
        """
        super().__init__()
        self.input_encoders = torch.nn.ModuleList()
        for config in input_encoders:
            config['d_model'] = d_model
            self.input_encoders.append(mlx.create_module(config))

        layer_config = dict(layer_config)
        layer_config.update({'d_model': d_model, 'num_inputs': self.num_inputs})
        self.gnot_layers = torch.nn.ModuleList([
            GNOTLayer(**layer_config)
            for _ in range(num_layers)
        ])

        self.query_encoder = mlx.modules.MLP(
            d_in=v_d_in,
            d_out=d_model,
            **query_mlp_config
        )

        self.output_decoder = mlx.modules.MLP(
            d_in=d_model,
            d_out=v_d_out,
            **output_mlp_config
        )

    @property
    def num_inputs(self):
        return len(self.input_encoders)

    @property
    def input_types(self):
        for encoder in self.input_encoders:
            yield encoder.input_type

    def input_encoding(self, inputs):
        """
        :param inputs: Same as inputs to forward.
        """
        encodings = []
        for input_value, encoder in zip(inputs, self.input_encoders):
            encodings.append(encoder(*input_value))
        return encodings

    def query(self, input_encodings, y):
        """
        :param input_encodings: Output from self.input_encoding()
        :param y: (B, ..., v_d_in) Output function query point coordinates
        :return: (B, ..., v_d_out) Output function sample values at the query
            points
        """
        # Save original shape so we can reshape the output
        original_shape = y.shape

        x = self.query_encoder(y.reshape(y.shape[0], -1, y.shape[-1]))
        # (B, N, d_model)

        for gnot_layer in self.gnot_layers:
            x = gnot_layer(x, input_encodings)  # (B, N, d_model)

        return self.output_decoder(x).reshape(*original_shape[:-1], -1)
        # (B, ..., v_d_out)

    def forward(self, inputs, y):
        """
        :param inputs: List of inputs. Type depends on required input types
            specified in self.input_types. For 'vector' types, a batch of
            vectors (B, d_input) is required. For 'geometry' types, a tuple
            (geometries, mask) is required, where geometries is a batch of
            point sequences (B, N, d_in), and mask is a boolean tensor (B, N)
            indicating which points are valid (in case number of points per
            geometry in the batch differs we use masking in cross attention
            to ignore padded points). For 'function' types, a tuple (f, x, mask)
            is required, where f is the function values of shape (B, N, f_d_out),
            x is the sample coordinates of shape (B, N, f_d_in), and mask is
            a boolean tensor of shape (B, N) indicating which samples are valid
            (again, in case the number of points varies per input)
        :param y: (B, ..., v_d_in) Output function query point coordinates
        :return: (B, ..., v_d_out) Output function values at the query points
        """
        input_encodings = self.input_encoding(inputs)
        return self.query(input_encodings, y)


class VectorEncoder(torch.nn.Module):
    def __init__(self, d_input, d_model, mlp_config):
        """
        Vector-valued input encoder
        :param d_input: Input vector dimension
        :param d_model: Model embedding dimension
        :param mlp_config: Config of MLP (name, d_in, and d_out will be overridden)
        """
        super().__init__()
        mlp_config = dict(mlp_config)
        mlp_config.update({'name': 'MLP', 'd_in': d_input, 'd_out': d_model})
        self.mlp = mlx.create_module(mlp_config)

    @property
    def input_type(self):
        return 'vector'

    def forward(self, theta):
        """
        :param theta: (B, d_input) Input parameter vector
        :return: Tuple (embedding, mask), where embedding has shape
            (B, 1, d_model), and mask is None (this is to conform with the
            interface for input embeddings; for vector inputs, all components
            are always visible in all batches by assumption of the model)
        """
        return self.mlp(theta)[:, None, :], None


class SampledGeometryEncoder(torch.nn.Module):
    def __init__(self, d_geometry, d_model, mlp_config):
        """
        :param d_geometry: Dimension of geometry points
        :param d_model: Model embedding dimension
        :mlp_config: Pointwise MLP config (name, d_in and d_out will be overridden)
        """
        super().__init__()
        mlp_config = dict(mlp_config)
        mlp_config.update({'name': 'MLP', 'd_in': d_geometry, 'd_out': d_model})
        self.mlp = mlx.create_module(mlp_config)

    @property
    def input_type(self):
        return 'geometry'

    def forward(self, x, mask=None):
        """
        :param x: (B, ..., d_geometry) Input points describing the geometry
        :param mask: (B, ...) point mask that is forwarded to the cross attention
        :return: Tuple (embedding, mask), where embedding has shape
            (B, N, d_model) for N the number of elements in the ..., and mask is
            the given mask reshaped to (B, N)
        """
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if mask is not None:
            mask = mask.reshape(mask.shape[0], -1)
        return self.mlp(x), mask


class SampledFunctionEncoder(torch.nn.Module):
    def __init__(self, f_d_in, f_d_out, d_model, mlp_config):
        """
        :param f_d_in: Dimension of input vector to function
        :param f_d_out: Dimension of output vector from function
        :param d_model: Model embedding dimension
        :param mlp_config: Embedding MLP config (name, d_in, and d_out will be
            overridden)
        """
        super().__init__()
        mlp_config = dict(mlp_config)
        mlp_config.update({'name': 'MLP', 'd_in': f_d_in + f_d_out, 'd_out': d_model})
        self.mlp = mlx.create_module(mlp_config)

    @property
    def input_type(self):
        return 'function'

    def forward(self, f, x, mask=None):
        """
        :param f: (B, ..., f_d_out) Input function samples
        :param x: (B, ..., f_d_in) Input function sample point coordinates
        :param mask: (B, ...) Mask indicating which samples are valid. Forwarded
            to cross attention module after reshaping
        :return: Tuple (embeddings, mask), where embeddings has shape
            (B, N, d_model) for N the number of elements in the ...,
            and mask is the given mask reshaped to (B, N)
        """
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if mask is not None:
            mask.reshape(mask.shape[0], -1)
        return self.mlp(torch.cat([f, x], dim=-1)), mask


class GNOTLayer(torch.nn.Module):
    def __init__(
            self,
            d_model,
            d_hidden,
            num_heads,
            num_inputs,
            activation,
            dropout=None,
            attention_dropout=None
    ):
        """
        :param d_model: Model dimension
        :param d_hidden: Hidden dimension in MLPs
        :param num_heads: Number of heads in multihead self-attention and
            multihead cross attention
        :param num_inputs: Number of model inputs
        :param activation: Activation function config
        :param dropout: Optional dropout rate for this layer
        :param attention_dropout: Optional dropout rate in self-attention
            module
        """
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.input_layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for _ in range(num_inputs)
        ])
        self.ln3 = torch.nn.LayerNorm(d_model)
        self.ln4 = torch.nn.LayerNorm(d_model)
        self.ln5 = torch.nn.LayerNorm(d_model)

        self.self_attn = MultiHeadLinearAttention(d_model, num_heads, attention_dropout)
        self.cross_attn = HeterogeneousNormalizedAttention(d_model, num_heads, num_inputs)

        self.act = mlx.create_module(activation)

        if dropout is None:
            self.dropout1 = torch.nn.Identity()
            self.dropout2 = torch.nn.Identity()
        else:
            self.dropout1 = torch.nn.Dropout(dropout)
            self.dropout2 = torch.nn.Dropout(dropout)

        self.mlp1 = mlx.modules.MLP(d_model, [d_hidden], d_model, activation)
        self.mlp2 = mlx.modules.MLP(d_model, [d_hidden], d_model, activation)

    @property
    def num_inputs(self):
        return len(self.input_layer_norms)

    def normalize_inputs(self, y):
        return [
            (self.input_layer_norms[i](y[i][0]), y[i][1])
            for i in range(self.num_inputs)
        ]

    def forward(self, x, y):
        x = x + self.dropout1(self.cross_attn(self.ln1(x), self.normalize_inputs(y)))
        x = x + self.mlp1(self.ln3(x))

        x = x + self.dropout2(self.self_attn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))

        return x


class MultiHeadLinearAttention(torch.nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end
    """

    def __init__(self, d_model, num_heads, dropout=None, attention_type='l1'):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_type = attention_type

        # key, query, value projections for all heads
        self.key = torch.nn.Linear(d_model, d_model)
        self.query = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)

        if dropout is None:
            self.dropout = torch.nn.Identity()
        else:
            self.dropout = torch.nn.Dropout(dropout)

        # output projection
        self.output_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        :param x: (B, N, d_model) Input tokens
        :return: (B, N, d_model) Output tokens after linear self attention
        """
        b, n, d_model = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)

        if self.attention_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)   #
            k_sum = k.sum(dim=-2, keepdim=True)
            d_inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)

        elif self.attention_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            d_inv = 1. / n  # galerkin normalization

        elif self.attention_type == "l2":
            # still use l1 normalization
            q = q / q.norm(dim=-1,keepdim=True, p=1)
            k = k / k.norm(dim=-1,keepdim=True, p=1)
            k_sum = k.sum(dim=-2, keepdim=True)
            d_inv = 1. / (q * k_sum).abs().sum(dim=-1, keepdim=True)  # normalized

        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.dropout((q @ context) * d_inv + q)
        # (B, num_heads, N, head_dim)

        y = y.transpose(1, 2).reshape(b, n, d_model)

        y = self.output_proj(y)  # (B, N, d_model)

        return y


class HeterogeneousNormalizedAttention(torch.nn.Module):
    """
    Heterogeneous normalized attention as described in GNOT
    """

    def __init__(self, d_model, num_heads, num_inputs):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_inputs = num_inputs

        # key, query, value projections for all heads
        self.query = torch.nn.Linear(d_model, d_model)
        self.keys = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(num_inputs)])
        self.values = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(num_inputs)])

        # output projection
        self.output_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, y):
        b, n, d_model = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)

        q = q.softmax(dim=-1)
        out = q

        for i in range(self.num_inputs):
            embedding, mask = y[i]
            # (B, N_i, d_model), (B, N_i)

            _, n_i, _ = embedding.shape
            k = self.keys[i](embedding).view(b, n_i, self.num_heads, self.head_dim).transpose(1, 2)
            # (B, num_heads, N_i, head_dim)

            v = self.values[i](embedding).view(b, n_i, self.num_heads, self.head_dim).transpose(1, 2)
            # (B, num_heads, N_i, head_dim)

            # Normalize keys
            k = k.softmax(dim=-1)

            # Use mask to set invalid keys to 0
            #
            # Note that this does not make the above normalization incorrect
            # because the normalization is done along the model dimension, not
            # token position. Furthermore, this must be done after normalization
            # or the softmax along the model dimension would necessarily result
            # in some nonzero values that should be masked
            if mask is not None:
                k = k.masked_fill(~mask[:, :, None], 0.0)

            k_sum = k.sum(dim=-2, keepdim=True)
            # (B, num_heads, 1, head_dim)

            d_inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)  # normalized
            # (B, num_heads, N, 1)

            out = out + (q @ (k.transpose(-2, -1) @ v)) * d_inv
            # (B, num_heads, N, head_dim)

        # output projection
        out = out.transpose(1, 2).reshape(b, n, d_model) / self.num_inputs

        out = self.output_proj(out)

        return out

