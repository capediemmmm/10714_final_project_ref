from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
            raise ValueError("attention batch/head dimensions and key/value lengths must agree")
        if queries_len == 0 or keys_values_len == 0 or q_dim == 0:
            raise ValueError("attention requires nonempty sequences and head dimensions")
        # The helper accepts its second argument already transposed: q @ k.T.
        scores = self.matmul(q, k) / (q_dim ** 0.5)
        if self.causal:
            mask = Tensor(self.create_causal_mask(queries_len, keys_values_len, q.device),
                          device=q.device, dtype=q.dtype, requires_grad=False)
            scores = scores + mask.broadcast_to(scores.shape)
        probs = self.softmax(scores)
        if self.causal and queries_len > keys_values_len:
            # Right alignment can leave early queries with no visible keys.
            # A finite masking sentinel would otherwise give a uniform softmax
            # on those fully masked rows; their output and gradients must be zero.
            visible_rows = (np.arange(queries_len) >= queries_len - keys_values_len)
            visible_rows = Tensor(visible_rows.astype(np.float32).reshape((1, 1, queries_len, 1)),
                                  device=q.device, dtype=q.dtype, requires_grad=False)
            probs = probs * visible_rows.broadcast_to(probs.shape)
        probs = self.dropout(probs)
        result = self.matmul(probs, v.transpose((2, 3)))
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, q_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        if (q_dim, k_dim, v_dim) != (self.q_features, self.k_features, self.v_features):
            raise ValueError("attention input feature dimensions do not match the layer")
        if k.shape[0] != batch_size or v.shape[:2] != (batch_size, keys_values_len):
            raise ValueError("attention batch dimensions and key/value lengths must agree")

        def project(x, norm, projection):
            x = projection(norm(x))
            return x.reshape((batch_size, x.shape[1], self.num_head, self.dim_head)).transpose((1, 2))

        queries = project(q, self.prenorm_q, self.q_projection)
        keys = project(k, self.prenorm_k, self.k_projection)
        values = project(v, self.prenorm_v, self.v_projection)
        attended, _ = self.attn(queries, keys, values)
        merged = attended.transpose((1, 2)).reshape(
            (batch_size, queries_len, self.num_head * self.dim_head))
        result = self.out_projection(merged)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attention = AttentionLayer(q_features, num_head, dim_head,
                                        dropout=dropout, causal=causal,
                                        device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.dropout2 = Dropout(dropout)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.dropout3 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        x = x + self.dropout1(self.attention(x))
        feedforward = self.linear1(self.norm(x))
        feedforward = self.dropout2(self.relu(feedforward))
        x = x + self.dropout3(self.linear2(feedforward))
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        if embedding_size < 1 or hidden_size < 1 or num_layers < 1 or sequence_len < 1:
            raise ValueError("transformer dimensions, layer count and sequence limit must be positive")
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len
        self.position_embedding = Embedding(sequence_len, embedding_size,
                                            device=device, dtype=dtype)
        self.layers = Sequential(*[
            TransformerLayer(embedding_size, num_head, dim_head, hidden_size,
                             dropout=dropout, causal=causal, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, features = x.shape
        if features != self.embedding_size:
            raise ValueError("input features must match embedding_size")
        if seq_len < 1 or seq_len > self.sequence_len:
            raise ValueError("input sequence length exceeds the learned position table")
        # Positions are constants; the table lookup itself remains differentiable.
        positions = Tensor(np.arange(seq_len, dtype=np.float32).reshape((1, seq_len)),
                           device=x.device, dtype=x.dtype, requires_grad=False)
        positional = self.position_embedding(positions).broadcast_to(x.shape)
        x = self.layers(x + positional)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
