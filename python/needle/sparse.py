"""Compressed sparse row (CSR) matrices for Needle's native float32 CPU backend.

Only stored values participate in differentiation. Row offsets and column indices
are fixed, read-only int32 metadata shared by each use of a matrix.
"""
import operator

import numpy as np

from .autograd import Tensor, TensorOp
from .backend_ndarray import cpu


def _shape_2d(shape):
    dims = tuple(operator.index(d) for d in shape)
    if len(dims) != 2 or any(d < 0 or d > np.iinfo(np.int32).max for d in dims):
        raise ValueError("a CSR shape must have two nonnegative int32 dimensions")
    return dims


def _integer_vector(data):
    a = np.asarray(data)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError("sparse matrix indices must be one-dimensional integer arrays")
    if a.size and (a.min() < 0 or a.max() > np.iinfo(np.int32).max):
        raise ValueError("indices must be nonnegative and fit in int32")
    return a.astype(np.int32, copy=False)


def _require_cpu_tensor(value):
    if not isinstance(value, Tensor):
        raise TypeError("expected a Needle Tensor")
    if value.device != cpu():
        raise ValueError("sparse operations require the native ndl.cpu() backend")


class CSRMatrix:
    """A fixed CSR pattern and a one-dimensional Needle tensor of values.

    Direct construction retains the supplied values tensor and its graph. Each
    CSR slot remains an independent value, including explicit zeros and repeated
    column indices. Use from_coo to construct a canonical pattern from host data.
    """

    def __init__(self, row_offsets, col_indices, values, shape):
        self._shape = _shape_2d(shape)
        offsets = _integer_vector(row_offsets)
        indices = _integer_vector(col_indices)
        _require_cpu_tensor(values)
        if values.shape != (len(indices),):
            raise ValueError("values must be a vector with one entry per column index")
        self._pattern = cpu().CSRPattern(*self.shape, offsets, indices)
        self.values = values

    @classmethod
    def from_coo(cls, rows, cols, values, shape, device=None, requires_grad=False):
        """Sort and sum duplicate input coordinates, then drop exact zeros.

        The host values initialize a new leaf tensor in CSR order. Gradients are
        available on matrix.values for the retained, coalesced entries.
        """
        m, n = _shape_2d(shape)
        rows, cols = _integer_vector(rows), _integer_vector(cols)
        if isinstance(values, Tensor):
            raise TypeError("from_coo expects host values; use the CSR constructor for a Tensor")
        data = np.asarray(values, dtype=np.float32)
        if data.ndim != 1 or not (rows.shape == cols.shape == data.shape):
            raise ValueError("COO rows, columns, and values must have matching vector shapes")
        if np.any(rows >= m) or np.any(cols >= n):
            raise ValueError("COO coordinate out of bounds")
        if device is None:
            device = cpu()
        if device != cpu():
            raise ValueError("CSR matrices require the native ndl.cpu() backend")
        if rows.size:
            order = np.lexsort((cols, rows))
            rows, cols, data = rows[order], cols[order], data[order]
            starts = np.flatnonzero(np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])])
            data = np.add.reduceat(data, starts)
            keep = data != 0
            rows, cols, data = rows[starts][keep], cols[starts][keep], data[keep]
        offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))]
        return cls(offsets, cols, Tensor(data, device=device, requires_grad=requires_grad), (m, n))

    @property
    def shape(self):
        return self._shape

    @property
    def row_offsets(self):
        return self._pattern.row_offsets

    @property
    def col_indices(self):
        return self._pattern.col_indices

    @property
    def nnz(self):
        return self.values.shape[0]

    @property
    def nbytes(self):
        """Persistent CSR payload, excluding Python objects and allocator padding."""
        return self.row_offsets.nbytes + self.col_indices.nbytes + self.nnz * 4

    def __matmul__(self, other):
        return spmm(self, other)


def _check_product(shape, x):
    _require_cpu_tensor(x)
    if len(x.shape) != 2 or shape[1] != x.shape[0]:
        raise ValueError(f"incompatible matrix shapes {shape} and {x.shape}")


def _multiply(matrix, values, x, transpose):
    rows = matrix.shape[1 if transpose else 0]
    width = x.shape[1]
    out = x.device.empty((rows, width))
    matrix._pattern.multiply(values.compact()._handle, x.compact()._handle,
                             out._handle, width, transpose)
    return out


class _CSRMatMul(TensorOp):
    """Multiply by A or A.T using the original CSR structure in either direction."""

    def __init__(self, matrix, transpose=False):
        self.matrix = matrix
        self.transpose = transpose

    def compute(self, values, x):
        return _multiply(self.matrix, values, x, self.transpose)

    def gradient(self, out_grad, node):
        values, x = node.inputs
        row_vectors, col_vectors = (x, out_grad) if self.transpose else (out_grad, x)
        dvalues = _SampledDot(self.matrix)(row_vectors, col_vectors)
        dx = _CSRMatMul(self.matrix, not self.transpose)(values, out_grad)
        return dvalues, dx


class _SampledDot(TensorOp):
    """Return <row_vectors[i], col_vectors[j]> at every stored (i, j)."""

    def __init__(self, matrix):
        self.matrix = matrix

    def compute(self, row_vectors, col_vectors):
        out = row_vectors.device.empty((self.matrix.nnz,))
        self.matrix._pattern.value_gradient(col_vectors.compact()._handle,
                                            row_vectors.compact()._handle,
                                            out._handle, row_vectors.shape[1])
        return out

    def gradient(self, out_grad, node):
        rows, cols = node.inputs
        return (_CSRMatMul(self.matrix)(out_grad, cols),
                _CSRMatMul(self.matrix, transpose=True)(out_grad, rows))


class _FixedCSRMatMul(TensorOp):
    def __init__(self, matrix, values, transpose=False):
        self.matrix = matrix
        self.values = values
        self.transpose = transpose

    def compute(self, x):
        return _multiply(self.matrix, self.values, x, self.transpose)

    def gradient(self, out_grad, node):
        return _FixedCSRMatMul(self.matrix, self.values, not self.transpose)(out_grad)


class _FixedDenseMatMul(TensorOp):
    def __init__(self, array):
        self.array = array

    def compute(self, x):
        return self.array @ x

    def gradient(self, out_grad, node):
        return _FixedDenseMatMul(self.array.permute((1, 0)))(out_grad)


def spmm(matrix, x):
    """Return A @ X and differentiate w.r.t. A.values and X."""
    _check_product(matrix.shape, x)
    return _CSRMatMul(matrix)(matrix.values, x)


def fixed_spmm(matrix, x):
    """Return A @ X with fixed CSR adjacency; only X enters the graph."""
    _check_product(matrix.shape, x)
    return _FixedCSRMatMul(matrix, matrix.values.realize_cached_data())(x)


def fixed_dense_matmul(matrix, x):
    """Dense native-CPU reference with the same fixed-left-operand semantics."""
    _require_cpu_tensor(matrix)
    if len(matrix.shape) != 2:
        raise ValueError("fixed dense matrix must have rank two")
    _check_product(matrix.shape, x)
    return _FixedDenseMatMul(matrix.realize_cached_data())(x)
