"""Full-batch Cora node classification on the native Needle CPU backend."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import needle as ndl
from needle import nn, ops


@dataclass
class GraphData:
    """Normalized features and adjacency entries, integer labels, and boolean masks."""

    features: np.ndarray
    labels: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def num_nodes(self):
        return self.features.shape[0]

    @property
    def num_classes(self):
        return int(self.labels.max()) + 1


def normalize_adjacency(rows, cols, num_nodes):
    """Build D^(-1/2) (A + I) D^(-1/2) as row-sorted COO for a binary graph.

    Reciprocal edges, duplicate citations, and existing self-loops count once.
    Sorting integer coordinate keys uses space proportional to nodes plus edges.
    """
    if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    if rows.ndim != 1 or rows.shape != cols.shape:
        raise ValueError("edge endpoints must be equally sized one-dimensional arrays")
    for endpoints in (rows, cols):
        if endpoints.size and (not np.issubdtype(endpoints.dtype, np.integer)
                               or endpoints.min() < 0 or endpoints.max() >= num_nodes):
            raise ValueError("edge endpoints must be integer node indices in range")
    if num_nodes == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    rows = rows.astype(np.int64, copy=False)
    cols = cols.astype(np.int64, copy=False)
    loops = np.arange(num_nodes, dtype=np.int64) * (num_nodes + 1)
    keys = np.unique(np.concatenate((rows * num_nodes + cols, cols * num_nodes + rows, loops)))
    rows, cols = np.divmod(keys, num_nodes)
    inv_sqrt_degree = 1.0 / np.sqrt(np.bincount(rows, minlength=num_nodes))
    values = (inv_sqrt_degree[rows] * inv_sqrt_degree[cols]).astype(np.float32)
    return rows.astype(np.int32), cols.astype(np.int32), values


def row_normalize_features(features):
    """Divide each nonempty bag-of-words row by its sum, preserving empty rows."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape (nodes, features)")
    totals = features.sum(axis=1, keepdims=True)
    return np.divide(features, totals, out=np.zeros_like(features), where=totals != 0)


def load_cora(data_dir):
    """Load the original, trusted Planetoid Cora pickles and public node split.

    The test block is stored in the order given by test.index, which differs
    from graph node order. SciPy is used only to read its serialized features.
    """
    directory = Path(data_dir)

    def read(name):
        with (directory / ("ind.cora." + name)).open("rb") as f:
            return pickle.load(f, encoding="latin1")

    x, y = read("x"), read("y")
    allx, ally = read("allx"), read("ally")
    tx, ty = read("tx"), read("ty")
    graph = read("graph")
    test_ids = np.loadtxt(directory / "ind.cora.test.index", dtype=np.int64, ndmin=1)
    n_train = x.shape[0]
    n_prefix, n_features = allx.shape
    n_nodes = n_prefix + tx.shape[0]
    if (x.shape[1] != n_features or tx.shape[1] != n_features
            or y.shape[0] != n_train or ally.shape[0] != n_prefix or ty.shape[0] != tx.shape[0]
            or not (y.shape[1] == ally.shape[1] == ty.shape[1])
            or n_train + 500 > n_prefix
            or not np.array_equal(np.sort(test_ids), np.arange(n_prefix, n_nodes))):
        raise ValueError("inconsistent dimensions or test indices in Planetoid Cora data")
    if (x != allx[:n_train]).nnz or not np.array_equal(y, ally[:n_train]):
        raise ValueError("training rows do not align with the graph node ordering")

    features = np.empty((n_nodes, n_features), dtype=np.float32)
    features[:n_prefix] = allx.toarray()
    features[test_ids] = tx.toarray()
    labels = np.empty(n_nodes, dtype=np.int64)
    labels[:n_prefix] = np.argmax(ally, axis=1)
    labels[test_ids] = np.argmax(ty, axis=1)

    rows, cols = [], []
    for node, neighbors in graph.items():
        rows.extend([node] * len(neighbors))
        cols.extend(neighbors)
    rows, cols, values = normalize_adjacency(rows, cols, n_nodes)
    train = np.arange(n_nodes) < n_train
    val = np.zeros(n_nodes, dtype=bool)
    val[n_train:n_train + 500] = True
    test = np.zeros(n_nodes, dtype=bool)
    test[test_ids] = True
    return GraphData(row_normalize_features(features), labels, rows, cols, values, train, val, test)


def make_adjacency(data, layout, device=None):
    """Create only the requested representation from the same normalized COO."""
    from needle.sparse import CSRMatrix
    device = ndl.cpu() if device is None else device
    shape = (data.num_nodes, data.num_nodes)
    if layout == "sparse":
        return CSRMatrix.from_coo(data.rows, data.cols, data.values, shape,
                                  device=device, requires_grad=False)
    if layout == "dense":
        dense = np.zeros(shape, dtype=np.float32)
        dense[data.rows, data.cols] = data.values
        return ndl.Tensor(dense, device=device, requires_grad=False)
    raise ValueError("layout must be 'sparse' or 'dense'")


def prepare_inputs(data, device=None):
    """Return constant feature, one-hot label, and training-mask tensors."""
    device = ndl.cpu() if device is None else device
    targets = np.eye(data.num_classes, dtype=np.float32)[data.labels]
    return tuple(ndl.Tensor(x, device=device, requires_grad=False)
                 for x in (data.features, targets, data.train_mask.astype(np.float32)))


def masked_cross_entropy(logits, targets, mask):
    """Mean cross entropy over the nodes selected by a binary train mask."""
    if len(logits.shape) != 2 or targets.shape != logits.shape or mask.shape != (logits.shape[0],):
        raise ValueError("expected logits and targets of shape (N, C) and a mask of shape (N,)")
    weights = mask.numpy()
    if not np.all((weights == 0) | (weights == 1)):
        raise ValueError("the training mask must contain only zeros and ones")
    count = int(np.count_nonzero(weights))
    if count == 0:
        raise ValueError("the training mask must include at least one node")
    losses = ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * targets, axes=(1,))
    return ops.summation(losses * mask) / count


class GCN(nn.Module):
    """Two bias-free graph convolution layers with ReLU and hidden dropout."""

    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5, device=None):
        super().__init__()
        device = ndl.cpu() if device is None else device
        self.W1 = nn.Parameter(ndl.init.xavier_uniform(in_features, hidden_features, device=device))
        self.W2 = nn.Parameter(ndl.init.xavier_uniform(hidden_features, num_classes, device=device))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        from needle.sparse import CSRMatrix, fixed_dense_matmul, fixed_spmm
        aggregate = fixed_spmm if isinstance(adj, CSRMatrix) else fixed_dense_matmul
        # Cora features are fixed. Use a one-input operation so the CPU engine
        # differentiates W1 without allocating an unused gradient for every word.
        projected = x @ self.W1 if x.requires_grad else fixed_dense_matmul(x, self.W1)
        hidden = self.dropout(ops.relu(aggregate(adj, projected)))
        return aggregate(adj, hidden @ self.W2)


def train_epoch(model, adj, inputs, optimizer):
    """Take one full-graph step; the supervised loss sees only training nodes."""
    x, targets, mask = inputs
    model.train()
    optimizer.reset_grad()
    loss = masked_cross_entropy(model(x, adj), targets, mask)
    loss.backward()
    optimizer.step()
    return float(loss.numpy().item())


def evaluate(model, adj, inputs, labels, mask):
    """Accuracy for the requested node split, with dropout disabled."""
    model.eval()
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels)
    if labels.ndim != 1 or mask.shape != labels.shape or not mask.any():
        raise ValueError("expected a nonempty node split aligned with the labels")
    scores = model(inputs[0], adj).numpy()
    return float(np.mean(np.argmax(scores[mask], axis=1) == labels[mask]))
