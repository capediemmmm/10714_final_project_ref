"""Check graph preprocessing, masked learning, and one complete optimizer step."""
from pathlib import Path
import pickle

import numpy as np
import pytest

import needle as ndl
from needle.sparse import CSRMatrix
from apps.gcn import (GCN, load_cora, masked_cross_entropy, normalize_adjacency,
                      row_normalize_features)


def tensor(value, requires_grad=False):
    return ndl.Tensor(np.asarray(value, dtype=np.float32), device=ndl.cpu(),
                      requires_grad=requires_grad)


def test_graph_normalization_merges_edges_and_adds_self_loops_once():
    # The 0-1 edge is listed in both directions and appears twice; node 3 is isolated.
    r, c, v = normalize_adjacency([0, 0, 1, 1, 1], [1, 1, 0, 2, 1], 4)
    a = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    degree = a.sum(axis=1)
    expected = a / np.sqrt(degree[:, None] * degree[None, :])
    assert len(v) == np.count_nonzero(expected)
    np.testing.assert_array_equal(np.stack([r, c]), np.stack(expected.nonzero()))
    np.testing.assert_allclose(v, expected[r, c], rtol=1e-6)


def test_feature_normalization_preserves_empty_rows():
    x = np.array([[0., 2., 2.], [0., 0., 0.], [0., 0., 4.]])
    np.testing.assert_allclose(row_normalize_features(x),
                               [[0, 0.5, 0.5], [0, 0, 0], [0, 0, 1]])


def test_masked_cross_entropy_and_gradient_ignore_held_out_labels():
    logits = np.array([[101, 103, 102], [-9, 1, 3], [0.2, 0.1, -0.4], [42, 44, 48]],
                      dtype=np.float32)
    labels = [0, 2, 1, 2]
    select = np.array([False, True, True, False])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    expected_loss = -np.log(probs[np.arange(4), labels][select]).mean()
    expected_grad = (probs - np.eye(3)[labels]) * select[:, None] / select.sum()
    for assignment in [labels, [1, 2, 1, 0]]:
        actual = tensor(logits, True)
        loss = masked_cross_entropy(actual, tensor(np.eye(3)[assignment]), tensor(select))
        np.testing.assert_allclose(loss.numpy().item(), expected_loss, atol=1e-6)
        loss.backward()
        np.testing.assert_allclose(actual.grad.numpy(), expected_grad, atol=1e-6)


def test_two_layer_gcn_one_training_step_sparse_vs_dense():
    # Compute the model and derivatives directly in NumPy to catch bugs common
    # to both implementations. Fixed weights keep ReLU away from its kink.
    r, c, v = normalize_adjacency([0, 1, 3], [1, 2, 1], 5)
    graph = np.zeros((5, 5), dtype=np.float32)
    graph[r, c] = v
    sparse = CSRMatrix.from_coo(r, c, v, graph.shape, device=ndl.cpu())
    x = np.array([[1, 0, 2], [0, 1, -1], [1, 2, 0], [-1, 1, 0], [2, 0, 1]],
                 dtype=np.float32)
    w1 = np.array([[0.5, -0.9, 0.3], [0.3, 0.7, -0.4], [-0.2, 0.4, 0.6]], np.float32)
    w2 = np.array([[0.7, -0.5], [-0.3, 0.1], [0.15, 0.5]], np.float32)
    labels = np.array([0, 1, 1, 0, 0])
    one_hot = np.eye(2, dtype=np.float32)[labels]
    mask = np.array([1, 1, 0, 0, 1], dtype=np.float32)
    z = graph @ (x @ w1)
    h = np.maximum(z, 0)
    logits = graph @ (h @ w2)
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    loss = -(np.log(prob[np.arange(len(labels)), labels]) * mask).sum() / mask.sum()
    delta = graph.T @ ((prob - one_hot) * mask[:, None] / mask.sum())
    grad2 = h.T @ delta
    grad1 = x.T @ (graph.T @ ((delta @ w2.T) * (z > 0)))

    for adjacency in [tensor(graph), sparse]:
        model = GCN(3, 3, 2, dropout=0.0, device=ndl.cpu())
        model.W1.data, model.W2.data = tensor(w1), tensor(w2)
        opt = ndl.optim.SGD(model.parameters(), lr=0.05)
        out = model(tensor(x), adjacency)
        actual_loss = masked_cross_entropy(out, tensor(one_hot), tensor(mask))
        np.testing.assert_allclose(out.numpy(), logits, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual_loss.numpy().item(), loss, rtol=1e-5)
        actual_loss.backward()
        np.testing.assert_allclose(model.W1.grad.numpy(), grad1, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model.W2.grad.numpy(), grad2, rtol=1e-4, atol=1e-5)
        opt.step()
        np.testing.assert_allclose(model.W1.numpy(), w1 - 0.05 * grad1, atol=1e-6)
        np.testing.assert_allclose(model.W2.numpy(), w2 - 0.05 * grad2, atol=1e-6)


def test_cora_public_split_and_test_node_order():
    directory = Path(__file__).resolve().parents[1] / "data" / "cora"
    if not (directory / "ind.cora.test.index").is_file():
        pytest.skip("Cora files are not installed")
    data = load_cora(directory)
    assert data.features.shape == (2708, 1433)
    assert data.labels.shape == (2708,)
    masks = np.stack((data.train_mask, data.val_mask, data.test_mask)).astype(int)
    np.testing.assert_array_equal(masks.sum(axis=1), [140, 500, 1000])
    assert np.all(masks.sum(axis=0) <= 1)
    test_ids = np.loadtxt(directory / "ind.cora.test.index", dtype=int)
    np.testing.assert_array_equal(np.flatnonzero(data.test_mask), np.sort(test_ids))
    # The pickle rows correspond to the original (unsorted) test IDs.
    # Compare labels and sampled feature rows directly to those shipped files.
    with (directory / "ind.cora.ty").open("rb") as f:
        target_rows = pickle.load(f, encoding="latin1")
    np.testing.assert_array_equal(data.labels[test_ids], target_rows.argmax(axis=1))
    with (directory / "ind.cora.tx").open("rb") as f:
        raw = pickle.load(f, encoding="latin1").tocsr()
    indices = [0, 19, 657, 999]
    ref = raw[indices].toarray()
    ref /= np.maximum(ref.sum(axis=1, keepdims=True), 1)
    np.testing.assert_allclose(data.features[test_ids[indices]], ref, atol=1e-6)
