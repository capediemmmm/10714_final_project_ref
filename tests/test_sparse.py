"""Independent forward and first-order gradient checks on Needle's native CPU."""
import numpy as np
import pytest

import needle as ndl
from needle.sparse import CSRMatrix, fixed_dense_matmul, fixed_spmm, spmm


def tensor(data, requires_grad=True):
    return ndl.Tensor(np.asarray(data, dtype=np.float32), device=ndl.cpu(),
                      requires_grad=requires_grad)


def test_project_plan_example():
    values = tensor([2, 3, 4])
    matrix = CSRMatrix([0, 1, 3], [1, 0, 2], values, (2, 3))
    x = tensor([[1], [2], [-1]])
    y = spmm(matrix, x)
    np.testing.assert_array_equal(y.numpy(), [[4], [-1]])
    y.backward(tensor([[1], [2]], requires_grad=False))
    np.testing.assert_array_equal(values.grad.numpy(), [2, 2, -2])
    np.testing.assert_array_equal(x.grad.numpy(), [[6], [2], [8]])


def test_coo_duplicate_coordinates_and_empty_rows_match_dense_autograd():
    # Unsorted input; (1, 3) occurs twice, and rows zero and two are empty.
    rows = [3, 1, 1, 3, 1]
    cols = [0, 3, 1, 4, 3]
    vals = [0.5, 2, -3, 1.5, -0.25]
    shape = (4, 5)
    adj = CSRMatrix.from_coo(rows, cols, vals, shape, device=ndl.cpu(), requires_grad=True)
    np.testing.assert_array_equal(adj.row_offsets, [0, 0, 2, 2, 4])
    np.testing.assert_array_equal(adj.col_indices, [1, 3, 0, 4])
    np.testing.assert_allclose(adj.values.numpy(), [-3, 1.75, 0.5, 1.5])

    full = np.zeros(shape, dtype=np.float32)
    np.add.at(full, (rows, cols), vals)
    dense = tensor(full)
    rng = np.random.default_rng(31)
    features = rng.normal(size=(5, 3))
    upstream = tensor(rng.normal(size=(4, 3)), requires_grad=False)
    xs, xd = tensor(features), tensor(features)
    ys, yd = spmm(adj, xs), dense @ xd
    np.testing.assert_allclose(ys.numpy(), yd.numpy(), atol=1e-5, rtol=1e-4)
    ys.backward(upstream)
    yd.backward(upstream)
    np.testing.assert_allclose(xs.grad.numpy(), xd.grad.numpy(), atol=1e-5, rtol=1e-4)
    # Compare only the independently known stored positions, in CSR order.
    dense_value_grads = dense.grad.numpy()[[1, 1, 3, 3], [1, 3, 0, 4]]
    np.testing.assert_allclose(adj.values.grad.numpy(), dense_value_grads,
                               atol=1e-5, rtol=1e-4)


def test_central_differences_of_nonlinear_loss_for_both_inputs():
    rows, cols = [0, 2, 3], [0, 2, 1]
    values = np.array([0.5, -1.0, 0.75], dtype=np.float32)
    features = np.array([[0.2, 0.8], [-0.5, 0.25], [0.7, -0.1]], dtype=np.float32)
    scale = tensor([[1.5, -0.25], [0.75, 2.5]], requires_grad=False)

    def objective(v, x):
        result = spmm(CSRMatrix(rows, cols, v, (2, 3)), x)
        return (result * result * scale).sum()

    v, x = tensor(values), tensor(features)
    objective(v, x).backward()
    for inputs, analytical in [(values, v.grad.numpy()), (features, x.grad.numpy())]:
        finite = np.empty_like(inputs)
        eps = 1e-3
        for idx in np.ndindex(inputs.shape):
            orig = inputs[idx].item()
            inputs[idx] = orig + eps
            plus = objective(tensor(values, False), tensor(features, False)).numpy().item()
            inputs[idx] = orig - eps
            minus = objective(tensor(values, False), tensor(features, False)).numpy().item()
            inputs[idx] = orig
            finite[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(analytical, finite, atol=2e-4, rtol=2e-3)


def test_strided_features_and_upstream_with_fixed_adjacency():
    # Transposed inputs catch kernels interpreting a view as a packed buffer.
    a = np.array([[0, 1, 2], [-3, 0, 0]], dtype=np.float32)
    values = tensor([1, 2, -3])
    adj = CSRMatrix([0, 2, 3], [1, 2, 0], values, (2, 3))
    dense = tensor(a)
    xt = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
    gt = (np.arange(8, dtype=np.float32).reshape(4, 2) - 2) / 3
    u = tensor(gt, requires_grad=False).transpose()
    for graph, product, weight in [(adj, fixed_spmm, values),
                                   (dense, fixed_dense_matmul, dense)]:
        leaf = tensor(xt)
        y = product(graph, leaf.transpose())
        np.testing.assert_allclose(y.numpy(), a @ xt.T, atol=1e-6)
        y.backward(u)
        np.testing.assert_allclose(leaf.grad.numpy(), (a.T @ gt.T).T, atol=1e-6)
        assert getattr(weight, "grad", None) is None


def test_empty_sparsity_pattern_has_zero_output_and_feature_gradients():
    empty = tensor(np.empty(0, dtype=np.float32))
    matrix = CSRMatrix([0, 0, 0, 0], np.array([], dtype=np.int32), empty, (3, 4))
    x = tensor(np.ones((4, 2)))
    out = spmm(matrix, x)
    np.testing.assert_array_equal(out.numpy(), np.zeros((3, 2)))
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros((4, 2)))
    assert empty.grad.shape == (0,)


@pytest.mark.parametrize("offsets,columns,values", [
    ([0, 2], [0, 1], [1, 2]),      # wrong pointer count for two rows
    ([1, 1, 2], [0, 1], [1, 2]),   # pointers do not start at zero
    ([0, 2, 1], [0], [1]),          # nonmonotone pointers
    ([0, 1, 2], [0], [1]),          # last pointer disagrees with nnz
    ([0, 1, 2], [0, 2], [1, 2]),   # column outside shape
    ([0, 1, 2], [-1, 1], [1, 2]),  # negative columns are invalid
    ([0, 1, 2], [0, 1], [1]),       # values/indices length mismatch
])
def test_malformed_csr_is_rejected(offsets, columns, values):
    with pytest.raises((TypeError, ValueError, IndexError)):
        CSRMatrix(offsets, columns, tensor(values), (2, 2))


def test_invalid_product_dimensions_are_rejected():
    adj = CSRMatrix([0, 1], [1], tensor([2]), (1, 3))
    for shape in [(2, 2), (3,), (1, 3, 2)]:
        with pytest.raises((ValueError, TypeError)):
            spmm(adj, tensor(np.ones(shape)))
