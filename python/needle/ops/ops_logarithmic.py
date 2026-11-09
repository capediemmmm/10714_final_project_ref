from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from .ops_mathematic import _axes_tuple, _compact_reshape

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axis = len(Z.shape) - 1
        shape = Z.shape[:-1] + (1,)
        normalizer = LogSumExp((axis,)).compute(Z)
        normalizer = array_api.broadcast_to(_compact_reshape(normalizer, shape), Z.shape)
        return Z - normalizer
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        axis = len(node.shape) - 1
        shape = node.shape[:-1] + (1,)
        total = summation(out_grad, axes=(axis,))
        return out_grad - exp(node) * broadcast_to(reshape(total, shape), node.shape)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = _axes_tuple(self.axes, len(Z.shape))
        if not axes:
            return Z
        maximum = Z
        for axis in sorted(axes, reverse=True):
            maximum = maximum.max(axis=axis)
        shape = tuple(1 if i in axes else size for i, size in enumerate(Z.shape))
        shifted = Z - array_api.broadcast_to(_compact_reshape(maximum, shape), Z.shape)
        exp_sum = array_api.exp(shifted)
        for axis in sorted(axes, reverse=True):
            exp_sum = exp_sum.sum(axis=axis)
        return array_api.log(exp_sum) + maximum
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        axes = _axes_tuple(self.axes, len(z.shape))
        if not axes:
            return out_grad
        shape = tuple(1 if i in axes else size for i, size in enumerate(z.shape))
        total = broadcast_to(reshape(node, shape), z.shape)
        grad = broadcast_to(reshape(out_grad, shape), z.shape)
        return grad * exp(z - total)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
