"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


def _axes_tuple(axes, ndim):
    """Normalize reduction axes without relying on the backend's tuple support."""
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    result = tuple(axis + ndim if axis < 0 else axis for axis in axes)
    if len(set(result)) != len(result) or any(axis < 0 or axis >= ndim for axis in result):
        raise ValueError("invalid or repeated axis")
    return result


def _compact_reshape(a, shape):
    return a.reshape(shape) if BACKEND == "np" else a.compact().reshape(shape)


def _sum_to_shape(value, shape):
    """Undo broadcasting, including singleton axes at the same input rank."""
    if value.shape == shape:
        return value
    leading = len(value.shape) - len(shape)
    axes = tuple(i for i, size in enumerate(value.shape)
                 if i < leading or (shape[i - leading] == 1 and size != 1))
    if axes:
        value = summation(value, axes)
    return reshape(value, shape)


def _empty_like_shape(a, shape):
    if BACKEND == "np":
        return numpy.empty(shape, dtype=a.dtype)
    return array_api.empty(shape, device=a.device, dtype=a.dtype)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return numpy.power(a, b)
        return (b * a.log()).exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad * b * power(a, b - 1), out_grad * node * log(a)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.scalar == 0:
            return out_grad * 0
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = (-2, -1) if self.axes is None else self.axes
        if len(axes) != 2:
            raise ValueError("transpose expects two axes to swap")
        order = list(range(len(a.shape)))
        i, j = axes
        order[i], order[j] = order[j], order[i]
        return numpy.transpose(a, order) if BACKEND == "np" else a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return _compact_reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return _sum_to_shape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = _axes_tuple(self.axes, len(a.shape))
        # Descending order keeps the remaining axis numbers unchanged.
        for axis in sorted(axes, reverse=True):
            a = a.sum(axis=axis)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        axes = _axes_tuple(self.axes, len(shape))
        reduced_shape = tuple(1 if i in axes else size for i, size in enumerate(shape))
        return broadcast_to(reshape(out_grad, reduced_shape), shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np" or (len(a.shape) == 2 and len(b.shape) == 2):
            return a @ b
        if not a.shape or not b.shape:
            raise ValueError("matmul inputs must have at least one dimension")
        a_vector, b_vector = len(a.shape) == 1, len(b.shape) == 1
        if a_vector:
            a = _compact_reshape(a, (1, a.shape[0]))
        if b_vector:
            b = _compact_reshape(b, (b.shape[0], 1))
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("matmul inner dimensions must agree")
        batch_shape = numpy.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        a = a.broadcast_to(batch_shape + (m, k))
        b = b.broadcast_to(batch_shape + (k, n))
        out = _empty_like_shape(a, batch_shape + (m, n))
        for index in numpy.ndindex(batch_shape):
            selection = tuple(slice(i, i + 1) for i in index) + (slice(None), slice(None))
            left = a[selection].compact().reshape((m, k))
            right = b[selection].compact().reshape((k, n))
            out[selection] = left @ right
        shape = batch_shape + (() if a_vector else (m,)) + (() if b_vector else (n,))
        return out.reshape(shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_shape, b_shape = a.shape, b.shape
        a_vector, b_vector = len(a_shape) == 1, len(b_shape) == 1
        if a_vector:
            a = reshape(a, (1, a_shape[0]))
        if b_vector:
            b = reshape(b, (b_shape[0], 1))
        batch_shape = numpy.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        grad = reshape(out_grad, batch_shape + (a.shape[-2], b.shape[-1]))
        da = _sum_to_shape(matmul(grad, transpose(b)), a.shape)
        db = _sum_to_shape(matmul(transpose(a), grad), b.shape)
        return reshape(da, a_shape), reshape(db, b_shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor(node.inputs[0].realize_cached_data() > 0,
                      device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-(node * node) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not args:
            raise ValueError("cannot stack an empty sequence")
        shape = args[0].shape
        if any(a.shape != shape for a in args):
            raise ValueError("stack inputs must have identical shapes")
        axis = self.axis + len(shape) + 1 if self.axis < 0 else self.axis
        if axis < 0 or axis > len(shape):
            raise ValueError("stack axis out of bounds")
        if BACKEND == "np":
            return numpy.stack(args, axis=axis)
        out_shape = shape[:axis] + (len(args),) + shape[axis:]
        out = _empty_like_shape(args[0], out_shape)
        for i, a in enumerate(args):
            slices = [slice(None)] * len(out_shape)
            slices[axis] = slice(i, i + 1)
            out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = _axes_tuple(self.axis, len(A.shape))[0]
        shape = A.shape[:axis] + A.shape[axis + 1:]
        out = []
        for i in range(A.shape[axis]):
            slices = [slice(None)] * len(A.shape)
            slices[axis] = slice(i, i + 1)
            out.append(_compact_reshape(A[tuple(slices)], shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = _axes_tuple(self.axes, len(a.shape))
        return numpy.flip(a, axes) if BACKEND == "np" else a.flip(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation < 0:
            raise ValueError("dilation must be nonnegative")
        axes = {axis % len(a.shape) for axis in self.axes
                if -len(a.shape) <= axis < len(a.shape)}
        shape = tuple(size * (self.dilation + 1) if i in axes else size
                      for i, size in enumerate(a.shape))
        out = _empty_like_shape(a, shape)
        out.fill(0)
        slices = tuple(slice(0, size, self.dilation + 1 if i in axes else 1)
                       for i, size in enumerate(shape))
        out[slices] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation < 0:
            raise ValueError("dilation must be nonnegative")
        axes = {axis % len(a.shape) for axis in self.axes
                if -len(a.shape) <= axis < len(a.shape)}
        slices = tuple(slice(0, size, self.dilation + 1 if i in axes else 1)
                       for i, size in enumerate(a.shape))
        out = a[slices]
        return out.copy() if BACKEND == "np" else out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = dilate(out_grad, self.axes, self.dilation)
        # A direct undilate need not receive a multiple of the dilation factor.
        # Crop the trailing zero entries to recover exactly the input shape.
        return _CropToShape(node.inputs[0].shape)(grad)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class _CropToShape(TensorOp):
    """Crop trailing entries, with a differentiable zero-padding adjoint."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def compute(self, a):
        if a.shape == self.shape:
            return a
        value = a[tuple(slice(0, size) for size in self.shape)]
        return value.copy() if BACKEND == "np" else value.compact()

    def gradient(self, out_grad, node):
        return _PadToShape(node.inputs[0].shape)(out_grad)


class _PadToShape(TensorOp):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def compute(self, a):
        if a.shape == self.shape:
            return a
        out = _empty_like_shape(a, self.shape)
        out.fill(0)
        out[tuple(slice(0, size) for size in a.shape)] = a
        return out

    def gradient(self, out_grad, node):
        return _CropToShape(node.inputs[0].shape)(out_grad)


class _Im2Col(TensorOp):
    """Flatten NHWC convolution windows into rows; col2im is its adjoint.

    Keeping this rearrangement as an op makes convolution backward both
    differentiable and correct when a stride leaves unused border pixels.
    """

    def __init__(self, kernel_shape, stride, padding):
        self.kernel_shape = tuple(kernel_shape)
        self.stride, self.padding = stride, padding

    def compute(self, a):
        batch, height, width, channels = a.shape
        kh, kw = self.kernel_shape
        padding, stride = self.padding, self.stride
        oh = (height + 2 * padding - kh) // stride + 1
        ow = (width + 2 * padding - kw) // stride + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("kernel is larger than the padded input")
        if padding:
            pads = ((0, 0), (padding, padding), (padding, padding), (0, 0))
            a = numpy.pad(a, pads) if BACKEND == "np" else a.pad(pads)
        ns, hs, ws, cs = a.strides
        shape = (batch, oh, ow, kh, kw, channels)
        strides = (ns, hs * stride, ws * stride, hs, ws, cs)
        if BACKEND == "np":
            windows = numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        else:
            windows = a.as_strided(shape, strides)
        return _compact_reshape(windows, (batch * oh * ow, kh * kw * channels))

    def gradient(self, out_grad, node):
        return _Col2Im(node.inputs[0].shape, self.kernel_shape,
                       self.stride, self.padding)(out_grad)


class _Col2Im(TensorOp):
    """Sum overlapping window contributions back into an NHWC tensor."""

    def __init__(self, shape, kernel_shape, stride, padding):
        self.shape, self.kernel_shape = tuple(shape), tuple(kernel_shape)
        self.stride, self.padding = stride, padding

    def compute(self, columns):
        batch, height, width, channels = self.shape
        kh, kw = self.kernel_shape
        padding, stride = self.padding, self.stride
        ph, pw = height + 2 * padding, width + 2 * padding
        oh, ow = (ph - kh) // stride + 1, (pw - kw) // stride + 1
        windows = _compact_reshape(columns, (batch, oh, ow, kh, kw, channels))
        out = _empty_like_shape(columns, (batch, ph, pw, channels))
        out.fill(0)
        # Different output windows overlap, but each fixed kernel offset maps
        # to a distinct strided view. Add one offset at a time on the device.
        for i in range(kh):
            for j in range(kw):
                target = (slice(None), slice(i, i + (oh - 1) * stride + 1, stride),
                          slice(j, j + (ow - 1) * stride + 1, stride), slice(None))
                source = windows[:, :, :, i:i + 1, j:j + 1, :]
                source = _compact_reshape(source, (batch, oh, ow, channels))
                out[target] = out[target] + source
        if padding:
            out = out[:, padding:padding + height, padding:padding + width, :]
        return out.copy() if BACKEND == "np" else out.compact()

    def gradient(self, out_grad, node):
        return _Im2Col(self.kernel_shape, self.stride, self.padding)(out_grad)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if len(A.shape) != 4 or len(B.shape) != 4:
            raise ValueError("conv expects NHWC inputs and HWIO weights")
        if self.stride < 1 or self.padding < 0 or A.shape[3] != B.shape[2]:
            raise ValueError("invalid convolution stride, padding, or channel count")
        batch, height, width, _ = A.shape
        kh, kw, channels, out_channels = B.shape
        oh = (height + 2 * self.padding - kh) // self.stride + 1
        ow = (width + 2 * self.padding - kw) // self.stride + 1
        columns = _Im2Col((kh, kw), self.stride, self.padding).compute(A)
        weights = _compact_reshape(B, (kh * kw * channels, out_channels))
        return _compact_reshape(columns @ weights, (batch, oh, ow, out_channels))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        kh, kw, channels, out_channels = b.shape
        rows = out_grad.shape[0] * out_grad.shape[1] * out_grad.shape[2]
        grad = reshape(out_grad, (rows, out_channels))
        weights = reshape(b, (kh * kw * channels, out_channels))
        columns = _Im2Col((kh, kw), self.stride, self.padding)(a)
        a_grad = _Col2Im(a.shape, (kh, kw), self.stride, self.padding)(
            matmul(grad, transpose(weights)))
        b_grad = reshape(matmul(transpose(columns), grad), b.shape)
        return a_grad, b_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


