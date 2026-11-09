"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weight = Parameter(init.kaiming_uniform(
            fan_in, fan_out, shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device, dtype=dtype, requires_grad=True
        ))
        bound = 1 / np.sqrt(fan_in)
        self.bias = Parameter(init.rand(
            out_channels, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True
        )) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # NCHW -> NHWC for the primitive convolution, then convert back.
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size // 2)
        if self.bias is not None:
            out = out + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION
