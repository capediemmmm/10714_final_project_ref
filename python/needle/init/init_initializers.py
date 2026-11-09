import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out) if shape is None else tuple(shape)
    bound = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out) if shape is None else tuple(shape)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out) if shape is None else tuple(shape)
    bound = math.sqrt(6 / fan_in)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out) if shape is None else tuple(shape)
    std = math.sqrt(2 / fan_in)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION
