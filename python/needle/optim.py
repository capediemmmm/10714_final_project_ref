"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def clip_grad_norm(self, max_norm=0.25):
        """Clip the global L2 norm, without attaching optimizer state to a graph."""
        if max_norm < 0:
            raise ValueError("max_norm must be nonnegative")
        params = [param for param in self.params if getattr(param, "grad", None) is not None]
        squared_norm = sum(float(ndl.ops.summation(param.grad.data ** 2).numpy().item())
                           for param in params)
        total_norm = np.sqrt(squared_norm)
        scale = min(1.0, max_norm / (total_norm + 1e-6))
        for param in params:
            param.grad.data = param.grad.data * scale
        return total_norm


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if getattr(param, "grad", None) is None:
                continue
            grad = param.grad.data
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data
            previous = self.u.get(param)
            if previous is None:
                previous = ndl.init.zeros_like(param, requires_grad=False)
            update = self.momentum * previous + (1 - self.momentum) * grad
            self.u[param] = update.data
            param.data = param.data - self.lr * update.data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        return super().clip_grad_norm(max_norm)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if getattr(param, "grad", None) is None:
                continue
            grad = param.grad.data
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data
            old_m, old_v = self.m.get(param), self.v.get(param)
            if old_m is None:
                old_m = ndl.init.zeros_like(param, requires_grad=False)
                old_v = ndl.init.zeros_like(param, requires_grad=False)
            m = self.beta1 * old_m + (1 - self.beta1) * grad
            v = self.beta2 * old_v + (1 - self.beta2) * grad ** 2
            self.m[param], self.v[param] = m.data, v.data
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
