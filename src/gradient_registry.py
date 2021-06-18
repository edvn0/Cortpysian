import numpy as np
from src.operation import *

_gradient_registry = {}


class RegisterGradient:

    def __init__(self, op_type):
        self._op_type = eval(op_type)

    def __call__(self, f, **kwargs):
        _gradient_registry[self._op_type] = f
        return f


@RegisterGradient("Negative")
def _negative_gradient(op, grad):
    return -grad


@RegisterGradient("Log")
def _log_gradient(op, grad):
    x = op.inputs[0]
    return grad / (x + 1e-9)


@RegisterGradient("Sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)


@RegisterGradient("Multiply")
def _multiply_gradient(op, grad):
    a = op.inputs[0]
    b = op.inputs[0]

    return [grad * b, grad * a]


@RegisterGradient("MatMul")
def _matmul_gradient(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]

    return [grad.dot(b.T), a.T.dot(grad)]


@RegisterGradient("Add")
def _add_gradient(op, grad):
    """Computes the gradients for `add`.

    Args:
      op: The `add` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `add` op.

    Returns:
      Gradients with respect to the input of `add`.
    """
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    grad_wrt_b = grad

    #
    # The following becomes relevant if a and b are of different shapes.
    #
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]


@RegisterGradient("ReduceSum")
def _reduce_sum_gradient(op, grad):
    a = op.inputs[0]

    output_shape = np.array(a.shape)
    output_shape[op.axis] = 1
    tile_scaling = a.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)


@RegisterGradient("Softmax")
def _softmax_gradient(op, grad):
    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
    )) * softmax


@RegisterGradient("Relu")
def _relu_gradient(op, grad):
    relu = np.copy(op.output)
    relu[relu > 0] = 1
    relu[relu <= 0] = 0
    return grad * relu


@RegisterGradient("LeakyRelu")
def _relu_gradient(op, grad):
    relu = np.copy(op.output)
    relu[relu > 0] = op.alpha
    relu[relu <= 0] = 0
    return grad * relu


@RegisterGradient("Tanh")
def _tanh_gradient(op, grad):
    tanh = op.output
    return grad * tanh * tanh
