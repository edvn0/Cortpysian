from typing import List
import numpy as np

import src.graph


class Operation:
    def __init__(self, name: str = None, input_nodes=None):
        if input_nodes is None:
            input_nodes = []
        self.input_nodes = input_nodes
        self.consumers: List[Operation] = []

        self.name = name

        for input_node in self.input_nodes:
            input_node.consumers.append(self)

        src.graph._default_graph.operations.append(self)

    def compute(self, **kwargs):
        raise NotImplementedError("Not implemented.")


class Add(Operation):

    def __init__(self, x, y):
        super().__init__(name="Add", input_nodes=[x, y])

    def compute(self, x_value, y_value):
        return x_value + y_value


class MatMul(Operation):

    def __init__(self, x, y):
        super().__init__(name="MatMul", input_nodes=[x, y])

    def compute(self, x_value, y_value):
        return np.dot(x_value, y_value)


class Sigmoid(Operation):

    def __init__(self, value):
        super(Sigmoid, self).__init__(name="Sigmoid", input_nodes=[value])

    def compute(self, a_value):
        a = np.exp(-a_value)
        return 1 / (1 + a)


class Softmax(Operation):
    def __init__(self, value):
        super(Softmax, self).__init__(name="Softmax", input_nodes=[value])

    def compute(self, a_value):
        def stable_softmax(x):
            z = x - np.max(x, axis=-1, keepdims=True)
            numerator = np.exp(z)
            denominator = np.sum(numerator, axis=-1, keepdims=True)
            softmax = numerator / denominator
            return softmax
        return stable_softmax(a_value)


class Log(Operation):
    def __init__(self, x):
        super(Log, self).__init__(name="Log", input_nodes=[x])

    def compute(self, x_val):
        return np.log(x_val + 1e-9)


class Multiply(Operation):

    def __init__(self, x, y):
        super().__init__(name="Multiply", input_nodes=[x, y])

    def compute(self, x_value, y_value):
        return x_value * y_value


class ReduceSum(Operation):
    def __init__(self, a, axis=None):
        super(ReduceSum, self).__init__(name="ReduceSum", input_nodes=[a])
        self.axis = axis

    def compute(self, a_value):
        return np.sum(a_value, self.axis)


class Negative(Operation):
    def __init__(self, x):
        super(Negative, self).__init__(name="Negative", input_nodes=[x])

    def compute(self, x_value):
        return -x_value


class Relu(Operation):
    def __init__(self, x):
        super(Relu, self).__init__(name="Relu", input_nodes=[x])

    def compute(self, x):
        out = np.array(x)
        return out * (out > 0)


class Tanh(Operation):
    def __init__(self, x):
        super(Tanh, self).__init__(name="Tanh", input_nodes=[x])

    def compute(self, x):
        return np.tanh(x)
