from queue import Queue

import numpy as np

from src.operation import Operation
from src.variable import Variable
from src.gradient_registry import _gradient_registry
from multiprocessing import Pool


def compute_gradients(loss):
    grad_table = {loss: 1}

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        if node != loss:
            grad_table[node] = 0

            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]

                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                lossgrads_wrt_consumer_inputs = bprop(
                    consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(
                        node)

                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    grad_table[node] += lossgrad_wrt_node

        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


class SGD:
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def fit(self, loss):
        lr = self.lr

        class MinimizationOperation(Operation):
            def __init__(self):
                super(MinimizationOperation, self).__init__(name="SGD")

            def _split_computation(self, gradient, node):
                node.value -= lr * gradient

            def _split_computation_array(self, gradients, nodes):
                for grad, node in zip(gradients, nodes):
                    self._split_computation(grad, node)

            def compute(self, **kwargs):
                gradient_table = compute_gradients(loss)
                applicable_nodes = [grad for grad in gradient_table if type(grad) == Variable]
                for node in applicable_nodes:
                    grad = gradient_table[node]
                    self._split_computation(gradient=grad, node=node)

        return MinimizationOperation()


class Adam:
    def __init__(self, learning_rate: float):
        self.lr = learning_rate
        self.w_m = {}
        self.w_n = {}

    def fit(self, loss):

        lr = self.lr
        w_m = self.w_m
        w_n = self.w_n

        class MinimizationOperation(Operation):
            def __init__(self, beta_one=0.9, beta_two=0.999):
                super(MinimizationOperation, self).__init__(name="Adam")
                self.beta_one = beta_one
                self.beta_two = beta_two

            def compute(self, **kwargs):
                gradient_table = compute_gradients(loss)
                grads = [grad for grad in gradient_table if type(grad) == Variable]
                for node in grads:
                    grad = gradient_table[node]

                    try:
                        w_m[node] = w_m[node] * self.beta_one + grad * (1 - self.beta_one)
                        w_n[node] = w_n[node] * self.beta_two + np.square(grad) * (1 - self.beta_two)
                    except KeyError:
                        w_m[node] = grad * (1 - self.beta_one)
                        w_n[node] = np.square(grad) * (1 - self.beta_two)

                    m_hat = w_m[node] / (1 - np.power(self.beta_one, node.layer))
                    v_hat = w_n[node] / (1 - np.power(self.beta_two, node.layer))
                    denom = np.sqrt(v_hat) + 1e-9
                    num = m_hat * lr

                    node.value -= denom / num

        return MinimizationOperation()
