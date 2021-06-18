from queue import Queue

import numpy as np

from src.operation import Operation
from src.variable import Variable
from src.gradient_registry import _gradient_registry


def compute_gradients(loss):
    grad_table = {loss: 1.0}

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    # Perform backwards BFS, starting at loss going to the placeholder node.
    while not queue.empty():
        node = queue.get()

        if node != loss:
            grad_table[node] = 0

            for consumer in node.consumers:
                dh_do = grad_table[consumer]

                # The registry has class strings stored, we consume by querying the object for its __class__
                consumer_operation_type = consumer.__class__
                calculate_gradient = _gradient_registry[consumer_operation_type]

                # Backpropagate the gradient of this consumer and the gradient
                # dH/dConsumerInput
                dh_d_ci = calculate_gradient(
                    consumer, dh_do)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += dh_d_ci
                else:
                    index_current_node = consumer.input_nodes.index(node)
                    # dH/dNode
                    dh_dn = dh_d_ci[index_current_node]
                    grad_table[node] += dh_dn

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
                grads = [gradient_table[grad] for grad in applicable_nodes]
                self._split_computation_array(gradients=grads, nodes=applicable_nodes)

        return MinimizationOperation()


class Momentum:
    def __init__(self, learning_rate: float, momentum: float = .99):
        self.lr = learning_rate
        self.momentum = momentum

    def fit(self, loss: Operation):
        class MinimizationOperation(Operation):
            def __init__(self, velocity=None, lr=0.01, momentum=.9):
                super(MinimizationOperation, self).__init__(name="Momentum")
                if velocity is None:
                    velocity = {}
                self.lr = lr
                self.velocity = velocity
                self.momentum = momentum

            def compute(self, **kwargs):
                gradient_table = compute_gradients(loss)
                grads = [grad for grad in gradient_table if type(grad) == Variable]
                for node in grads:
                    grad = gradient_table[node]

                    try:
                        self.velocity[node] = self.momentum * self.velocity[node] - self.lr * grad
                    except KeyError:
                        self.velocity[node] = -self.lr * grad

                    grad = grad + self.velocity[node]

                    node.value = grad

        return MinimizationOperation(None, self.lr, self.momentum)


class Adam:
    def __init__(self, learning_rate: float, beta_one, beta_two):
        self.lr = learning_rate
        self.w_m = {}
        self.w_n = {}
        self.beta_one = beta_one
        self.beta_two = beta_two

    def fit(self, loss):

        w_m = self.w_m
        w_n = self.w_n

        class MinimizationOperation(Operation):
            def __init__(self, alpha, beta_one, beta_two):
                super(MinimizationOperation, self).__init__(name="Adam")
                self.alpha = alpha
                self.beta_one = beta_one
                self.beta_two = beta_two
                self.t = 1

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

                    layer = node.layer + 1
                    b1_of_t = float(1 - self.beta_one ** layer)
                    b2_of_t = float(1 - self.beta_two ** layer)
                    m_hat = w_m[node] / b1_of_t
                    v_hat = w_n[node] / b2_of_t

                    self.t += 1

                    denom = np.sqrt(v_hat) + 1e-9
                    num = m_hat * self.alpha

                    adam = num / denom

                    node.value -= adam

        return MinimizationOperation(self.lr, self.beta_one, self.beta_two)
