import time
from collections import namedtuple

import numpy as np

from src.gradient_descent import SGD, Adam, Momentum
from src.graph import Graph
from src.operation import MatMul, Add, Sigmoid, Negative, ReduceSum, Multiply, Log, Tanh, Relu, Softmax, LeakyRelu
from src.placeholder import Placeholder
from src.session import Session
from src.variable import Variable

loss_registry = {
    'categorical_cross_entropy': lambda true, p: Negative(ReduceSum(ReduceSum(Multiply(true, Log(p)), axis=1))),
    'mean_squared_error': lambda true, p: ReduceSum(ReduceSum(Add(true, Negative(p)), axis=1))
}

activation_registry = {
    'sigmoid': lambda w, a_p, b: Sigmoid(Add(MatMul(a_p, w), b)),
    'tanh': lambda w, a_p, b: Tanh(Add(MatMul(a_p, w), b)),
    'relu': lambda w, a_p, b: Relu(Add(MatMul(a_p, w), b)),
    'leaky_relu': lambda w, a_p, b, alpha=0.01: LeakyRelu(Add(MatMul(a_p, w), b), alpha=alpha),
    'softmax': lambda w, a_p, b: Softmax(Add(MatMul(a_p, w), b))
}

optimizer_registry = {
    'Adam': lambda lr, loss, beta_one=0.9, beta_two=0.999: Adam(learning_rate=lr, beta_one=beta_one,
                                                                beta_two=beta_two).fit(loss),
    'SGD': lambda lr, loss: SGD(learning_rate=lr).fit(loss),
    'Momentum': lambda lr, loss, momentum=0.5: Momentum(learning_rate=lr, momentum=momentum).fit(loss),
}

Layer = namedtuple('Layer', ['input_nodes', 'activation'])


def glorot_weight(input_nodes, output_nodes):
    return np.random.randn(input_nodes, output_nodes) * np.sqrt(6.0 / (input_nodes + output_nodes))


def glorot_bias(input_nodes):
    return np.random.randn(input_nodes) * np.sqrt(6.0 / input_nodes)


def zip_xs_ys(xs, ys, size: int):
    indices = np.random.choice(xs.shape[0], size, replace=False)
    return xs[indices, :], ys[indices, :]


class Sequential:
    def __init__(self, layers: list[Layer] = None):
        if layers is None:
            layers = []

        self.layers = layers
        self.J = None
        self.min = None

        self.g: Graph = Graph()

    def compile(self, optimizer='SGD', loss='categorical_cross_entropy', learning_rate=0.001):

        self.g = Graph()
        self.g.as_default()

        X = Placeholder("X")
        c = Placeholder("c")

        self.X = X
        self.c = c

        size = len(self.layers)

        W_input = Variable(name=f"W_0_{self.layers[0].input_nodes, self.layers[0].input_nodes}",
                           initial_value=glorot_weight(
                               self.layers[0].input_nodes, self.layers[0].input_nodes), layer=0)
        b_input = Variable(name=f"b_0_{self.layers[0].input_nodes, 1}", initial_value=glorot_bias(
            self.layers[0].input_nodes), layer=0)
        input_sigmoid = Sigmoid(Add(MatMul(X, W_input), b_input))

        its = 1

        while its < size:
            glorot_w = glorot_weight(
                self.layers[its - 1].input_nodes,
                self.layers[its].input_nodes,
            )

            glorot_b = glorot_bias(self.layers[its].input_nodes)

            W_layer = Variable(name=f"W_{its}_{glorot_w.shape}",
                               initial_value=glorot_w, layer=its)
            b_layer = Variable(
                name=f"b_{its}_{glorot_b.shape}", initial_value=glorot_b, layer=its)

            layer_sigmoid = activation_registry[self.layers[its].activation](
                W_layer, input_sigmoid, b_layer)

            input_sigmoid = layer_sigmoid
            its += 1

        self.Z = input_sigmoid

        self.J = loss_registry[loss](c, input_sigmoid)

        self.min = optimizer_registry[optimizer](learning_rate, self.J)

    def fit(self, xs, ys, epochs=10, batch_size=64):
        info_split = epochs // 10

        s = Session()
        initial_loss = s.run(self.J, feed_dict={self.X: xs, self.c: ys})

        stats = {
            'initial_loss': initial_loss,
            'loss_epoch': [],
            'time_epoch': [],
            'correct_epoch': [],
            'epochs': epochs,
        }

        batches = xs.shape[0] // batch_size

        indices = np.arange(
            start=0, stop=xs.shape[0], step=batch_size, dtype=int)

        for i in range(epochs):
            t0 = time.time()
            s.run(self.min, feed_dict={self.X: xs, self.c: ys})
            t1 = time.time()

            stats['time_epoch'].append(t1 - t0)
            loss = s.run(self.J, feed_dict={
                         self.X: xs, self.c: ys}) / xs.shape[0]
            if i % info_split == 0 and i != 0:
                print(f"Epoch: {i}: Loss: {loss}")
            stats['loss_epoch'].append(loss)

        return stats

    def classify(self, xs):
        feed_dict = {
            self.X: xs
        }

        s = Session()
        out = s.run(operation=self.Z, feed_dict=feed_dict)
        return np.argmax(out, axis=1)

    def accuracy(self, Xs, labels):
        feed_dict = {
            self.X: Xs,
        }

        s = Session()
        out = s.run(self.Z, feed_dict=feed_dict)
        return np.array(np.argmax(out) == labels, dtype=int).sum()

    def predict(self, xs):
        feed_dict = {
            self.X: xs
        }

        s = Session()
        out = s.run(operation=self.Z, feed_dict=feed_dict)
        return out
