from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from src.gradient_descent import SGD, Adam
from src.graph import Graph
from src.operation import MatMul, Add, Sigmoid, Negative, ReduceSum, Multiply, Log, Tanh, Relu, Softmax
from src.placeholder import Placeholder
from src.session import Session
from src.variable import Variable

loss_registry = {
    'categorical_cross_entropy': lambda x, p: Negative(ReduceSum(ReduceSum(Multiply(x, Log(p)), axis=1))),
    'mean_squared_error': lambda x, p: ReduceSum(ReduceSum(Add(x, Negative(p)), axis=1))
}

activation_registry = {
    'sigmoid': lambda weight, activation_or_placeholder, bias: Sigmoid(
        Add(MatMul(activation_or_placeholder, weight), bias)),
    'tanh': lambda weight, activation_or_placeholder, bias: Tanh(Add(MatMul(activation_or_placeholder, weight), bias)),
    'relu': lambda weight, activation_or_placeholder, bias: Relu(Add(MatMul(activation_or_placeholder, weight), bias)),
    'softmax': lambda weight, activation_or_placeholder, bias: Softmax(
        Add(MatMul(activation_or_placeholder, weight), bias))
}

Layer = namedtuple('Layer', ['input_nodes', 'activation'])


def glorot_weight(input_nodes, output_nodes):
    return np.random.randn(input_nodes, output_nodes) * np.sqrt(6.0 / (input_nodes + output_nodes))


def glorot_bias(input_nodes):
    return np.random.randn(input_nodes) * np.sqrt(6.0 / input_nodes)


class Sequential:
    def __init__(self, layers: list[Layer] = None):
        if layers is None:
            layers = []

        self.layers = layers
        self.J = None
        self.min = None

        self.g: Graph = Graph()

    def compile(self, optim='SGD', loss='categorical_cross_entropy', learning_rate=0.001):

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

            if its != size - 1:
                layer_sigmoid = Sigmoid(Add(MatMul(input_sigmoid, W_layer), b_layer))
            else:
                layer_sigmoid = Softmax(Add(MatMul(input_sigmoid, W_layer), b_layer))

            input_sigmoid = layer_sigmoid
            its += 1

        self.J = Negative(ReduceSum(ReduceSum(Multiply(c, Log(input_sigmoid)), axis=1))) if \
            loss == 'categorical_cross_entropy' else \
            ReduceSum(ReduceSum(Multiply(Add(c, Negative(input_sigmoid)), Add(c, Negative(input_sigmoid))), axis=1))

        self.min = Adam(learning_rate=learning_rate).fit(
            self.J) if optim == 'Adam' else SGD(learning_rate=learning_rate).fit(self.J)

    def fit(self, xs, ys, epochs=100):
        feed_dict = {
            self.X: xs,
            self.c: ys
        }

        info_split = epochs // 10

        s = Session(self.J, feed_dict=feed_dict)
        s.run()

        s = Session(self.min, feed_dict=feed_dict)
        losses = []
        for i in range(epochs):
            loss_session = Session(self.J, feed_dict=feed_dict)
            l = loss_session.run()
            if i % info_split == 0:
                print(f"Epoch: {i}: Loss: {l}")
            losses.append(l)
            s.run()

        epochs = [i + 1 for i in range(len(losses))]

        fig, ax = plt.subplots()
        ax.plot(epochs, losses)

        ax.set(xlabel='epochs', ylabel='loss',
               title='Loss over time')
        ax.grid()

        fig.savefig("test.png")
        plt.show()
