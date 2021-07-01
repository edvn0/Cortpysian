import time
from collections import namedtuple
from typing import Iterable, List

from sklearn.model_selection import train_test_split

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

metrics_registry = {
    'mse': lambda x, y: np.mean(np.square(np.array(x-y))),
    'mae': lambda x, y: np.mean(np.abs(np.array(x-y))),
    'accuracy': lambda x, y: np.mean(np.argmax(x, axis=1) == np.argmax(y, axis=1))
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
    def __init__(self, layers: Iterable[Layer] = None):
        if layers is None:
            layers = []

        self.layers = list(layers)
        self.loss_function = None
        self.min = None
        self.metrics = []
        self.g: Graph = Graph()

    def compile(self, optimizer='SGD', loss='categorical_cross_entropy', learning_rate=0.001, metrics: List[str] = []):
        for metric in metrics:
            self.metrics.append(metrics_registry[metric])

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
        input_sigmoid = activation_registry[self.layers[0].activation](
            W_input, self.X, b_input)

        its = 1

        while its < size:
            glorot_w = glorot_weight(
                self.layers[its - 1].input_nodes,
                self.layers[its].input_nodes,
            )

            glorot_b = glorot_bias(self.layers[its].input_nodes)

            W_layer = Variable(name=f"W_{its}_{glorot_w.shape}",
                               initial_value=glorot_w,
                               layer=its)

            b_layer = Variable(name=f"b_{its}_{glorot_b.shape}",
                               initial_value=glorot_b,
                               layer=its)

            layer_sigmoid = activation_registry[self.layers[its].activation](
                W_layer, input_sigmoid, b_layer)

            input_sigmoid = layer_sigmoid
            its += 1

        self.last_activation = input_sigmoid

        self.loss_function = loss_registry[loss](c, input_sigmoid)

        self.min = optimizer_registry[optimizer](
            learning_rate, self.loss_function)

    def fit(self, xs, ys, epochs=10, batch_size=64):
        info_split = epochs // 10

        s = Session()
        initial_loss = s.run(self.loss_function, feed_dict={
                             self.X: xs, self.c: ys})

        stats = {
            'initial_loss': initial_loss,
            'loss_epoch': [],
            'time_epoch': [],
            'correct_epoch': [],
            'epochs': epochs,
        }

        trainX = xs[0:int(0.8*xs.shape[0])]
        trainY = ys[0:int(0.8*xs.shape[0])]
        testX = xs[int(0.8*xs.shape[0]):]
        testY = ys[int(0.8*xs.shape[0]):]
        batches = trainX.shape[0] // batch_size

        x_t = np.array_split(trainX, batches)
        y_t = np.array_split(trainY, batches)

        for i in range(epochs):
            t0 = time.time()
            self.fit_on_batch(s, x_t, y_t)
            t1 = time.time()

            stats['time_epoch'].append(t1 - t0)
            loss = s.run(self.loss_function, feed_dict={
                self.X: testX, self.c: testY}) / testX.shape[0]

            preds = s.run(self.last_activation, feed_dict={
                self.X: testX})
            if i % info_split == 0 and i != 0:
                metrics = [metric(preds, testY) for metric in self.metrics]
                print(f"Epoch: {i}: Loss: {loss}, metrics: {metrics}")

            stats['loss_epoch'].append(loss)

        return stats

    def fit_on_batch(self, s: Session, x_t: List[np.ndarray], y_t: List[np.ndarray]):
        for x, y in zip(x_t, y_t):
            s.run(self.min, feed_dict={
                self.X: x,
                self.c: y
            })

    def classify(self, xs):
        feed_dict = {
            self.X: xs
        }

        s = Session()
        out = s.run(operation=self.last_activation, feed_dict=feed_dict)
        return np.argmax(out, axis=1)

    def accuracy(self, Xs, labels):
        feed_dict = {
            self.X: Xs,
        }

        s = Session()
        out = s.run(self.last_activation, feed_dict=feed_dict)
        preds = np.argmax(out, axis=1)
        return np.mean(np.array(preds == labels, dtype=int))

    def predict(self, xs):
        feed_dict = {
            self.X: xs
        }

        s = Session()
        out = s.run(operation=self.last_activation, feed_dict=feed_dict)
        return out

    def loss(self, xs, truths, s: Session = None):
        session = Session()
        if s is not None:
            session = s
        session = session

        return session.run(self.loss_function, feed_dict={self.X: xs, self.c: truths})
