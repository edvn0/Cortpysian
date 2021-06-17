import src.graph

from src.placeholder import Placeholder
from src.variable import Variable
from src.operation import Add, Log, MatMul, Multiply, Negative, ReduceSum, Sigmoid, Softmax
from src.session import Session
from src.gradient_descent import SGD

import numpy as np

# Create a new graph
g = src.graph.Graph()
g.as_default()

X = Placeholder("X")
c = Placeholder("c")

# Initialize weights randomly
W_0 = Variable(np.random.randn(2, 2), "W_0_(2, 2)")
b_0 = Variable(np.random.randn(2), "b_0_(2, 1)")

W_1 = Variable(np.random.randn(2, 5), "W_1_(2, 5)")
b_1 = Variable(np.random.randn(5), "b_1_(5, 1)")

W = Variable(np.random.randn(5, 2), "W_2_(2, 2)")
b = Variable(np.random.randn(2), "b_2_(2, 1)")

red_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 0]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 1]] * 25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 1]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 0]] * 25)
))

truths = [[1, 0]] * len(red_points) + [[0, 1]] * len(blue_points)

# Build perceptron
s = Sigmoid(Add(MatMul(X, W_0), b_0))
s2 = Sigmoid(Add(MatMul(s, W_1), b_1))
p = Softmax(Add(MatMul(s2, W), b))

# Build cross-entropy loss
J = Negative(ReduceSum(ReduceSum(Multiply(c, Log(p)), axis=1)))

# Build minimization op
minimization_op = SGD(learning_rate=0.01).fit(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
# Perform 100 gradient descent steps
for step in range(100):
    session = Session(J, feed_dict)

    J_value = session.run()
    session = Session(minimization_op, feed_dict=feed_dict)
    session.run()

# Print final result
