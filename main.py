from src.network import Sequential, Layer
import numpy as np

import src.graph

if __name__ == "__main__":
    # Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
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

    # Create a new graph
    net = Sequential([
        Layer(input_nodes=2, activation='sigmoid'),
        Layer(input_nodes=256, activation='sigmoid'),
        Layer(input_nodes=256, activation='sigmoid'),
        Layer(input_nodes=256, activation='sigmoid'),
        Layer(input_nodes=256, activation='sigmoid'),
        Layer(input_nodes=2, activation='softmax'),
    ])

    net.compile(learning_rate=0.0001, loss='adsa')

    net.fit(
        xs=np.concatenate((red_points, blue_points)),
        ys=np.array(truths)
    )
