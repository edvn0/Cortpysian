import pandas as pandas
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.network import Sequential, Layer
import numpy as np

from sklearn.datasets import fetch_openml

import pandas as pd

if __name__ == "__main__":
    mnist = fetch_openml('CIFAR_10', as_frame=False)

    y = np.array(mnist.target, dtype=np.int8)
    b = np.zeros((y.size, np.max(y, axis=0) + 1))
    b[np.arange(y.size), y] = 1
    x = mnist.data.reshape(-1, 3072)
    x /= 255.0

    # Create a new graph
    net = Sequential([
        Layer(input_nodes=3072, activation='relu'),
        Layer(input_nodes=256, activation='relu'),
        Layer(input_nodes=256, activation='relu'),
        Layer(input_nodes=256, activation='relu'),
        Layer(input_nodes=256, activation='relu'),
        Layer(input_nodes=10, activation='softmax'),
    ])

    net.compile(learning_rate=1e-4, optimizer='Adam', loss='categorical_cross_entropy')

    stats = net.fit(
        xs=x,
        ys=b,
        epochs=100
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    epochs = [e + 1 for e in range(stats['epochs'])]
    ax1.plot(epochs, stats['loss_epoch'])
    ax1.set(xlabel='epochs', ylabel='loss',
            title='Loss')
    ax1.grid()
    ax2.plot(epochs[1:], stats['time_epoch'][1:])

    ax2.set(xlabel='epochs', ylabel='time',
            title='Time')

    ax2.grid()
    plt.show()
