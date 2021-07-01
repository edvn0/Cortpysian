from matplotlib import pyplot as plt
from src.network import Sequential, Layer
import numpy as np

from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    mnist = fetch_openml('mammography', as_frame=False)

    y = np.array(mnist.target, dtype=np.int8)
    y[y < 0] = 0
    b = np.zeros((y.size, np.max(y, axis=0) + 1))
    b[np.arange(y.size), y] = 1
    x = mnist.data.reshape(-1, 6)

    # Create a new graph
    net = Sequential([
        Layer(input_nodes=6, activation='relu'),
        Layer(input_nodes=2, activation='tanh'),
        Layer(input_nodes=2, activation='tanh'),
        Layer(input_nodes=2, activation='tanh'),
        Layer(input_nodes=2, activation='softmax'),
    ])

    net.compile(learning_rate=0.001, optimizer='Adam',
                loss='categorical_cross_entropy', metrics=['accuracy', 'mse', 'mae'])

    stats = net.fit(
        xs=x,
        ys=b,
        epochs=100
    )

    truths = np.argmax(b, axis=1)
    print(net.accuracy(x, truths))

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
