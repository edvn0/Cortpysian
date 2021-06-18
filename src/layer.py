import abc
from abc import ABC
from typing import Union

from src.network import activation_registry
from src.operation import Operation


class Layer(ABC):
    def __init__(self, input_nodes: int, activation: Union[Operation, str], shape: tuple = ()):
        super(Layer, self).__init__()
        self.input_nodes = input_nodes
        self.shape = shape
        self.needs_initialization = False
        if type(activation) == str:
            z = activation_registry[activation]
            self.needs_initialization = True
        elif type(activation) == Operation:
            z = activation
        else:
            raise ValueError(
                "You need to provide either an operation or a str to initialize an activation")

        self.z = z
        self.activation = None

    @abc.abstractmethod
    def create_layer(self):
        raise NotImplementedError("Method is not implemented.")


class Dense(Layer):
    def create_layer(self):
        pass

    def __init__(self, input_nodes: int, activation: Union[Operation, str], shape: tuple = ()):
        super(Dense, self).__init__(input_nodes=input_nodes,
                                    activation=activation, shape=shape)
