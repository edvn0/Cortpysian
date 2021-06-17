import numpy as np

from src.operation import Operation
from src.placeholder import Placeholder
from src.variable import Variable

import src.graph


def traverse_postorder(operation: Operation):
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:
    def __init__(self, operation, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}

        self.feed_dict = feed_dict
        self.operation = operation
        self.nodes_postorder = traverse_postorder(operation)

    def run(self):
        for node in self.nodes_postorder:
            if type(node) == Placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = self.feed_dict[node]
            elif type(node) == Variable:
                # Set the node value to the variable's value attribute

                node.output = node.value
            elif isinstance(node, Operation):  # Operation
                # Get the input values for this operation from the output values of the input nodes
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)
            else:
                raise ValueError("Not an implemented node type.")

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        return self.operation.output
