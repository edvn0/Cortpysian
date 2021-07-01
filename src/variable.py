import src.graph


class Variable:
    """Variable for the network.
    """

    def __init__(self, initial_value=None, name: str = None, layer=None, optimizable=True):
        """Variable is something that can be calculated.

        Args:
            initial_value ([type], optional): [description]. Defaults to None.
            name (str, optional): [description]. Defaults to None.
        """
        self.value = initial_value
        self.consumers = []
        self.name = name
        self.output = None
        self.layer = layer
        self.optimizable = optimizable

        src.graph._default_graph.variables.append(self)
