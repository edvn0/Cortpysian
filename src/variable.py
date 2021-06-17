import src.graph


class Variable:
    """Variable for the network.
    """

    def __init__(self, initial_value=None, name: str = None, layer=None):
        """Variable is something that can be calculated.

        Args:
            initial_value ([type], optional): [description]. Defaults to None.
            name (str, optional): [description]. Defaults to None.
        """
        self.value = initial_value
        self.consumers = []
        self.name = name
        self.layer = layer

        src.graph._default_graph.variables.append(self)
