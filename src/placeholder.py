import numpy as np
import src.graph


class Placeholder:
    def __init__(self, name: str):
        self.consumers = []
        self.name = name
        self.output: np.ndarray = None
        src.graph._default_graph.placeholders.append(self)
