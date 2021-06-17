import src.graph


class Placeholder:
    def __init__(self, name: str):
        self.consumers = []
        self.name = name
        src.graph._default_graph.placeholders.append(self)
