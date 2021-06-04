from node import Node
from abstract_network import AbstractNetwork


class CreateNetwork(AbstractNetwork):
    def __init__(self):
        """
        Creates a network from a network file.
        :param file_path: path to the network file
        """
        # super() initialises the AbstractNetwork class from which GenericNetwork inherits its other functions
        # since we only want nodes and edges from the file, we want the network to be empty at first
        super().__init__(n_nodes=0, n_edges=0)

        # TODO: open the file and build the network

    def __create_network__(self, n_nodes, n_edges):
        """
        Overrides the method in AbstractNetwork to avoid NotImplementedErrors. There is nothing to do here.
        """
        pass
