
import random
from node import Node
from abstract_network import AbstractNetwork

node_list = dict()
class RandomNetwork(AbstractNetwork):
    """
    This network class inherits basic network functionality from the AbstractNetwork-class. It overwrites the
    __create_network__ function to create a random network.
    """

    def __create_network__(self, n_nodes, n_edges):
        """
        Creates a random network with the specified number of nodes and edges.
        1. build a list of n nodes
        2. create m edges between randomly selected nodes that are not yet connected
        (Hint: when adding an edge A --> B, do not forget to also add the edge B --> A)
        :param n_nodes: number of nodes
        :param n_edges: number of edges
        """


        for i in range(n_nodes):

            self.nodes[i] = Node(i);

        for i in range(n_edges):
            qnode = random.choice(list(self.nodes.values()))
            pnode = random.choice(list(self.nodes.values()))
            if qnode != pnode:
                if qnode.identifier not in pnode.neighbour_nodes:

                    pnode.neighbour_nodes.add(qnode.identifier)
                    qnode.neighbour_nodes.add(pnode.identifier)


