from abstract_network import AbstractNetwork
from node import Node
import  random


class ScaleFreeNetwork(AbstractNetwork):
    def __create_network__(self, n_nodes, n_edges):
        """
        Creates a scale-free network using the Barabasi-Albert method.
        1. create 3 nodes and fully connect them
        2. add the remaining number of nodes:
            - for each node add the specified number of new edges
        Make sure that you do not create infinite loops, do not add self-edges and do not add edges more than once.
        :param n_nodes: total number of nodes to be created
        :param n_edges: number of edges to add for each new node
        """
        # TODO
        if n_nodes < 3:
            print("Number of nodes too small")
        else:
            self.nodes[1] = Node(1)
            self.nodes[2] = Node(2)
            self.nodes[3] = Node(3)
            for node in self.nodes.values():
                node.neighbour_nodes = set(self.nodes.keys())
                node.neighbour_nodes.discard(node.identifier)

            node_probability_list = [1, 1, 2, 2, 3, 3]

            for i in range(4, n_nodes + 1):
                self.nodes[i] = Node(i)
                for m in range(n_edges):
                    new_neighbour_id = random.choice(node_probability_list)
                    self.nodes[i].add_edge(self.nodes[new_neighbour_id])
                    self.nodes[new_neighbour_id].add_edge(self.nodes[i])
                    node_probability_list += [self.nodes[i].identifier, self.nodes[new_neighbour_id].identifier]
        #raise NotImplementedError
