


class AbstractNetwork:
    """
    Abstract network definition. It cannot be instantiated (= you cannot create an AbstractNetwork-object).
    However, it can be inherited by other classes, which gives those other classes basic network functionality.
    """

    def __init__(self, n_nodes, n_edges):
        """
        Creates a network with the specified number of nodes and edges. There is nothing to do here.
        :param n_nodes: number of nodes
        :param n_edges: number of edges
        """
        # key = node ID, value = Node-object
        self.nodes = dict()
        # call the create network method of the respective sub-class
        self.__create_network__(n_nodes, n_edges)

    def __create_network__(self, n_nodes, n_edges):
        """
        Method overwritten by sub-classes. There is nothing to do here.
        :param n_nodes: number of nodes
        :param n_edges: number of edges
        """
        #rasise NotImplementedError


    def print(self):
        """
        Prints the network as a sorted adjacency list. For example:

        1: [2, 3]
        2: [1, 4]
        3: [1, 4]
        4: [2, 3]

        (Hint: you can make use of the __str__() function already implemented for each node.)

        """
        # mofty
        for key in self.nodes:
            print(key, ':', self.nodes[key].neighbour_nodes)

        # TODO
        #rasise NotImplementedError

    def size(self):
        """
        :return: number of nodes in the network

        """
        # mofty
        return len(self.nodes)

        # TODO
        #rasise NotImplementedError

    def add_node(self, node):
        """
        Adds a node to the network. If there is already a node with the same identifier in the network, simply
        overwrite it.
        :param node: Node-object
        """
        # mofty
        self.nodes[node.identifier] = node

        # TODO
        #rasise NotImplementedError

    def get_node(self, identifier):
        """
        :param identifier: node ID
        :return: node with the given identifier
        :#rasise: KeyError if there is no node with that ID in the network
        """
        # mofty
        return self.nodes[identifier]
        # TODO
        #rasise NotImplementedError

    def max_degree(self):
        """
        :return: highest node degree in the network
        """
        # mofty
        max = 0
        for key in self.nodes:
            if len(self.nodes[key].neighbour_nodes) > max:
                max = len(self.nodes[key].neighbour_nodes)
                # TODO

        return max
        #rasise NotImplementedError

    def add_edge(self, node_1, node_2):
        """
        Adds an undirected edge between the specified nodes.
        :param node_1: first node
        :param node_2: second node
        :raise: KeyError if either node is not in the network
        """
        if node_1.identifier not in self.nodes:
            raise KeyError('There is no node in the network with identifier:', node_1.identifier)
        if node_2.identifier not in self.nodes:
            raise KeyError('There is no node in the network with identifier:', node_2.identifier)
        node_1.add_edge(node_2)
        node_2.add_edge(node_1)

    def add_edge_by_id(self, id_1, id_2):
        """
        Adds an undirected edge between the specified nodes.
        :param id_1: identifier of the first node
        :param id_2: identifier of the second node
        :raise: KeyError if either node is not in the network
        """
        node_1 = self.get_node(id_1)
        node_2 = self.get_node(id_2)
        node_1.add_edge(node_2)
        node_2.add_edge(node_1)