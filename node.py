class Node:
    def __init__(self, identifier):
        """
        Creates a Node-object with the given identifier.
        :param identifier: node ID
        """
        self.identifier = identifier
        # contains the identifiers of other nodes connected to this node
        self.neighbour_nodes = set()

    def __str__(self):
        """
        :return: string representation of the node identifier
        """
        return '{0}: {1}'.format(self.identifier, sorted(self.neighbour_nodes))

    def __eq__(self, node):
        """
        :param node: Node-object
        :return: True if the other node has the same identifier, False otherwise
        """

        if self.identifier == node.identifier:
            return True
        else:
            return False
        # TODO

    ##raise NotImplementedError

    def has_edge_to(self, node):
        """
        :param node: Node-object
        :return: True if this node has an edge to the other node, False otherwise
        """

        if node.identifier in self.neighbour_nodes:
            return True
        else:
            return False
        # TODO

    ##raise NotImplementedError

    def add_edge(self, node):
        """
        Adds an edge to the other node by adding it to the neighbour-nodes.
        :param node: Node-object
        """

        self.neighbour_nodes.add(node.identifier)
        # TODO

    ##raise NotImplementedError

    def remove_edge(self, node):
        """
        Removes the edge to the other node, if that edge exists, by removing the other node from the neighbour nodes.
        :param node: Node-object
        """

        if node.identifier in self.neighbour_nodes:
            self.neighbour_nodes.remove(node.identifier)
        # TODO

    ##raise NotImplementedError

    def degree(self):
        """
        :return: the degree of this node (= number of neighbouring nodes)
        """

        return len(self.neighbour_nodes)
        # TODO
    ##raise NotImplementedError
