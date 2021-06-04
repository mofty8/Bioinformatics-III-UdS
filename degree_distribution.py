from abstract_network import AbstractNetwork



class DegreeDistribution:
    def __init__(self, network):
        """
        Computes the degree distribution of a network. Make sure that both degree 0 and the maximum degree
        are included!
        """
        # TODO: initialise a list to store the observations for each degree (including degree 0!)

        self.histogram = [0] * (network.max_degree() + 1)
        self.normalised = [0] * (network.max_degree() + 1)



        # TODO: fill the histogram with the degree distribution
        ...
        for i in network.nodes:
            self.histogram[len(network.nodes[i].neighbour_nodes)] += 1

        #print('Degree Distribution', self.histogram)

        # TODO: normalize with amount of nodes in network
        for i in range(len(self.histogram)):

            self.normalised[i] = self.histogram[i] / network.size()


        #print('Normalised Degree Distribution', self.normalised)

