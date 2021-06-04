from node import Node
import pandas as pd
from abstract_network import AbstractNetwork
with open('mapping.txt') as f:
    NCBI = dict(x.rstrip().split(None, 1) for x in f)

class GenericNetwork(AbstractNetwork):
    def __init__(self, file_path):
        """
        Creates a network from a network file.
        :param file_path: path to the network file
        """
        # super() initialises the AbstractNetwork class from which GenericNetwork inherits its other functions
        # since we only want nodes and edges from the file, we want the network to be empty at first
        super().__init__(n_nodes=0, n_edges=0)
        df = pd.read_csv(file_path, sep='\t', header = 0)
        df.drop(['EXPERIMENTAL_SYSTEM', 'SOURCE', 'PUBMED_ID'], axis=1, inplace=True)

        temp = df.query('ORGANISM_A_ID != ORGANISM_B_ID').index
        df.drop(index =temp, inplace=True)

        temp = df.query('OFFICIAL_SYMBOL_A == OFFICIAL_SYMBOL_B').index
        df.drop(index=temp, inplace=True)

        self.df = df


        # TODO: open the file and build the network

    def __create_network__(self, n_nodes, n_edges):
        """
        Overrides the method in AbstractNetwork to avoid NotImplementedErrors. There is nothing to do here.
        """
        pass



