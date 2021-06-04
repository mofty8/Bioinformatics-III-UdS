import csv
import pandas as pd
import operator
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
with open('mapping.txt') as f:
    NCBI = dict(x.rstrip().split(None, 1) for x in f)


class BioGRIDReader:
    """
    Reads a BioGRID database file.
    """
    def __init__(self, file_path):
        """
        :param file_path: path to the BioGRID file
        """
        df = pd.read_csv(file_path, sep='\t', header = 0)
        df.drop(['EXPERIMENTAL_SYSTEM', 'SOURCE', 'PUBMED_ID'], axis=1, inplace=True)

        #df[df[('OFFICIAL_SYMBOL_A' == 'OFFICIAL_SYMBOL_B')]]
        #df.drop(df[df[('OFFICIAL_SYMBOL_A' == 'OFFICIAL_SYMBOL_B')]])

        #df.drop(index = 5)
        #df.drop(temp, inplace=True)

        temp = df.query('ORGANISM_A_ID != ORGANISM_B_ID').index
        df.drop(index =temp, inplace=True)

        temp = df.query('OFFICIAL_SYMBOL_A == OFFICIAL_SYMBOL_B').index
        df.drop(index=temp, inplace=True)



        self.df = df
        #print(df)


        #tsv_read.info()

        #print(df[df['OFFICIAL_SYMBOL_A'] != df['OFFICIAL_SYMBOL_B']])


        #temp = df[df['ORGANISM_A_ID'] == 9606]
        #print(temp[['OFFICIAL_SYMBOL_A','OFFICIAL_SYMBOL_B']])




        # TODO
        #raise NotImplementedError

    def most_abundant_taxon_ids(self, n):
        """
        Compute the organisms with the most interactions in BioGRID.
        :param n: number of organisms
        :return: the n organisms with the most interactions and the respective number of interactions
        """
        tempdict = {}

        temp = self.df
        for i in NCBI:
            #print(temp[temp['ORGANISM_A_ID'] == int(i)].shape[0])
            tempdict[i] = temp[temp['ORGANISM_A_ID'] == int(i)].shape[0]

        sorted_dict = dict(sorted(tempdict.items(), key=operator.itemgetter(1), reverse=True))
        #print(sorted_dict)

        first_n= list(sorted_dict.keys())[:n]
        print('The most', n, 'abundant organisms', first_n)
        # TODO
        #raise NotImplementedError

    def network_size(self, taxon_id):
        """
        :param taxon_id: NCBI taxon ID of an organism
        :return: number of interactions for the specified organism
        :raise: KeyError if there is no organism with that ID
        """
        temp = self.df

        print('Network has ',temp[temp['ORGANISM_A_ID'] == taxon_id].shape[0], 'interactions')

        #return temp[temp['ORGANISM_A_ID'] == taxon_id].shape[0]
        # TODO
        #raise NotImplementedError

    def export_network(self, taxon_id, file_path):
        """
        Writes the interactions of the specified organism into the specified file.
        :param taxon_id: NCBI taxon ID of an organism
        :param file_path: path to the output network file
        :raise: ValueError if there is no organism with that ID

        """
        df = self.df
        df = df[df['ORGANISM_A_ID'] == id]

        df.to_csv('OrganismSpecific.csv')
        # TODO
        #raise NotImplementedError

