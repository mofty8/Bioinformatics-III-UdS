import tools
import math
from math import comb
from network_communities import decomposition
from network_communities import classify
from network_communities import rebuild
import sys
sys.float_info.max
from time import time
from random_network import RandomNetwork
from degree_distribution import DegreeDistribution
from reader import BioGRIDReader
from node import Node
import pandas as pd
import operator
from create_network import CreateNetwork
from generic_network import GenericNetwork
from scale_free_network import ScaleFreeNetwork
import heapq
import copy
from collections import defaultdict
from PSSM import  PSSM
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plotnetwork(title, network):
    plot_data = []
    plot_legend = []

    #print(title)

    # compute the normalised degree distribution histogram
    start = time()
    plot_data.append(DegreeDistribution(network).normalised)
    #plot_legend.append('r: {0:,}/{1:,}'.format(network.size(), n_edges))
    #print('\t--> computed degree distribution in {0:.2f}s'.format(time() - start))

    tools.plot_distribution_comparison_log(plot_data, plot_legend, title)


# NOTHING TO DO HERE
def plot(title, data):
    plot_data = []
    plot_legend = []

    print(title + ':')
    for n_nodes, n_edges in data:
        print('\tnodes: {0:,}, edges: {1:,}'.format(n_nodes, n_edges))
        # build random network
        start = time()
        network = RandomNetwork(n_nodes, n_edges)
        #network.print()
        print('Highest degree in the network ', network.max_degree())
        print('\t--> generated random network in {0:.2f}s'.format(time() - start))
        # compute the normalised degree distribution histogram
        start = time()
        plot_data.append(DegreeDistribution(network).normalised)
        plot_legend.append('r: {0:,}/{1:,}'.format(n_nodes, n_edges))
        print('\t--> computed degree distribution in {0:.2f}s'.format(time() - start))

        # build the histogram of the Poisson distribution
        start = time()
        plot_data.append(tools.poisson_histogram(n_nodes, n_edges, network.max_degree()))
        plot_legend.append('p: {0:,}/{1:,}'.format(n_nodes, n_edges))
        print('\t--> computed Poisson distribution in {0:.2f}s'.format(time() - start))
        print('/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/')

    tools.plot_distribution_comparison(plot_data, plot_legend, title)
def exercise_1b():
    network1 = ScaleFreeNetwork(10000, 2)
    network2 = ScaleFreeNetwork(100000, 2)

    plot1_data = []
    plot1_legend = []
    plot2_data = []
    plot2_legend = []

    plot1_data.append(DegreeDistribution(network1).normalised)
    plot1_legend.append('{0:,} nodes/{1:,} edges'.format(network1.size(), 20000))
    plot1_data.append(DegreeDistribution(network2).normalised)
    plot1_legend.append('{0:,} nodes/{1:,} edges'.format(network2.size(), 200000))

    tools.plot_distribution_comparison_log(plot1_data, plot1_legend,
                                           "Logarithmic degree distribution of scale-free networks")

    random_network = RandomNetwork(10000, 20000)

    plot2_data.append(DegreeDistribution(network1).normalised)
    plot2_legend.append('scale-free: {0:,} nodes/{1:,} edges'.format(network1.size(), 20000))
    plot2_data.append(DegreeDistribution(random_network).normalised)
    plot2_legend.append('random: {0:,} nodes/{1:,} edges'.format(network1.size(), 20000))

    tools.plot_distribution_comparison_log(plot2_data, plot2_legend,
                                           "Logarithmic degree distribution of a random network and a scale-free network")


def exercise_1c():
    network = ScaleFreeNetwork(10000, 2)
    empirical_dist = DegreeDistribution(network).normalised
    best_gamma = 10

    for gamma in range(10, 31):
        if (tools.KS_dist(empirical_dist,
                          tools.scale_free_distribution(network.max_degree(), gamma / 10.0)) < tools.KS_dist(
                empirical_dist, tools.scale_free_distribution(network.max_degree(), best_gamma / 10.0))):
            best_gamma = gamma

    best_gamma /= 10.0

    plot1_data = []
    plot1_legend = []

    plot1_data.append(empirical_dist)
    plot1_legend.append('scale-free network: {0:,} nodes/{1:,} edges'.format(network.size(), 20000))
    plot1_data.append(tools.scale_free_distribution(network.max_degree(), best_gamma))
    plot1_legend.append('power law distribution with gamma = ' + str(best_gamma))

    tools.plot_distribution_comparison_log(plot1_data, plot1_legend, "Exercise 1 c)")


def exercise_2b_and_c(bio_grid, id, n):
    bio_grid.network_size(id);
    bio_grid.most_abundant_taxon_ids(n)
    # TODO
    pass


def exercise_2e(bio_grid):
    df = bio_grid.df
    df = df[df['ORGANISM_A_ID'] == 9606]


    lista = df['INTERACTOR_A'].value_counts(sort=True, ascending=False)
    listb = df['INTERACTOR_B'].value_counts(sort=True, ascending=False)
    lista.to_dict()
    listb.to_dict()
    #print(lista, listb)
    #print(lista, listb)
    dict3 = {**lista, **listb}


    for key, value in dict3.items():
        if key in lista and key in listb:
            dict3[key] = value + lista[key]

    dict3 = dict(sorted(dict3.items(), key=operator.itemgetter(1), reverse=True))

    c = 0
    print('10 proteins with the highest degree in the human interaction network')
    for i in dict3.keys():
       if c < 10:
           print(i, dict3[i])
       c+=1




    #print(df['INTERACTOR_A'].value_counts())
    #print(df['INTERACTOR_B'].value_counts())


    pass


def exercise_2f(bio_grid):

    df = bio_grid.df
    df = df[df['ORGANISM_A_ID']==9606]

    lista = df['INTERACTOR_A'].tolist()
    listb = df['INTERACTOR_B'].tolist()


    listn = dict()


    for i in range(len(lista)):
        if lista[i] not in listn.values():
            listn[i] = lista[i]
    for i in range(len(listb)):
        if listb[i] not in listn.values():
            listn[i + len(lista)] = listb[i]



    humanInteraction = CreateNetwork()


    for i in listn.values():
        humanInteraction.nodes[i] = Node(i)


    for i in range(df.shape[0]):
        nodea = humanInteraction.nodes[lista[i]]
        nodeb = humanInteraction.nodes[listb[i]]


        nodea.neighbour_nodes.add(nodeb.identifier)
        nodeb.neighbour_nodes.add(nodea.identifier)

    plotnetwork("Human Interaction Network Logarithmic Degree Distribution", humanInteraction)

    # TODO
    pass




def shortestPath(graph, src, dest):

    vertices = []
    heapq.heappush(vertices, (0, src))
    #print(vertices)

    while len(vertices) != 0:
        currcost, currvtx = heapq.heappop(vertices)
        if currvtx == dest:
            print("Path Exisits {} to {} with cost {}".format(src, dest, currcost))
            return
        for neigh, neighcost in graph[currvtx]:
                heapq.heappush(vertices, (currcost + neighcost, neigh))
        #print(vertices)
    print("No Path Exisits from {} to {}".format(src, dest))

def create_grapph_directed():
    graph = defaultdict(list)
    v, e = map(int, input().split())
    for i in range(e):
        u, v, w = map(str, input().split())
        graph[u].append((v, int(w)))

    return graph
def create_grapph_undirected():
    graph = defaultdict(list)
    v, e = map(int, input().split())
    for i in range(e):
        u, v, w = map(str, input().split())
        graph[u].append((v, int(w)))
        graph[v].append((u, int(w)))

    return graph

def find_most_least(df,h2, h4, h24, n):


    df2 = pd.DataFrame(columns=['Gene', 'GoID'])
    df4 = pd.DataFrame(columns=['Gene', 'GoID'])
    df24 = pd.DataFrame(columns=['Gene', 'GoID'])
    for Gene in h2:
        df_ = df[df['Alt'] == Gene]
        df2 = df2.append(df_)
    for Gene in h4:
        df_ = df[df['Alt'] == Gene]
        df4 = df4.append(df_)
    for Gene in h24:
        df_ = df[df['Alt'] == Gene]
        df24 = df24.append(df_)


    counts2 = df2['GoID'].value_counts().to_dict()
    counts4 = df4['GoID'].value_counts().to_dict()
    counts24 = df2['GoID'].value_counts().to_dict()
    print("t=2h First ", n, "elements")
    for x in list(counts2)[0:n]:
         print("GoID {}, occurrences {} ".format(x, counts2[x]))
    print("t=2h Last ", n, "elements")
    for x in list(counts2)[len(counts2) - n :len(counts2)]:
        print("GoID {}, occurrences {} ".format(x, counts2[x]))

    print("t=4h First ", n, "elements")
    for x in list(counts4)[0:n]:
        print("GoID {}, occurrences {} ".format(x, counts4[x]))
    print("t=4h Last ", n, "elements")
    for x in list(counts4)[len(counts4) - n:len(counts4)]:
        print("GoID {}, occurrences {} ".format(x, counts4[x]))

    print("t=24h First ", n, "elements")
    for x in list(counts24)[0:n]:
        print("GoID {}, occurrences {} ".format(x, counts24[x]))
    print("t=24h Last ", n, "elements")
    for x in list(counts24)[len(counts24) - n:len(counts24)]:
        print("GoID {}, occurrences {} ".format(x, counts24[x]))

    dfAll = pd.DataFrame(columns=['Gene', 'GoID'])
    for Gene in h2:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    for Gene in h4:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    for Gene in h24:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    countsAll = dfAll['GoID'].value_counts().to_dict()
    print("Combined First ", n, "elements")
    for x in list(countsAll)[0:n]:
        print("GoID {}, occurrences {} ".format(x, countsAll[x]))
    print("combined Last ", n, "elements")
    for x in list(countsAll)[len(countsAll) - n:len(countsAll)]:
        print("GoID {}, occurrences {} ".format(x, countsAll[x]))

def hyperGeometric(df,h2, h4, h24, N, k):
    dfAll = pd.DataFrame(columns=['Gene', 'GoID'])
    for Gene in h2:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    for Gene in h4:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    for Gene in h24:
        df_ = df[df['Alt'] == Gene]
        dfAll = dfAll.append(df_)
    countsAll = dfAll['GoID'].value_counts().to_dict()
    n = len(h2) + len(h24) + len(h4) #no. of genes in differential sets
    m = df['Alt'].nunique() #no. of genes with inidicators(P) (already filtered)
    #N total no of distinct genes in dataset

    #math.factorial(m) / (math.factorial(i) * math.factorial(m-i))
    #math.factorial(N-m) / (math.factorial(n- i) * math.factorial(N - m - n + i))
    #math.factorial(N) / (math.factorial(n) * math.factorial( N - n))
    p_values = list()
    # throws overflow error
    """
    for i in range(k, min(n,m) + 1):
        p = ((math.factorial(m) / (math.factorial(i) * math.factorial(m-i)))*(math.factorial(N-m) / (math.factorial(n- i) * math.factorial(N - m - n + i)))) / (math.factorial(N) / (math.factorial(n) * math.factorial( N - n)))
        p_values.append(p)
    print(p_values)
    """


    for i in range(k, min(n, m) + 1):
        p = (comb(m,i) * comb(N-m, n - i)) / comb(N, n)
        p_values.append(p)
    p_values.sort()
    print("P values hypergeometric")
    print(p_values)
    return p_values


def adjust_pval(pvals):
    adjusted_pvals = dict()
    c = 0
    for i in pvals:
        adjusted_pvals[i] = len(pvals) * pvals[c]
        if 1.0 < adjusted_pvals[i]:
            adjusted_pvals[i] = 1.0
        c +=  1
    return adjusted_pvals

attractor = list()
def bool_net(start_state):
    visited_states = list()
    temp = "{0:b}".format(start_state)  # convert to bin
    weights = [16, 8, 4, 2, 1]
    #print("bin", temp)

    state_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    # assigning each charater it's bin value
    c = len(temp) - 1
    for x in sorted(state_dict):
        if c < 0:
            break
        state_dict[x] = temp[c]
        c -= 1
    #print("state_dict", state_dict)
    i = 0
    flag = 0
    visited_states.append(start_state)


    while (flag == 0):
        temp_state_dict = {'A': '0', 'B': '0', 'C': '0', 'D': '0', 'E': '0'}

        if ((state_dict['B'] == '0') and (state_dict['C'] == '1')):
            temp_state_dict['A'] = '1'

        if (state_dict['A'] == '1'):
            temp_state_dict['B'] = '1'

        if (state_dict['D'] == '1'):
            temp_state_dict['C'] = '1'

        if ((state_dict['B'] == '0') and ((state_dict['E'] == '1') or (state_dict['C'] == '1'))):
            temp_state_dict['D'] = '1'

        temp_state_dict['E'] = state_dict['E']

        state_dict = temp_state_dict

        c = len(weights) - 1
        state_int = 0
        for k in sorted(state_dict):
            state_int += int(state_dict[k]) * weights[c]
            c -= 1

        if (state_int in visited_states):
            visited_states.append(state_int)
            flag = 1
        else:
            visited_states.append(state_int)

    print("Visisted States", visited_states)

    index1, index2 = 0,0
    for i in range(len(visited_states)):
        for j in range(i,len(visited_states)):
            if (visited_states[i] == visited_states[j]) and (i != j) :
                index1, index2 = i, j



    orbit = list()
    print("Start State: ", start_state)
    for i in range (index1, index2):
        orbit.append(visited_states[i])
    print("orbit", orbit)
    print("length:", len(orbit))
    if (len(orbit) > 1):
        attractor.append(orbit)

    return state_int


if __name__ == '__main__':
    """
    seq = [['T', 'C', 'A', 'C', 'A', 'C', 'G', 'T', 'G', 'G', 'G', 'A'],
           ['G', 'G', 'C', 'C', 'A', 'C', 'G', 'T', 'G', 'C', 'A', 'G'],
           ['T', 'G', 'A', 'C', 'A', 'C', 'G', 'T', 'G', 'G', 'G', 'T'],
           ['C', 'A', 'G', 'C', 'A', 'C', 'G', 'T', 'G', 'G', 'G', 'G'],
           ['T', 'T', 'C', 'C', 'A', 'C', 'G', 'T', 'G', 'C', 'G', 'A'],
           ['A', 'C', 'G', 'C', 'A', 'C', 'G', 'T', 'T', 'G', 'G', 'T'],
           ['C', 'A', 'G', 'C', 'A', 'C', 'G', 'T', 'T', 'T', 'T', 'C'],
           ['T', 'A', 'C', 'C', 'A', 'C', 'G', 'T', 'T', 'T', 'T', 'C']]

    # seq = [['A', 'C', 'A', 'T'] ,['A', 'C', 'C', 'T'],['A', 'G', 'G', 'G'],['C', 'C', 'T', 'G'],['A', 'T', 'A', 'G'],['C', 'A', 'G', 'T']]
    prob = [0.325, 0.175, 0.175, 0.325]
    PSSM.get_frequency_matrix(seq)

    print("////////")
    PSSM.get_corr_frequency_matrix(seq, prob, 1)
    print("////////")
    PSSM.get_scoring_matrix(seq, prob, 1)
    print("////////")

    df = pd.read_csv('human_GO.gaf', sep='\t', header=0)

    df.columns = ['Database', 'AccessionNumber', 'Alt', 'Coulmn3', "GoID", "Coulmn5", "colum6", "colum7", "Indicator",
                  "Colum9", "Colum10", "Colum11", "Colum12", "Colum13", "Colum14", "Colum15", "Colum16"]
    total_N_Genes = df["Alt"].nunique()
    df = df[df['Database'] == 'UniProtKB']
    df = df[df['Indicator'] == 'P']
    # print(df[df["Alt"]== "ATF3"])

    h2 = pd.read_csv('differentially_expressed_at_t=2h.tsv', sep='\t', header=0)
    h2 = h2["GeneSymbol"].tolist()
    h4 = pd.read_csv('differentially_expressed_at_t=4h.tsv', sep='\t', header=0)
    h4 = h4["GeneSymbol"].tolist()
    h24 = pd.read_csv('differentially_expressed_at_t=2h.tsv', sep='\t', header=0)
    h24 = h24["GeneSymbol"].tolist()

    find_most_least(df, h2, h4, h24, 20)
    p_values = hyperGeometric(df, h2, h4, h24, total_N_Genes, 4)
    p_adj = adjust_pval(p_values)
    print("adjusted Hypergeometric")
    print(p_adj.values())
    
    exercise_1b()
    exercise_1c()

    # read the BioGRID database
    bio_grid_reader = BioGRIDReader('BIOGRID-ALL-3.4.159.tsv')
    bio_grid_reader.network_size(7227)
    bio_grid_reader.most_abundant_taxon_ids(5)
    print(bio_grid_reader.df)
    exercise_2b_and_c(bio_grid_reader, 9606, 5)
    exercise_2e(bio_grid_reader)
    exercise_2f(bio_grid_reader)
    GN = GenericNetwork('BIOGRID-ALL-3.4.159.tsv')
    bio_grid_reader.export_network(9606,'BIOGRID-ALL-3.4.159.tsv')

    plot('Plot 1', [(50, 100), (500, 1000), (5000, 10000), (50000, 100000)])
    plot('Plot 2', [(20000, 5000), (20000, 17000), (20000, 40000), (20000, 70000)])
    network = CreateNetwork()
    names = set()
    df = pd.read_csv('network.tsv', sep='\t', header = 0)
    df.columns = ['name1', 'name2']
    for i in range(df.shape[0]):
        names.add(df['name1'].iloc[i])
        names.add(df['name2'].iloc[i])

    for i in names:
        network.nodes[i] = Node(i)

    for i in range(df.shape[0]):
        network.nodes[df['name1'].iloc[i]].neighbour_nodes.add(network.nodes[df['name2'].iloc[i]].identifier)
        network.nodes[df['name2'].iloc[i]].neighbour_nodes.add(network.nodes[df['name1'].iloc[i]].identifier)


    decomoposed = decomposition(network)
    community = CreateNetwork()
    community.nodes = {'Catelyn' : network.nodes['Catelyn'], 'Lukas':network.nodes['Lukas'], 'Kate':network.nodes['Kate'], 'Jennifer' : network.nodes['Jennifer']}
    #print(classify(community))

    rebuild(decomoposed, network)
    
    
    print("enter 1 for a directed a graph or 2 for an undirected graph")
    a = int(input())
    if a == 1:
        print("enter No. of vertices, No. of edges and the edges ")
        graph = create_grapph_directed()
    elif a == 2:
        print("enter No. of vertices, No. of edges and the edges ")
        graph = create_grapph_undirected()
    #graph = create_grapph_undirected()

    src, dest = map(str, input().split())

    graphlist = graph
    for i in graphlist.keys():
        graph2 = copy.copy(graph)
        if src == i:
            continue
        shortestPath(graph2, src, i)

    """
    all_possible_states = dict()
    for i in range(0,(2**5)):
        all_possible_states[i] = bool_net(i)
        print('---------------------------––––---–––––––')

    print("Start State : Revisited State")
    print(all_possible_states)
    print('---------------------------––––---–––––––')
    for i in range(len(attractor)):
        print("Attractor:", i, attractor[i] )

