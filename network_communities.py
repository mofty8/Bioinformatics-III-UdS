from node import Node
from create_network import  CreateNetwork

def triangles(i, j):
    """
    :param i: first node in the edge
    :param j: second node in the edge
    :return: number of triangles to which the edge between i and j contributes, if the edge exists at all
    """
    c = 0

    for n in i.neighbour_nodes:
        if n in j.neighbour_nodes:
            c += 1

    return c
    #raise NotImplementedError


def edge_clustering_coefficient(i, j):
    """
    :param i: first node in the edge
    :param j: second node in the edge
    :raise: ValueError if there is no undirected edge between i and j
    :return: edge clustering coefficient

    """
    # to handle division by zero problem
    if j.identifier not in i.neighbour_nodes:
        raise ValueError
    elif min(len(i.neighbour_nodes)-1, len(j.neighbour_nodes)-1) == 0:
        return float('inf')

    else:
        c = (triangles(i,j) + 1) / min(len(i.neighbour_nodes)-1, len(j.neighbour_nodes)-1)
        return c

    #raise NotImplementedError


def decomposition(network):
    """
    1. Construct a list of undirected edges. Make sure that there are no duplicates and that the edges are sorted for
       reproducibility. First make sure that the two nodes in the edge itself are sorted and then sort the list of
       edges. For example:
       [(Ben, Ellen), (Ben, John), (Catelyn, Eddard), (Catelyn, Jennifer)]
    2. Until all edges are delete:
       i.   Calculate the edge coefficient for each remaining edge.
       ii.  Find the edge with the smallest coefficient, store it in a list and then remove it. Print:
            Step X: removed node_1 --> node_2 with ECC: coefficient
       iii. Repeat i. and ii. until all edges are deleted.
    :param network: a network, e.g. a Network objects, or an edge list,...
    :return: list of edges in order of deletion
    """

    edges = list()
    removed_edges = list()

    for i in network.nodes.values():

        for y in i.neighbour_nodes:
            if ([i.identifier, y] not in edges) and ([y, i.identifier] not in edges):
                if i.identifier < y:
                    edges.append([i.identifier, y])
                else:
                    edges.append([y, i.identifier])

    sortedEdges = sorted(edges, key=lambda x: (x[0], x[1]))
    #print(sortedEdges)
    step = 1

    while(len(sortedEdges) != 0 ):
        min_c = edge_clustering_coefficient(network.nodes[sortedEdges[0][0]], network.nodes[sortedEdges[0][1]])
        index = 0
        for i in range (len(sortedEdges)):
            if edge_clustering_coefficient(network.nodes[sortedEdges[i][0]], network.nodes[sortedEdges[i][1]]) < min_c:
                min_c = edge_clustering_coefficient(network.nodes[sortedEdges[i][0]], network.nodes[sortedEdges[i][1]])
                index = i

        removed_edges.append([sortedEdges[index][0],sortedEdges[index][1]])
        print('Step ', step ,': removed' , sortedEdges[index][0] , ' --> ', sortedEdges[index][1] , 'with ECC:', min_c)
        step += 1
        network.nodes[sortedEdges[index][0]].remove_edge(network.nodes[sortedEdges[index][1]])
        network.nodes[sortedEdges[index][1]].remove_edge(network.nodes[sortedEdges[index][0]])
        del sortedEdges[index]
    return removed_edges




def classify(community):
    """
    Use the definitions by Radicchi to classify the community as strong or weak, or as not a community.
    :param community: e.g. a set or list of nodes in the community
    :return: the classification of the community ('strong', 'weak' or 'none')

    """
    kin = 0
    kout=0
    kOutSum = 0
    kInSumm = 0
    flag = False
    for i in community.nodes.values():

        for j in i.neighbour_nodes:
            if j in community.nodes.keys():
                kin += 1
            else:
                kout += 1
        if kout > kin:
            flag = True

        kOutSum += kout
        kInSumm += kin

    if flag == False:
        return "strong community"
    elif kOutSum > kInSumm:
        return "none"
    else:
        return "weak community"





    raise NotImplementedError

def generateNetwrok(namesList, network):
    communityNetwork = CreateNetwork()
    for x in namesList:
        communityNetwork.nodes[x] = network.nodes[x]
    return communityNetwork


def rebuild(edges, network):
    """
    Iterate over the edges in reverse order (last deleted edge first). In each iteration:
    i.   If the edge is not connected to an existing community, create a new one.
    ii.  If the edge has a single node in common with an existing community, then add the order node to the
         community as well.
    iii. If both nodes of the edge are already part of the same community, there is nothing to do. If they are,
         however, part of two different communities, merge the two communities into one. In this case, print the
         current communities with their classification:
         Step X:
            none: [Hanna, Peter]
            strong: [Catelyn, Eddard, Jennifer, Kate]
            strong: [Ben, Ellen, John]
    :param edges: list of deleted edges in order of deletion
    """
    community = list()
    flag = False
    index1, index2 = None, None
    step = 1
    #print((edges))

    for i in reversed(edges):

        if (not any(i[0] in subl for subl in community)) and not (any(i[1] in subl for subl in community)):
            community.append(i)

            print('step', step)
            for i in community:
                print(classify(generateNetwrok(i,network)), ':', i)
        else:
            for j in range(len(community)):

                if i[0]  in community[j]  and not any(i[1] in subl for subl in community):
                    community[j].append(i[1])

                    flag = True

                elif i[1]  in community[j] and not any(i[0] in subl for subl in community):
                    community[j].append(i[0])

                    flag = True

            if (flag == False):

                for index, lst in enumerate(community):
                    if i[0] in lst:
                        index1 = index
                for index, lst in enumerate(community):
                    if i[1] in lst:
                        index2 = index
                community[index1].append(community[index2])
                del community[index2]
            print('step', step)
            for i  in community:
                print(classify(generateNetwrok(i, network)),i)


        step += 1













