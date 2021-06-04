import matplotlib.pyplot as plt
import math


def plot_distribution_comparison(histograms, legend, title):
    """
    Plots a list of histograms with matching list of descriptions as the legend.
    :param histograms: list of histograms
    :param legend: list of matching histogram names
    :param title: plot title
    """
    # TODO: determine length of the longest distribution
    #max_dist = max(histograms)
    # TODO: extend "shorter" distributions

    # plots histograms, nothing to do here
    for histogram in histograms:
        plt.plot(range(len(histogram)), histogram, marker='x')

    # TODO: axis labels
    plt.xlabel('degree')
    plt.ylabel('p')

    # finish the plot, nothing to do here
    plt.legend(legend)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def poisson(k, lamb):
    """
    :param k: observed events in an interval
    :param lamb: average number of events in an interval, (lambda is a Python keyword)
    :return: Poisson probability P(k) of observing k events in an interval
    """
    # TODO
    fact = 1
    for i in range(1, k+1):
        fact = fact * i
    poisson =(math.pow(lamb, k)/ fact)*math.exp(-lamb)
    return poisson
    #raise NotImplementedError


def poisson_histogram(n_nodes, n_edges, max_degree):
    """
    Generate a histogram of the Poisson distribution from degree 0 to max_degree.
    :param n_nodes: number of nodes
    :param n_edges: number of edges
    :param max_degree: maximum degree in the histogram
    :return: Poisson distribution histogram
    """
    # TODO
    poisson_dist = [None]*(max_degree + 1 )
    lamb = (2*n_edges)/n_nodes
    for i in range(max_degree + 1):

        poisson_dist[i] = poisson(i,lamb)

    print("test poisson", [ '%.2f' % elem for elem in poisson_dist ])

    return poisson_dist

    #raise NotImplementedError

def plot_distribution_comparison_log(histograms, legend, title):
    """
    Plots a list of histograms with matching list of descriptions as the legend with logarithmic axes.
    There is nothing to do here.
    :param histograms: list of histograms
    :param legend: list of matching histogram names
    :param title: plot title
    """
    # set the axes to logarithmic scale
    ax = plt.subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # determine max. length
    longest = max(len(histogram) for histogram in histograms)

    # extend "shorter" distributions
    for histogram in histograms:
        histogram.extend([0.0] * (longest - len(histogram)))

    # plots histograms
    for histogram in histograms:
        ax.plot(range(len(histogram)), histogram, marker='x', linestyle='')

    # axis labels
    plt.xlabel('degree')
    plt.ylabel('p')

    # finish the plot
    plt.legend(legend)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def scale_free_distribution(max_degree, gamma):
    """
    Generates a power law distribution histogram up to the maximum degree with slope gamma.
    :param max_degree: maximum degree
    :param gamma: slope
    :return: normalised power law histogram
    """
    # TODO
    #raise NotImplementedError
    histogram = [0] * (max_degree + 1)
    histogram[0] = 0
    c = 0

    for k in range(1, max_degree+1):
        histogram[k] = k ** (-1 * gamma)
        c += histogram[k]

    return [x / c for x in histogram]


def cumulative(dist):
    """
    Computes the cumulative distribution of a probabilistic distribution.
    :param dist: probabilistic distribution
    :return: cumulative distribution
    """
    # TODO
    #raise NotImplementedError
    cumulative_dist = [0] * len(dist)
    cumulative_dist[0] = dist[0]

    for i in range(1, len(dist)):
        cumulative_dist[i] = cumulative_dist[i-1] + dist[i]

    return cumulative_dist


def KS_dist(histogram_a, histogram_b):
    """
    Computes the Kolmogorov-Smirnov distance between two histograms.
    1. convert the histograms to cumulative distributions
    2. find the position where the cumulative distributions differ the most and return that distance
    :param histogram_a: first histogram
    :param histogram_b: second histogram
    :return: maximal distance
    """
    # TODO
    #raise NotImplementedError
    longest = max(len(histogram_a), len(histogram_b))
    if len(histogram_a) < longest:
        histogram_a.extend([0.0] * (longest - len(histogram_a)))
    else:
        histogram_b.extend([0.0] * (longest - len(histogram_b)))

    cumulative_a = cumulative(histogram_a)
    cumulative_b = cumulative(histogram_b)
    distance = 0

    for i in range(longest):
        if abs(cumulative_a[i] - cumulative_b[i]) > distance:
            distance = abs(cumulative_a[i] - cumulative_b[i])

    return distance