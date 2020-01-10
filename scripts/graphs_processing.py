#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: contains classes and functions to compute graph properties and
spectral clustering.
"""
import numpy as np
import networkx as nx
import collections
import os

# Spectral Clustering imports:
from scipy import linalg as LA
from sklearn.cluster import KMeans

# Global variables
nested_dict = lambda: collections.defaultdict(nested_dict)

########################################
######## Graph Properties class ########
########################################

class GraphProperties:
    """Compute and display basic graph properties.

    Attributes
    ----------
    graph (networkx Graph object)       : the graph
    number_of_nodes (int)               : number of nodes of the graph
    number_of_edges (int)               : number of edges of the graph
    number_of_connected_components (int): number of connected components of the graph
    degree_sequence (list)              : sequence of degrees of every node of the graph
    degree_count (dict)                 : count of each degree (degree: number of nodes with this degree)
    average_node_degree (float)         : average node degree of the graph
    average_degree_squared (float)      : average node squared degree <k^2> of the graph
    median_node_degree (float)          : median node degree of the graph
    average_neighbor_degree (float)     : average neighbor degree of the graph
    average_degree_connectivity (float) : average degree connectivity of the graph
    average_shortest_path (float)       : average shortest path of the graph
    average_clustering (float)          : average clustering of the graph
    diameter (int)                      : diameter of the graph
    """
    def __init__(self, graph):
        self.graph = graph
        self.number_of_nodes = None
        self.number_of_edges = None
        self.number_of_connected_components = None
        self.degree_sequence = None
        self.degree_count = None
        self.average_node_degree = None
        self.average_degree_squared = None
        self.median_node_degree = None
        self.average_neighbor_degree = None
        self.average_degree_connectivity = None
        self.average_shortest_path = None
        self.average_clustering = None
        self.diameter = None

    def compute_properties(self, comprehensive=False):
        """Compute properties of the graph.

        Parameter:
        ----------
        comprehensive : boolean
            If true, compute the computationally demanding properties.
        """
        self.number_of_nodes = self.graph.number_of_nodes()
        self.number_of_edges = self.graph.number_of_edges()
        self.number_of_connected_components = len( list(nx.connected_components(self.graph)) )
        self.degree_sequence = sorted([d for n, d in self.graph.degree()])
        self.degree_count = collections.Counter(self.degree_sequence)
        self.average_node_degree = np.mean(self.degree_sequence)
        self.average_degree_squared = np.mean(np.mean(np.array(self.degree_sequence)**2))
        self.median_node_degree = np.median(self.degree_sequence)
        self.average_clustering = nx.average_clustering(self.graph)
        # compute the following properties only if comprehensive=True
        # because they are computationally more demanding
        if comprehensive:
            print('Computing average neighbor degree...')
            average_neighbor_degrees = nx.average_neighbor_degree(self.graph)
            self.average_neighbor_degree = np.mean(list(average_neighbor_degrees.values()))
            print('Computing average degree connectivity...')
            average_degree_connectivities = nx.average_degree_connectivity(self.graph)
            self.average_degree_connectivity = np.mean(list(average_degree_connectivities.values()))
            if nx.is_connected(self.graph):
                print('Computing average shortest path...')
                self.average_shortest_path = nx.average_shortest_path_length(self.graph)
                print('Computing diameter...')
                self.diameter = nx.diameter(self.graph)

    def __str__(self):
        strings_list = [
            '--------------------------',
            '    NETWORK PROPERTIES    ',
            '--------------------------',
            'The network has {} nodes and {} edges.'.format(self.number_of_nodes, self.number_of_edges),
            'The network has {} connected components.'.format(self.number_of_connected_components),
            'The average node degree is {:.3f}.'.format(self.average_node_degree),
            'The average degree squared <k^2> is {:.3f}.'.format(self.average_degree_squared),
            'The median degree is {}.'.format(self.median_node_degree),
            'The average clustering coefficient is {:.3f}.'.format(self.average_clustering)]

        # display average_neighbor_degree, average_degree_connectivity, 
        # shortest path, diameter only if they have been computed
        if self.average_neighbor_degree is not None:
            if nx.is_connected(self.graph):
                strings_list_advanced = [
                    'The average _neighbor_ degree is {:.3f}.'.format(self.average_neighbor_degree),
                    'The average shortest path is {:.3f}.'.format(self.average_shortest_path),
                    'The diameter is {}.'.format(self.diameter),
                    'The average degree connectivity is {:.3f}.'.format(self.average_degree_connectivity)]
            else:
                strings_list_advanced = [
                    'The average _neighbor_ degree is {:.3f}.'.format(self.average_neighbor_degree),
                    'The average degree connectivity is {:.3f}.'.format(self.average_degree_connectivity)]

            strings_list = strings_list + strings_list_advanced

        return os.linesep.join(strings_list)

########################################
### Methods for Spectral Clustering ####
########################################

def spectral_decomposition(laplacian: np.ndarray):
    """ Compute eigenvalues and associated eigenvectors of laplacian.

    Parameter:
    ----------
    laplacian : square and symmetric np.ndarray
        A laplacian matrix.
    Returns:
    --------
    e : (np.array) of shape len(laplacian)
        The eigenvalues of the laplacian
    U : (np.ndarray) of shape (len(laplacian), len(laplacian))
        The corresponding eigenvectors. Eigenvector U[:, i] correspondig to eigenvalue e[i].
    """
    return LA.eigh(laplacian)

########################################
###### Spectral Clustering Class #######
########################################

class SpectralClustering():
    """Compute spectral clustering of graph."""
    def __init__(self, graph, normalize=True):
        self.graph = graph
        self.n_classes = None
        self.normalize = normalize
        self.laplacian = None
        self.e = None
        self.U = None
        self.y_pred = None
        self.clustering_method = None

    def fit(self):
        """ Compute laplacian matrix of the graph and its eigendecomposition."""
        print('Compute laplacian matrix of graph...')
        if self.normalize:
            self.laplacian = nx.normalized_laplacian_matrix(self.graph)
        else:
            self.laplacian = nx.laplacian_matrix(self.graph)

        print('Compute eigenvalues and eigenvectors of laplacian matrix...')
        self.e, self.U = spectral_decomposition(self.laplacian.todense())

    def predict(self, n_classes: int):
        """ Computes cluster assignments for n_classes clusters.

        Parameters
        ----------
        n_classes : int
           Number of clusters.

        Computes:
        ---------
        self.y_pred : numpy array of length self.graph.number_of_nodes()
            The assignments of nodes to the clusters.
        """
        self.n_classes = n_classes
        self.clustering_method = KMeans(n_clusters=n_classes)
        # compute cluster assignments
        if self.normalize:
            self.y_pred = self.clustering_method.fit_predict(self.U[:, :self.n_classes] / np.linalg.norm(self.U[:, :self.n_classes]))
        else:
            self.y_pred = self.clustering_method.fit_predict(self.U[:, :self.n_classes])

########################################
########## Clusters functions ##########
########################################

def construct_nodes_clusters(gsc_SC_KM, cluster_method_and_size_dict, verbose=False):
    """Construct a nested dict with nodes list and size associated with clusters.

    Parameters:
    -----------
    gsc_SC_KM : networkx Graph
        The graph on which to extract the clustering information.
    cluster_method_and_size_dict : dict with keys [clustering_method][cluster_size_k]
        The dict giving the value of parameter k to use with the associated key method.
    Returns:
    --------
    nodes_clusters : dict with keys [clustering_method][nodes_or_length][cluster_id]
        The dict containing the nodes and length of each cluster from the input graph and clustering methods.
    """
    if verbose:
        print('Computing nodes clusters...')
    nodes_clusters = nested_dict()

    for clustering_method, cluster_size_k in cluster_method_and_size_dict.items():
        # initialize nodes list to empty list
        for cluster_id in range(cluster_size_k):
            nodes_clusters[clustering_method]['cluster_nodes_dict'][cluster_id] = []

        # compute the list of nodes for each cluster_id
        for node_id in gsc_SC_KM.nodes():
            cluster_id = gsc_SC_KM.nodes[node_id][clustering_method + '_' + str(cluster_size_k).zfill(2)]
            nodes_clusters[clustering_method]['cluster_nodes_dict'][cluster_id].append(node_id)

        # compute the size of every list
        for cluster_id in range(cluster_size_k):
            nodes_clusters[clustering_method]['cluster_length_dict'][cluster_id] = \
                len(nodes_clusters[clustering_method]['cluster_nodes_dict'][cluster_id])

    return nodes_clusters

