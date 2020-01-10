#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Contains all the control strategies functions.
"""
import numpy as np
import networkx as nx
import collections
import random

########################################
######### Control strategies ###########
########################################

def random_airports_removal(gsc_SC_KM, nodes_statuses_pre_treatment, verbose, method_kwargs):
    """Remove random nodes (airports) of the graph."""
    if verbose:
        print('Perform random vaccination...')
    num_nodes_to_remove = int( gsc_SC_KM.number_of_nodes() * \
                              method_kwargs['nodes_percentage_to_treat'] / 100 )
    nodes_list = list(gsc_SC_KM.nodes)
    random.shuffle(nodes_list)
    nodes_to_remove = nodes_list[:num_nodes_to_remove]

    # get the graph and dict with these nodes removed
    treated_graph, nodes_statuses_post_treatment = \
        remove_nodes_from_graph_and_dict(nodes_to_remove, gsc_SC_KM,
                                         nodes_statuses_pre_treatment)
    # report changes in a dict
    treatment_info = method_kwargs.copy()
    treatment_info['num_nodes_to_remove'] = num_nodes_to_remove
    treatment_info['nodes_to_remove'] = nodes_to_remove

    return treated_graph, nodes_statuses_post_treatment, treatment_info

def random_neighbors_removal(gsc_SC_KM, nodes_statuses_pre_treatment, verbose, method_kwargs):
    """Remove random neighbor of random nodes (airports) of the graph."""
    if verbose:
        print('Perform selected vaccination...')
    num_nodes_to_remove = int( gsc_SC_KM.number_of_nodes() * \
                              method_kwargs['nodes_percentage_to_treat'] / 100 )
    nodes_list = list(gsc_SC_KM.nodes)
    random.shuffle(nodes_list)
    random_initial_nodes = nodes_list[:num_nodes_to_remove]

    # pick one random adjacent node from each of the initial nodes
    nodes_to_remove_set = set()
    for node_id in random_initial_nodes:
        neighbours_list = [neighbor for neighbor in gsc_SC_KM.neighbors(node_id)]
        random.shuffle(neighbours_list)
        nodes_to_remove_set.add(neighbours_list[0])
    nodes_to_remove = list(nodes_to_remove_set)

    # get the graph and dict with these nodes removed
    treated_graph, nodes_statuses_post_treatment = \
        remove_nodes_from_graph_and_dict(nodes_to_remove, gsc_SC_KM,
                                         nodes_statuses_pre_treatment)
    # report changes in a dict
    treatment_info = method_kwargs.copy()
    treatment_info['num_nodes_to_remove'] = num_nodes_to_remove
    treatment_info['nodes_to_remove'] = nodes_to_remove

    return treated_graph, nodes_statuses_post_treatment, treatment_info

def largest_airports_removal(gsc_SC_KM, nodes_statuses_pre_treatment, verbose, method_kwargs):
    """Remove nodes (airports) with the highest degree (number of connections)."""
    if verbose:
        print('Perform largest_airports_removals...')

    # get the node ids to remove
    num_nodes_to_remove = int( gsc_SC_KM.number_of_nodes() * \
                              method_kwargs['nodes_percentage_to_treat'] / 100 )
    degree_dict = {n: d for n, d in gsc_SC_KM.degree()}
    ordered_degree_dict = {key: val for key, val in sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)}
    nodes_to_remove = list(ordered_degree_dict.keys())[:num_nodes_to_remove]

    # get the graph and dict with these nodes removed
    treated_graph, nodes_statuses_post_treatment = \
        remove_nodes_from_graph_and_dict(nodes_to_remove, gsc_SC_KM,
                                         nodes_statuses_pre_treatment)
    # report changes in a dict
    treatment_info = method_kwargs.copy()
    treatment_info['num_nodes_to_remove'] = num_nodes_to_remove
    treatment_info['nodes_to_remove'] = nodes_to_remove

    return treated_graph, nodes_statuses_post_treatment, treatment_info


def largest_infected_airports_removal(gsc_SC_KM, nodes_statuses_pre_treatment,
                                       verbose, method_kwargs):
    """Remove infected nodes (airports) with the highest degree (number of connections)."""
    if verbose:
        print('Perform largest_infected_airports_removals...')

    # get the node ids to remove
    num_nodes_to_remove = int( gsc_SC_KM.number_of_nodes() * \
                              method_kwargs['nodes_percentage_to_treat'] / 100 )
    degree_dict = {n: d for n, d in gsc_SC_KM.degree()}
    ordered_degree_dict = {key: val for key, val in sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)}
    infected_ordered_degree_dict = {node_id: deg for node_id, deg in ordered_degree_dict.items() \
                                    if nodes_statuses_pre_treatment[node_id] == 'I'}
    nodes_to_remove = list(infected_ordered_degree_dict.keys())[:num_nodes_to_remove]

    # get the graph and dict with these nodes removed
    treated_graph, nodes_statuses_post_treatment = \
        remove_nodes_from_graph_and_dict(nodes_to_remove, gsc_SC_KM,
                                         nodes_statuses_pre_treatment)
    # report changes in a dict
    treatment_info = method_kwargs.copy()
    treatment_info['num_nodes_to_remove'] = num_nodes_to_remove
    treatment_info['nodes_to_remove'] = nodes_to_remove

    return treated_graph, nodes_statuses_post_treatment, treatment_info


def largest_routes_removal(gsc_SC_KM, nodes_statuses_pre_treatment, verbose, method_kwargs):
    """Remove edges (routes) with the highest weights (number of flights)."""
    if verbose:
        print('Perform largest_routes_removals...')

    # get the edges to remove
    num_edges_to_remove = int( gsc_SC_KM.number_of_edges() * \
                              method_kwargs['edges_percentage_to_treat'] / 100 )
    edges_dict = {edge: gsc_SC_KM.edges[edge]['weight'] for edge in gsc_SC_KM.edges}
    ordered_edges_dict = {key: val for key, val in sorted(edges_dict.items(), key=lambda item: item[1], reverse=True)}
    edges_to_remove = list(ordered_edges_dict)[:num_edges_to_remove]

    # get the graph and dict with these edges removed
    treated_graph = \
        remove_edges_from_graph(edges_to_remove, gsc_SC_KM)
    nodes_statuses_post_treatment = nodes_statuses_pre_treatment.copy()

    # report changes in a dict
    treatment_info = method_kwargs.copy()
    treatment_info['num_edges_to_remove'] = num_edges_to_remove
    treatment_info['edges_to_remove'] = edges_to_remove

    return treated_graph, nodes_statuses_post_treatment, treatment_info


#########################################################
# Helper functions for the control strategies functions #
#########################################################

def remove_nodes_from_graph_and_dict(nodes_to_remove, gsc_SC_KM, nodes_statuses_pre_treatment):
    """Remove nodes from graph and nodes_statuses dict."""
    treated_graph = gsc_SC_KM.copy()
    nodes_statuses_post_treatment = nodes_statuses_pre_treatment.copy()
    for node_id in nodes_to_remove:
        treated_graph.remove_node(node_id)
        del nodes_statuses_post_treatment[node_id]

    return treated_graph, nodes_statuses_post_treatment

def remove_edges_from_graph(edges_to_remove, gsc_SC_KM):
    """Remove edges from graph."""
    treated_graph = gsc_SC_KM.copy()
    for u, v in edges_to_remove:
        treated_graph.remove_edge(u, v)

    return treated_graph


