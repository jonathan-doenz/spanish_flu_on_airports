#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Contains classes and functions necessary to run epidemics
simulation on graphs.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os
from graphs_processing import construct_nodes_clusters

# Epidemics on Networks package to easily run different epidemics simulations
import EoN

# Global variables
nested_dict = lambda: collections.defaultdict(nested_dict)
# compartment_to_color_dict = {'S': 'C0', 'I': 'orange', 'R': 'green'}
compartment_to_color_dict = {'S': '#009a80','I':'#ff2000', 'R':'gray'}

########################################
###### Epidemics simulation class ######
########################################

class EpidemicsSimulation():
    """Compute epidemics simulations on graph and offers different
    visualizations.

    Attributes
    ----------
    gsc_SC_KM (networkx Graph object): the graph on to which the epidemics simulation is applied
    method (EoN method) : the epidemics simulation method
    epidemics_simulation_kwargs (dict) : keyword arguments that `method` needs
    tau (float) : transmission rate parameter of the disease
    gamma (float) : recovery rate parameter of the disease
    initial_infecteds (list) : airports infected at the beginning of the simulation
    tmax (float) : maximum time of the simulation
    num_simulations (int) : number of simulations to run
    sim_kwargs (dict) : extra keyword arguments needed by `method`
    reducing_factor (float) : tune size of the bins for averge-binning
    verbose (bool) : prints steps of computation if true
    compute_average (bool) : compute bin-averages if true
    cluster_method_and_size_dict : cluster_method_and_size_dict
    simulation_runs_list (list) : EoN Simulation_Investigation objects
    clusters_time_series (dict) : keys [clustering_method][cluster_id][compartment][run]
    time_series (dict) : keys [compartment][run]
    average_final_number_of_infecteds (float) : average final number of infecteds
    std_final_number_of_infecteds : std of final number of infecteds
    nodes_clusters (dict) : keys [clustering_method][nodes_or_length][cluster_id]
    tmin (float) : optional, time at which the simulation starts
    transmission_weight : optional, tune 
    pos (dict) : optional, position of the nodes
    """

    def __init__(self, gsc_SC_KM, method, epidemics_simulation_kwargs,
                 cluster_method_and_size_dict=None, verbose=True):
        self.gsc_SC_KM = gsc_SC_KM
        self.method = method
        self.epidemics_simulation_kwargs = epidemics_simulation_kwargs
        self.tau = epidemics_simulation_kwargs['tau']
        self.gamma = epidemics_simulation_kwargs['gamma']
        self.initial_infecteds = epidemics_simulation_kwargs['initial_infecteds']
        self.tmax = epidemics_simulation_kwargs['tmax']
        self.num_simulations = epidemics_simulation_kwargs['num_simulations']
        self.sim_kwargs = epidemics_simulation_kwargs['sim_kwargs']
        self.reducing_factor = 2
        self.verbose = verbose
        if  'compute_average' in epidemics_simulation_kwargs.keys():
            self.compute_average = epidemics_simulation_kwargs['compute_average']
        else:
            self.compute_average = True
        self.cluster_method_and_size_dict = cluster_method_and_size_dict

        # Optional arguments
        if 'tmin' in epidemics_simulation_kwargs.keys():
            self.tmin = epidemics_simulation_kwargs['tmin']
        else:
            self.tmin = 0

        if 'transmission_weight' in epidemics_simulation_kwargs.keys():
            self.transmission_weight = epidemics_simulation_kwargs['transmission_weight']
        else:
            self.transmission_weight = None

        # attribute computed by function construct_nodes_clusters at
        # initializaton
        if cluster_method_and_size_dict is not None:
            self.nodes_clusters = construct_nodes_clusters(self.gsc_SC_KM, self.cluster_method_and_size_dict)
        else:
            self.nodes_clusters = None

        # make position dict if 'pos' is in the simulation kwargs
        if 'pos' in self.sim_kwargs.keys():
            # print('Computing nodes coordinates...')
            nodes_coords_inverted_dict = {node_id: (self.gsc_SC_KM.nodes[node_id]['longitude'], self.gsc_SC_KM.nodes[node_id]['latitude']) for node_id in gsc_SC_KM.nodes}
            self.pos = nodes_coords_inverted_dict
        else:
            self.pos = None

        # attributes computed after calling run()
        self.simulation_runs_list = None
        self.clusters_time_series = None
        self.time_series = None
        self.average_final_number_of_infecteds = None
        self.std_final_number_of_infecteds = None

    def __str__(self):
        strings_list = [
            '-----------------------------',
            '    SIMULATION PARAMETERS    ',
            '-----------------------------',
            'method: {}'.format(self.method.__name__),
            'tau = {}'.format(self.tau),
            'gamma = {}'.format(self.gamma),
            'tmax = {}'.format(self.tmax),
            'number of simulations = {}'.format(self.num_simulations),
            os.linesep + 'Initial infected airports:']

        for node_id in self.initial_infecteds:
            airport_name = self.gsc_SC_KM.nodes[node_id]['name']
            airport_degree = self.gsc_SC_KM.degree(node_id)
            strings_list.append('- node_id = {}, name = {}, degree = {}'\
                                .format(node_id, airport_name, airport_degree))

        # if self.sim_kwargs is not None:
        #     strings_list.append(os.linesep + 'Arguments of sim_kwargs:')
        #     for key in self.sim_kwargs.keys():
        #         strings_list.append('- {}'.format(key))
        #     # for key, value in self.sim_kwargs.items():
        #     #     strings_list.append('{}: {}'.format(key, value))

        if self.cluster_method_and_size_dict is not None:
            strings_list.append(os.linesep + 'Clustering methods:')
            for cluster_method, cluster_size_k in self.cluster_method_and_size_dict.items():
                strings_list.append(
                    '- cluster method: {}, cluster size k = {}'\
                    .format(cluster_method, cluster_size_k))

        return os.linesep.join(strings_list)

    def run(self, specified_tmax=None):
        """Run the simulations and compute the average time series if
        there are more than one simulations.
        """
        if specified_tmax is None:
            specified_tmax = self.tmax

        simulation_runs_list = []
        clusters_time_series = nested_dict()
        time_series = nested_dict()

        for r in range(self.num_simulations):
            sim = self.method(self.gsc_SC_KM, self.tau, self.gamma,
                               initial_infecteds=self.initial_infecteds,
                               tmax=self.tmax,
                               transmission_weight=self.transmission_weight,
                               return_full_data=True, sim_kwargs={'pos': self.pos})
            # define the colors of the plots
            sim.sim_update_color_dict(compartment_to_color_dict)

            simulation_runs_list.append(sim)

            # print("Extracting data from clusters' perspective...")
            for clustering_method, cluster_size_k in self.cluster_method_and_size_dict.items():
                for cluster_id in range(cluster_size_k):
                    t, sim_ts = sim.summary(nodelist=self.nodes_clusters[clustering_method]['cluster_nodes_dict'][cluster_id])
                    for compartment in sim_ts.keys():
                        clusters_time_series[clustering_method][cluster_id][compartment][r] = (t, sim_ts[compartment])

            # print("Extracting time series...")
            for compartment in sim_ts.keys():
                t, sim_ts = sim.summary()
                time_series[compartment][r] = (t, sim_ts[compartment])

        self.simulation_runs_list = simulation_runs_list


        if (self.num_simulations > 1) and self.compute_average:
            # compute mean and std for every cluster method and cluster_id
            if self.verbose:
                print('Computing average and std over the runs...')
            for clustering_method in clusters_time_series.keys():
                for cluster_id in clusters_time_series[clustering_method].keys():
                    for compartment in clusters_time_series[clustering_method][cluster_id].keys():
                        runs_ts_dict = clusters_time_series[clustering_method][cluster_id][compartment].copy()
                        clusters_time_series[clustering_method][cluster_id][compartment]['mean'], \
                        clusters_time_series[clustering_method][cluster_id][compartment]['std'] = \
                            compute_binned_avg_and_std(runs_ts_dict, self.reducing_factor)

            # compute mean and std over the runs for every compartment
            for compartment in time_series.keys():
                runs_ts_dict = time_series[compartment].copy()
                time_series[compartment]['mean'], time_series[compartment]['std'] = \
                    compute_binned_avg_and_std(runs_ts_dict, self.reducing_factor)

        # change time series from defaultdict to normal dict
        self.clusters_time_series = dict(clusters_time_series)
        self.time_series = dict(time_series)

        # compute average and std number of infected airports at tmax
        if self.num_simulations > 1:
            if self.compute_average:
                ts_mean = self.time_series['I']['mean'][1]
                ts_std = self.time_series['I']['std'][1]
                non_nan_indices = np.where(~np.isnan(ts_mean))
                self.average_final_number_of_infecteds = ts_mean[non_nan_indices[0][-1]]
                self.std_final_number_of_infecteds = ts_std[non_nan_indices[0][-1]]
            else:
                final_num_of_infecteds_list = []
                for r in range(self.num_simulations):
                    sim = self.simulation_runs_list[r]
                    (t, sim_ts) = sim.summary()
                    final_num_of_infecteds_list.append(sim_ts['I'][-1])
                self.average_final_number_of_infecteds = np.mean(final_num_of_infecteds_list)
                self.std_final_number_of_infecteds = np.std(final_num_of_infecteds_list)

        else:
            r = 0
            self.average_final_number_of_infecteds = self.time_series['I'][r][1][-1]

    def print_summary(self):
        """Print results of the simulations.
        """
        strings_list = [
            '-------------------------',
            '   Simulations summary   ',
            '-------------------------',
            'Number of simulations = {}'.format(self.num_simulations)]

        if self.num_simulations > 1:
            strings_list.append(
                'Average (std) number of airports infecteds at time {} = {:.1f} ({:.1f})'
                .format(self.tmax,
                        self.average_final_number_of_infecteds,
                        self.std_final_number_of_infecteds))
        else:
            strings_list.append(
                'Number of airports infecteds at time {} = {}'
                .format(self.tmax, self.average_final_number_of_infecteds))

        return print(os.linesep.join(strings_list))


    def plot_time_series(self, ax, error_method='se', legend_kwargs=None):
        """Plot time series of number of nodes in a given compartment.
        """
        for compartment in self.time_series.keys():
            if self.num_simulations > 1 and self.compute_average:
                t, y = self.time_series[compartment]['mean']
                _, y_std = self.time_series[compartment]['std']
                if error_method == 'se':
                    y_se = y_std / np.sqrt(self.num_simulations)
                    ax.fill_between(t, y - y_se, y + y_se, alpha=0.2)
                else:
                    ax.fill_between(t, y - y_std, y + y_std, alpha=0.2)

            else:
                t, y = self.time_series[compartment][0]

            # ax.plot(t, y, label=compartment)
            ax.plot(t, y, color=compartment_to_color_dict[compartment], label=compartment)

        # Possibility to specify legend properties like its location
        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        else:
            ax.legend()

        ax.set_xlabel('Time')
        ax.set_title("Time series")
        ax.set_ylabel('Number of nodes')


    def plot_clusters_time_series(self, ax, clustering_method, compartment, fractions=True,
                      clusters_list=[], error_method='se',
                                  show_percentages=True, legend_kwargs=None):
        """Plot time series of number of nodes from each cluster in a given compartment.
        """
        if not clusters_list: # if clusters_list is empty, consider every cluster
            clusters_list = self.clusters_time_series[clustering_method].keys()

        for cluster_id in clusters_list:

            if self.num_simulations > 1 and self.compute_average:
                t, y = self.clusters_time_series[clustering_method][cluster_id][compartment]['mean']
                _, y_std = self.clusters_time_series[clustering_method][cluster_id][compartment]['std']
                y_se = y_std / np.sqrt(self.num_simulations)
                if fractions:
                    yf = y / self.nodes_clusters[clustering_method]['cluster_length_dict'][cluster_id]
                    yf_std = y / self.nodes_clusters[clustering_method]['cluster_length_dict'][cluster_id]
                else:
                    yf = y
                    yf_std = y_std
                if error_method == 'se':
                    yf_se = yf_std / np.sqrt(self.num_simulations)
                    ax.fill_between(t, yf - yf_se, yf + yf_se, alpha=0.2)
                else:
                    ax.fill_between(t, yf - yf_std, yf + yf_std, alpha=0.2)

            else:
                t, y = self.clusters_time_series[clustering_method][cluster_id][compartment][0]
                if fractions:
                    yf = y / self.nodes_clusters[clustering_method]['cluster_length_dict'][cluster_id]
                else:
                    yf = y

            if show_percentages:
                cluster_percentage = 100 * self.nodes_clusters[clustering_method]['cluster_length_dict'][cluster_id] \
                    / self.gsc_SC_KM.number_of_nodes()
                percentage_string = ' ({:.1f}%)'.format(cluster_percentage)
            else:
                percentage_string = ''

            ax.plot(t, yf, label='cluster ' + str(cluster_id) + percentage_string)

        # Possibility to specify legend properties like its location
        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlabel('Time')
        ax.set_title("Time series of compartment '{}' for {} clusters"
                  .format(compartment, clustering_method))
        if fractions:
            ax.set_ylabel('Fraction of nodes')
        else:
            ax.set_ylabel('Number of nodes')

class EpidemicsSimulationWithControl(EpidemicsSimulation):
    """Compute epidemics simulations with control strategy.

    Attributes
    ----------
    time_of_treatment : float
        The time at which the control strategy is implemented.
    self.control_strategy = control_strategy
    self.time_of_treatment = time_of_treatment
    self.tmax_control = epidemics_simulation_kwargs['tmax']
    epidemics_simulation_kwargs['tmax'] = self.time_of_treatment
    EpidemicsSimulation.__init__(self, gsc_SC_KM, method,
                                 epidemics_simulation_kwargs,
                                 cluster_method_and_size_dict=cluster_method_and_size_dict,
                                 verbose=verbose)

    # attributes computed after calling run()
    self.graph_at_time_of_treatment = None
    self.time_series_post = None
    self.treatment_info_list = None
    self.episim_post_list = None
    self.average_final_number_of_infecteds = None
    self.std_final_number_of_infecteds = None
    """

    def __init__(self, gsc_SC_KM, method, epidemics_simulation_kwargs,
                 control_strategy, time_of_treatment,
                 cluster_method_and_size_dict=None, verbose=True):
        self.control_strategy = control_strategy
        self.time_of_treatment = time_of_treatment
        # initialize as EpidemicsSimulation parent instance with
        # time_of_treatment as tmax
        self.tmax_control = epidemics_simulation_kwargs['tmax']
        epidemics_simulation_kwargs['tmax'] = self.time_of_treatment
        EpidemicsSimulation.__init__(self, gsc_SC_KM, method,
                                     epidemics_simulation_kwargs,
                                     cluster_method_and_size_dict=cluster_method_and_size_dict,
                                     verbose=verbose)

        # attributes computed after calling run()
        self.graph_at_time_of_treatment = None
        self.time_series_post = None
        self.treatment_info_list = None
        self.episim_post_list = None
        self.average_final_number_of_infecteds = None
        self.std_final_number_of_infecteds = None

    def __str__(self):
        control_strategy_string = self.control_strategy.__str__()
        # treatment_strings = [
        #     'Control strategy',
        #     '----------------',
        #     'method: {}'.format(self.control_method.__name__),
        #     'time of treatment: {}'.format(self.time_of_treatment)]
        # for key in self.control_strategy_kwargs.keys():
        #     treatment_strings.append('{}: {}'.format(key, self.control_strategy_kwargs[key]))

        return super().__str__() + \
            os.linesep + os.linesep + \
            control_strategy_string + os.linesep + \
            'time of treatment: {}'.format(self.time_of_treatment)
            # os.linesep.join(treatment_strings)

    def run(self):
        """Run the simulations and compute the average time series if
        there are more than one simulations.
        """
        if self.verbose:
            print('Compute simulations until time of treatment...')
        super().run(specified_tmax=self.time_of_treatment)

        if self.verbose:
            print('Compute simulations from time of treatment to tmax...')
        # set kwargs for simulation from time_of_treatment to tmax
        post_simulation_kwargs = self.epidemics_simulation_kwargs
        post_simulation_kwargs['tmin'] = self.time_of_treatment
        post_simulation_kwargs['tmax'] = self.tmax_control - self.time_of_treatment
        post_simulation_kwargs['num_simulations'] = 1

        # For each simulation, construct a new EpidemicsSimulation object
        # to simulate evolution of the treated graph from time_of_treatment to tmax
        episim_post_list = []
        treatment_info_list = []
        for r in range(self.num_simulations):
            if self.verbose:
                print('- simulation r = {}...'.format(r))
            nodes_statuses_pre_treatment = self.simulation_runs_list[r] \
                .get_statuses(self.gsc_SC_KM.nodes(), self.time_of_treatment)

            # set intial conditions at time_of_treatment
            graph_post_treatment, nodes_statuses_post_treatment, treatment_info = \
                    self.control_strategy.run(self.gsc_SC_KM,
                                              nodes_statuses_pre_treatment,
                                              self.verbose)
            treatment_info_list.append(treatment_info)
            initial_infecteds = [node_id for node_id, status in
                                 nodes_statuses_post_treatment.items() if status == 'I']
            post_simulation_kwargs['initial_infecteds'] = initial_infecteds

            # run simulation post treatment
            epidemics_simulation_post = EpidemicsSimulation(graph_post_treatment, self.method,
                                post_simulation_kwargs,
                                cluster_method_and_size_dict=self.cluster_method_and_size_dict)
            epidemics_simulation_post.run()
            episim_post_list.append(epidemics_simulation_post)

        self.episim_post_list = episim_post_list
        self.treatment_info_list = treatment_info_list

        if self.verbose:
            print('Extract post treatment time-series...')
        time_series_post = nested_dict()
        clusters_time_series_post = nested_dict()

        for r in range(self.num_simulations):
            episim = self.episim_post_list[r]
            for compartment in episim.time_series.keys():
                time_series_post[compartment][r] = episim.time_series[compartment][0]
            clusters_time_series_post[r] = episim.clusters_time_series
            # runs_ts_dict[r] = episim.time_series

        if self.compute_average:
            if self.verbose:
                print('Computing average and std over the runs post treatment...')
            time_series_post['mean'], time_series_post['std'] = compute_binned_avg_and_std(time_series_post)
            # clusters_time_series_post['mean'], clusters_time_series_post['std'] = \
            #         compute_binned_avg_and_std(clusters_time_series_post)

        self.time_series_post = time_series_post
        self.clusters_time_series_post = clusters_time_series_post

        # compute average and std number of nodes infected at time tmax_control
        if self.num_simulations > 1:
            if self.compute_average:
                ts_mean = self.time_series['I']['mean'][1]
                ts_std = self.time_series['I']['std'][1]
                non_nan_indices = np.where(~np.isnan(ts_mean))
                self.average_final_number_of_infecteds = ts_mean[non_nan_indices[0][-1]]
                self.std_final_number_of_infecteds = ts_std[non_nan_indices[0][-1]]
            else:
                final_num_of_infecteds_list = []
                for r in range(self.num_simulations):
                    episim = self.episim_post_list[r]
                    sim = episim.simulation_runs_list[0]
                    (t, sim_ts) = sim.summary()
                    final_num_of_infecteds_list.append(sim_ts['I'][-1])
                self.average_final_number_of_infecteds = np.mean(final_num_of_infecteds_list)
                self.std_final_number_of_infecteds = np.std(final_num_of_infecteds_list)

        else:
            r = 0
            episim = self.episim_post_list[r]
            sim = episim.simulation_runs_list[0]
            self.final_number_of_infecteds = sim.time_series['I'][r][1][-1]

    def print_summary(self):
        strings_list = [
            '--------------------------------------',
            '   Simulations with control summary   ',
            '--------------------------------------',
            'Number of simulations = {}'.format(self.num_simulations)]

        if 'num_nodes_to_remove' in self.treatment_info_list[0].keys():
            strings_list.append(
                'Number of airports removed = {}'
                .format(self.treatment_info_list[0]['num_nodes_to_remove']))

        if 'num_edges_to_remove' in self.treatment_info_list[0].keys():
            strings_list.append(
                'Number of routes removed = {}'
                .format(self.treatment_info_list[0]['num_edges_to_remove']))

        if self.num_simulations > 1:
            strings_list.append(
                'Average (std) number of airports infected at time {} = {:.1f} ({:.1f})'
                .format(self.tmax_control, self.average_final_number_of_infecteds,
                        self.std_final_number_of_infecteds))
        else:
            strings_list.append(
                'Number of airports infecteds at time {} = {}'
                .format(self.tmax_control, self.final_number_of_infecteds))

        return print(os.linesep.join(strings_list))



    def plot_time_series(self, ax, error_method='se', legend_kwargs=None, time_sequence='both'):
        """Plot time series of number of nodes in a given
        compartment.
        """
        # plot sequence from t=0 to t=time_of_treatment for 'pre' and 'both'
        if (time_sequence == 'pre') or (time_sequence == 'both'):
            super().plot_time_series(ax, error_method, legend_kwargs)

        # plot sequence from t=time_of_treatment to tmax for 'post' and 'both'
        if (time_sequence == 'post') or (time_sequence == 'both'):
            for compartment in self.time_series.keys():
                if self.num_simulations > 1 and self.compute_average:
                    t, y = self.time_series_post[0][compartment]['mean']
                    _, y_std = self.time_series_post[0][compartment]['std']
                    if error_method == 'se':
                        y_se = y_std / np.sqrt(self.num_simulations)
                        ax.fill_between(t, y - y_se, y + y_se, alpha=0.2)
                    else:
                        ax.fill_between(t, y - y_std, y + y_std, alpha=0.2)

                else:
                    t0, y = self.time_series_post[compartment][0]
                    t = t0 + self.time_of_treatment

                ax.plot(t, y, label=compartment, color=compartment_to_color_dict[compartment])
                # if time_sequence == 'post':
                #     ax.plot(t, y, label=compartment, color=compartment_to_color_dict[compartment])
                # elif time_sequence == 'both':
                #     ax.plot(t, y, color=compartment_to_color_dict[compartment])

            if time_sequence == 'post':
                # Possibility to specify legend properties like its location
                if legend_kwargs is not None:
                    ax.legend(**legend_kwargs)
                else:
                    ax.legend()

                ax.set_xlabel('Time')
                ax.set_title('Time series')
                ax.set_ylabel('Number of nodes')

            elif time_sequence == 'both':
                ax.axvline(self.time_of_treatment, color='k', linestyle='--',
                           alpha=0.5)


class ControlStrategy():
    """Control strategies methods to counteract epidemics spread on the graph.
    
    Attributes
    ----------
    method: function
        The control strategy function.
    method_kwargs: dict
        The keyword arguments that the method needs.
    """

    def __init__(self, method, method_kwargs):
        self.method = method
        self.method_kwargs = method_kwargs

    def __str__(self):
        strings_list = [
            'Control strategy',
            '----------------',
            'method: {}'.format(self.method.__name__)]
        for key, val in self.method_kwargs.items():
            strings_list.append('{}: {}'.format(key, val))

        return os.linesep.join(strings_list)

    def run(self, *args):
        return self.method(*args, self.method_kwargs)


########################################
#### Epidemics simulation functions ####
########################################

def compute_binned_avg_and_std(runs_ts_dict, reducing_factor=2):
    """Compute time average and standard deviation of the number of nodes over the simulation runs.

    Parameters
    ----------
    runs_ts_dict : dict with keys [run_id] and tuple of numpy arrays (time, number of nodes)
        The dict with the time-series (value) of each simulation run (key).
    reducing_factor : float
        The factor determining the size of the bins to use to average the time
        series. The larger it is, the larger the bins.

    Returns
    -------
    (t_seq_average, avg_binned_signal) : tuple of numpy arrays each of size
    (n_intervals - 1)
        The binned-averaged time series.
    (t_seq_average, std_binned_signal) : tuple of numpy arrays each of size
    (n_intervals - 1)
        The binned-standard deviation time series.
    """
    # set the number of intervals as half of the mean of the average length of the time-series
    n_intervals = int( np.mean([len(runs_ts_dict[key][0]) for key in runs_ts_dict.keys()]) / reducing_factor )
    # set max time of the bins as the minimum of the maximum time of the time-series
    minmax_time_value = np.min([runs_ts_dict[key][0][-1] for key in runs_ts_dict.keys()])
    t_seq_average = np.linspace(0, minmax_time_value, num=n_intervals)

    # t_prev and t represent respectively the left and right limits of the current bin
    t_prev = 0.
    avg_binned_signal = np.zeros(n_intervals - 1)
    std_binned_signal = np.zeros(n_intervals - 1)
    for i, t in enumerate(t_seq_average[1:]):
        num_term_sum = 0
        signal_sum = 0
        signal_squared_sum = 0
        # loop over the runs
        for r in runs_ts_dict.keys():
            t_run, y_run = runs_ts_dict[r]
            mask = (t_run >= t_prev) & (t_run < t)
            num_term_sum += mask.sum()
            signal_sum += y_run[mask].sum()
            signal_squared_sum += np.sum(y_run[mask]**2)

        if num_term_sum > 0:
            average_signal = signal_sum / num_term_sum
            avg_binned_signal[i] = average_signal
            std_binned_signal[i] = np.sqrt( signal_squared_sum / num_term_sum - average_signal**2 )
        else:
            # if there is no term in the bin, set to values to nan
            avg_binned_signal[i] = np.nan
            std_binned_signal[i] = np.nan

        # update t_prev for next iteration
        t_prev = t

    return (t_seq_average[1:], avg_binned_signal), (t_seq_average[1:], std_binned_signal)
