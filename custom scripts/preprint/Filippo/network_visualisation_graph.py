#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15/05/2024 22:32
@author: hheise

"""

# this script only considers stable, and nonstable cell categories (all pairs of categories)
# importantly, only cells that are measured on all days are present.
import pickle
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import ListedColormap
import networkx as nx
# matplotlib.use('Agg')

import pandas as pd
import scipy as sp

import sys

sys.path.append('../')
from preprint.Filippo.utilities import plot_quantiles_with_data, remove_unwanted_mice, remove_unwanted_mice_df, df_corr_result, \
    get_correlation_vector, avg_over_columns, divide_pre_early_late

import argparse

np.random.seed(42)

"""
Email by Hendrik Heiser on 19.11.2023:
Wir haben uns übrigens auf eine Gruppeneinteilung für das Paper geeinigt, was jetzt hoffentlich bestehen bleibt.
Da die Sphere-Zahlen in den Sham und No Deficit Mäusen nicht signifikant unterschiedlich ist,
haben wir uns entschlossen, die beiden Gruppen im ganzen Paper unter "Sham" zusammenzuschmeissen. Das bedeutet,
dass die Zweier-Gruppierung (Stroke - Sham) bestehen bleibt, und die Vierer-Gruppe zur Dreier-Gruppe wird
(Sham - Recovery - No Recovery). Die Einteilung der Mäuse in die einzelnen Gruppen bleibt aber wie bisher.
So kannst du das in deinen Auswertungen auch übernehmen.

Exclude Moude 121

Coarse groups:
'Stroke' (deficit, Recovery + No Recovery groups): [33, 41, 63, 69, 85, 86, 89, 90, 110, 113]
'Control' (no deficit, Sham + No Deficit groups): [83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

Fine groups:
'Recovery' (performance below "early", but not "late" threshold): [85, 86, 89, 90, 113],
'No Recovery' (performance below "early" and "late" threshold): [41, 63, 69, 110],
'Sham' (No deficit, and control): [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

#####################################
#BEWARE!!
#the following plots are not computed with all mice, because some mice don't appear in the
#calcium imaging dataframe or have too few neuron pairs that are measured over all poststroke sessions (63)
#or the mouse does not appear in the calcium imaging dataframe (112)!!!

#################################

"""
control = [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

# remove unwanted mice
control.remove(112)
sham.remove(112)

unwanted_mice = [121, 112]


def get_pair_correlation_statistic_cell(correlation_vector, category_vector, category=1, callable=np.mean):
    return callable(correlation_vector[category_vector == category])


def get_pair_correlation_statistic_series(paircorrs, paircats, category=1, callable=np.mean):
    correlation_pair_statistic = {}
    for mouse, cvec, catvec in zip(paircorrs.index, paircorrs, paircats):
        print(f"mouse: {mouse}")
        correlation_pair_statistic[mouse] = get_pair_correlation_statistic_cell(cvec, catvec, category=category,
                                                                                callable=callable)

    return pd.Series(correlation_pair_statistic, name=f'category {category} statistic')


def filter_cell_idx(dff, mouse_index_df_stable):
    # filter the cell indices of e.g. stable cells by those of cells that are imaged on all sessions
    filtered_cell_idx = {}
    for mouse in dff.keys():
        filtered_cell_idx[mouse] = np.intersect1d(dff[mouse].index, mouse_index_df_stable[mouse])
    return filtered_cell_idx


def remap_cell_pairs(cell, mapping_dict):
    return np.vectorize(mapping_dict.get)(cell)


def remove_mice_from_df(df, mice, axis=0):
    filtered_df = df
    for m in mice:
        if m in df.index:
            filtered_df.drop(m, axis=axis, inplace=True)
    return filtered_df


def get_unique_cell_pair_categories(pc_df):
    unique_values_in_cells = pc_df.applymap(lambda x: np.unique(x), na_action='ignore')
    flattened_uniques_withnans = unique_values_in_cells.apply(lambda col: np.hstack(col))
    unique_colvals = flattened_uniques_withnans.apply(np.unique)
    valarray_colvals = np.hstack(unique_colvals)
    unique_general = np.unique(valarray_colvals)
    nonan_unique_final = unique_general[~np.isnan(unique_general)].astype(int)
    return nonan_unique_final


def apply_function_to_cells(df1, df2, func, ignore_nan=False, ignore_inside=False):
    """
    Apply a function to corresponding cells of two dataframes.
    Assumes df1 and df2 have equal shapes!

    Parameters:
    - df1, df2: Input dataframes.
    - func: Function to apply to corresponding cells.
    - ignore_nan: If True, ignores np.NaN values in cells.

    Returns:
    - Resulting dataframe.
    """
    result_data = []
    rows, cols = df1.shape

    for i in range(rows):
        row_data = []
        for j in range(cols):
            val1, val2 = df1.iat[i, j], df2.iat[i, j]

            # Check if any value is NaN of type float
            if not ignore_inside:
                nan_condition = ignore_nan and (isinstance(val1, float) and np.isnan(val1) or
                                                isinstance(val2, float) and np.isnan(val2))
            else:
                nan_condition = ignore_nan and (np.any(np.isnan(val1)) or
                                                np.any(np.isnan(val2)))
            if nan_condition:
                cell_result = np.NaN
            else:
                cell_result = func(val1, val2)

            row_data.append(cell_result)

        result_data.append(row_data)

    result_df = pd.DataFrame(result_data, index=df1.index, columns=df1.columns)
    return result_df


def get_statistic_forall_categories(correlation_vector, category_vector, unique_categories, callable=np.mean):
    res = []
    for cat in unique_categories:
        try:
            res.append(get_pair_correlation_statistic_cell(correlation_vector, category_vector, category=cat,
                                                           callable=callable))
        except:
            res.append(np.NaN)
    return np.array(res)


def make_boxplots_pre_early_late(fig, axs, statistic_df_list, mouse_groups_coarse,
                                 categories_str_list=['control', 'stroke'],
                                 titlestr_list=[None, None, None]):
    # cb: callable
    # cstring: string of the callable

    for i, titlestr, statistic in zip(range(len(axs)), titlestr_list, statistic_df_list):
        data_arrays = [statistic.loc[group].dropna() for group in mouse_groups_coarse]

        axs[i] = plot_quantiles_with_data(axs[i], categories_str_list,
                                          data_arrays, titlestr=titlestr)

    return fig


def counts_from_uniques(cell, values):
    return np.array([np.sum(cell == v) for v in values])


def plot_cell(ax, dvec, corrvec, mouse, day):
    if isinstance(dvec, np.ndarray) and isinstance(corrvec, np.ndarray):
        ax.scatter(dvec, corrvec, marker='x', c='k', s=1)
        ax.set_title(f"Mouse {mouse}, day {day}", fontsize=14)
    else:
        ax.axis('off')  # Hide axis for NaN values


def replace_empty_with_nan(value):
    if isinstance(value, np.ndarray) and value.size == 0:
        return np.nan
    return value


def corr_if_not_len_1(x, y):
    if (len(x) == 1) or (len(y) == 1):
        return np.NaN
    else:
        return np.array(sp.stats.pearsonr(x, y))


def pick_random_sample_indices(cell, minlength):
    l = len(cell)
    if l <= minlength:
        return np.arange(l)
    else:
        return np.random.choice(np.arange(l), size=minlength, replace=False)


def plot_graph(graph, positions, node_cmap, edge_cmap, weight_offset, scale=100):
    weightlist = np.array(list(nx.get_edge_attributes(graph, 'weight').values())) + weight_offset
    alpha_from_weights = (weightlist - np.min(weightlist)) / (np.max(weightlist) - np.min(weightlist))

    node_type_array = np.array(list(nx.get_node_attributes(graph, 'cell-type').values()))
    edge_type_array = np.array(list(nx.get_edge_attributes(graph, 'conn-type').values()))

    fig, ax = plt.subplots()
    nodes = nx.draw_networkx_nodes(graph, pos=positions,
                                   ax=ax,
                                   node_size=12,
                                   node_color=node_type_array,
                                   cmap=node_cmap,
                                   edgecolors='k',
                                   linewidths=0.1)

    edges = nx.draw_networkx_edges(graph, pos=positions,
                                   ax=ax,
                                   width=weightlist,
                                   edge_cmap=edge_cmap,
                                   edge_color=edge_type_array,
                                   alpha=alpha_from_weights)

    ax.axis('off')

    scalebar = AnchoredSizeBar(ax.transData,
                               scale, f'{scale} um', 'lower left',
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=1)

    ax.add_artist(scalebar)

    return fig, ax


def construct_graph(cmat, pc_div, edges_types, quantile=0.8):
    # cmat: correlation matrixto serve as adjacency matrix. ignore self-loops
    # quantile: weights (cmat entries) below this quantile will not be included as edges
    cmat_qt = np.quantile(get_correlation_vector(cmat), quantile)
    cmat_greater_qt = np.copy(cmat)
    cmat_greater_qt[cmat <= cmat_qt] = np.NaN
    idx = np.indices(edges_types.shape)

    g = nx.Graph(cmat_greater_qt)
    nx.set_node_attributes(g, {u: pc_div[u] for u in range(len(pc_div))}, name='cell-type')  # set cell type
    nx.set_edge_attributes(g, {(u, v): edges_types[u, v] for u, v in zip(idx[0].flatten(), idx[1].flatten())},
                           name='conn-type')  # set type of connection
    g.remove_edges_from(nx.selfloop_edges(g))  # remove self loops
    g.remove_edges_from([(u, v) for u, v, data in g.edges(data=True) if np.isnan(data['weight'])])
    return g


def plot_graph_mouse_session(cmat, pc_div, edgetype, quantile=0.8, weight_offset=0,
                             node_cmap=ListedColormap(['#8C8C8C', '#68BC6B']),
                             edge_cmap=ListedColormap(['#8C8C8C', '#FFCE17', '#68BC6B']),
                             legend=True):
    graph = construct_graph(cmat, pc_div, edgetype, quantile=quantile)
    fig, ax = plot_graph(graph, positions, weight_offset=weight_offset,
                         node_cmap=node_cmap,
                         edge_cmap=edge_cmap)
    ax.set_title(f'Mouse {sample_mouse}, day {sample_session}, corr. quantile = {quantile}')

    # generate legend
    if legend:
        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='none', color=node_cmap(0), lw=1, markeredgewidth=0.2,
                   markeredgecolor='k', label='Nc'),
            Line2D([0], [0], marker='o', linestyle='none', color=node_cmap(1), lw=1, markeredgewidth=0.2,
                   markeredgecolor='k', label='Pc'),
            Line2D([0], [0], color=edge_cmap(0), lw=1, label='Nc-Nc'),
            Line2D([0], [0], color=edge_cmap(1), lw=1, label='Nc-Pc'),
            Line2D([0], [0], color=edge_cmap(2), lw=1, label='Pc-Pc')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)

    # fig.savefig(f'{savestr}/visualisation-graph-neurons-mouse-{sample_mouse}-day-{sample_session}.png')
    # fig.savefig(f'{savestr}/visualisation-graph-neurons-mouse-{sample_mouse}-day-{sample_session}.svg')

    return fig


def remove_unwanted_mice(dict, mice):
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy


class Arguments:
    def __init__(self, dset):
        self.dataset = dset


if __name__ == '__main__':

    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]

    # command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', type=str, default='dff', help = 'Use Delta F/F or deconvolved activity',
    #                    choices = ['dff', 'decon'])
    # args = parser.parse_args()

    args = Arguments('dff')

    savestr_base = f'code/08012024/all-cells/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)

    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/08012024/plotstyle.mplstyle')

    # dataset selection:
    if args.dataset == 'dff':
        traces_corrmat_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-dff.pkl'
    elif args.dataset == 'decon':
        traces_corrmat_path = 'code/08012024/all-cells/outputs/neural-data-funccon-decon-dff-calc-corrmat-distancemat/correlation-mat-unsorted-decon.pkl'
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')

    with open(traces_corrmat_path, 'rb') as file:
        traces_corrmat_dict = pickle.load(file)

    pc_division_path =  r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/mouse-cell-pair-identifiers.pkl'
    with open(pc_division_path, 'rb') as file:
        pc_classes_pairs_matrix = pickle.load(file)

    pc_classes_pairs_matrix = remove_mice_from_df(pc_classes_pairs_matrix, unwanted_mice)

    pc_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\is_pc_all_cells.pkl'
    with open(pc_path, 'rb') as file:
        pc_classes = pickle.load(file)
    pc_classes = {k: v.reset_index(drop=True) for k, v in pc_classes.items()}
    pc_classes = remove_unwanted_mice(pc_classes, unwanted_mice)

    # load coordinates
    coords_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\cell_coords_all_cells.pkl'
    with open(coords_path, 'rb') as file:
        coords = pickle.load(file)
    coords = {k: v.reset_index(drop=True) for k, v in coords.items()}
    coords = remove_unwanted_mice(coords, unwanted_mice)

    # get offdiagonals of correlation matrices
    traces_corrmat_dict_filtered = remove_unwanted_mice(traces_corrmat_dict, unwanted_mice)  # remove mouse 121 and 112.
    filtered_corrmat_traces_df = df_corr_result(traces_corrmat_dict_filtered)

    # remap the cell pair categories to place cell and non-place cell
    remapped_final_pc_vec = pc_classes_pairs_matrix.applymap(get_correlation_vector, na_action='ignore')
    # compute dataframe of vectors for correlation statistic for every pair category:
    unique_categories = get_unique_cell_pair_categories(
        remapped_final_pc_vec)  # the ordering of these unique values applies to all contents of the
    # derivative of the apply_function_to_cells function

    # data preparation
    sample_mouse = 115
    sample_session = 3
    cmat = filtered_corrmat_traces_df.loc[sample_mouse, sample_session]

    positions = coords[sample_mouse][sample_session].dropna().reset_index(drop=True)
    pc_div = pc_classes[sample_mouse][sample_session].dropna().reset_index(drop=True)
    edgetype = pc_classes_pairs_matrix.loc[sample_mouse, sample_session]  # could colr edges by cell pair types

    fig = plot_graph_mouse_session(cmat, pc_div, edgetype, quantile=0.95, weight_offset=0.25)

    # this should be done with tracked cells across multiple days to visualise the change in neuronal rewiring!
