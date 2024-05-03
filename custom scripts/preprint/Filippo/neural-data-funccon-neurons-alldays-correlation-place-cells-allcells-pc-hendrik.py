#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 03/05/2024 11:33
@author: hheise

"""
# this script only considers stable, and nonstable cell categories (all pairs of categories)
# importantly, only cells that are measured on all days are present.
import pickle
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

import pandas as pd

from preprint.Filippo.utilities import plot_quantiles_with_data, remove_unwanted_mice, df_corr_result, get_correlation_vector, \
    avg_over_columns, divide_pre_early_late

import argparse


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


def apply_function_to_cells(df1, df2, func, ignore_nan=False):
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
            nan_condition = ignore_nan and (isinstance(val1, float) and np.isnan(val1) or
                                            isinstance(val2, float) and np.isnan(val2))

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


if __name__ == '__main__':

    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dff', help='Use Delta F/F or deconvolved activity',
                        choices=['dff', 'decon'])
    args = parser.parse_args()

    savestr_base = f'code/08012024/all-cells/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)

    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/08012024/plotstyle.mplstyle')

    # dataset selection:
    if args.dataset == 'dff':
        traces_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-dff.pkl'
    elif args.dataset == 'decon':
        traces_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-decon.pkl'
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')

    with open(traces_path, 'rb') as file:
        traces_corrmat_dict = pickle.load(file)

    pc_division_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices\mouse-cell-pair-identifiers.pkl'
    with open(pc_division_path, 'rb') as file:
        pc_classes_matrix = pickle.load(file)

    pc_classes_matrix = remove_mice_from_df(pc_classes_matrix, unwanted_mice)

    traces_corrmat_dict_filtered = remove_unwanted_mice(traces_corrmat_dict,
                                                        unwanted_mice)  # remove mouse 121 and 63. 63 is removed here because it has too few cells that are tracked on all days
    filtered_corrmat_traces_df = df_corr_result(traces_corrmat_dict_filtered)
    traces_correlation_vectors = filtered_corrmat_traces_df.applymap(get_correlation_vector, na_action='ignore')

    """
    Insert: Map cell pairs based on their stable PC:
    0 if pair does not include stable PCs
    1 if one cell is stable PC
    2 if both cells are stable PCs
    
    Create correlation matrix shaped from all imaged cells of a session (e.g. 678x678 cells for M95, -11)
    Stable PCs have a 1 in their row/column, and double stable PC pairs have a 2 at their intersect point
    """



    # remap the cell pair categories to place cell and non-place cell
    remapped_final_pc_vec = pc_classes_matrix.applymap(get_correlation_vector, na_action='ignore')
    # compute dataframe of vectors for correlation statistic for every pair category:
    unique_categories = get_unique_cell_pair_categories(
        remapped_final_pc_vec)  # the ordering of these unique values applies to all contents of the
    string_pair_mapping = {0: 'non-coding-non-coding', 1: 'non-coding-place-cell', 2: 'place-cell-place-cell'}


    # derivative of the apply_function_to_cells function

    # calculate quantiles of correlation vectors (equivalently of correlation matrices)
    def get_quantile_means_of_distribution_pc_counts(quantile, traces_correlation_vectors, remapped_final_pc_vec,
                                                     qfunction=lambda x, y: x > y):
        correlation_vec_quantiles = traces_correlation_vectors.applymap(lambda cell: np.quantile(cell, quantile),
                                                                        na_action='ignore')
        corr_greater_than_quantile = apply_function_to_cells(traces_correlation_vectors, correlation_vec_quantiles,
                                                             qfunction, ignore_nan=True)

        # calculate fraction of cells that place cells and greater than the 0.0 quantile
        cellpairs_greater_than_quantile = apply_function_to_cells(corr_greater_than_quantile, remapped_final_pc_vec,
                                                                  lambda x, y: y[x], ignore_nan=True)

        cellcats_greater_than_quantile_counts = cellpairs_greater_than_quantile.applymap(
            lambda x: counts_from_uniques(x, unique_categories), na_action='ignore')
        cellcats_greater_than_quantile_fractions = cellcats_greater_than_quantile_counts.applymap(lambda x: x / x.sum(),
                                                                                                  na_action='ignore')

        # split fractions according to pre, early late
        cellcats_quantile_frac_pre_early_late_division = divide_pre_early_late(cellcats_greater_than_quantile_fractions)

        # calculate mean fractions by pair category over pre, early and late poststroke
        cellcats_quantile_frac_pre_early_late_meanlist = [
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[0], last3=True),
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[1]),
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[2])]  # results should sum to 1

        return cellcats_quantile_frac_pre_early_late_meanlist, cellcats_greater_than_quantile_fractions


    quantile = 0
    cellcats_quantile_frac_pre_early_late_meanlist_greater, cellcats_greater_than_quantile_fractions = get_quantile_means_of_distribution_pc_counts(
        quantile, traces_correlation_vectors, remapped_final_pc_vec)
    # make plots for coarse and for fine division
    length = len(cellcats_quantile_frac_pre_early_late_meanlist_greater[0].iloc[0])

    ylabel_list = ['Nc-Nc', 'Nc-Pc', 'Pc-Pc']
    for mouse_grouping, mouse_group_list, division_name in zip(
            [['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                           [no_recovery, recovery, sham]],
            ['coarse', 'fine']):
        # coarse division
        fig_coarse, axs_coarse = plt.subplots(length, 3, figsize=(16, 18), sharex=True, sharey=True)
        for i in range(length):
            fig_coarse = make_boxplots_pre_early_late(fig_coarse, axs_coarse[i], [arr.apply(lambda x: x[i]) for arr in
                                                                                  cellcats_quantile_frac_pre_early_late_meanlist_greater],
                                                      mouse_group_list, categories_str_list=mouse_grouping)

            axs_coarse[i, 0].set_ylabel(ylabel_list[i])

        axs_coarse[0, 0].set_title('Prestroke')
        axs_coarse[0, 1].set_title('Early ps')
        axs_coarse[0, 2].set_title('Late ps')

        fig_coarse.tight_layout(rect=(0, 0, 1, 0.95))
        fig_coarse.suptitle(f'Fractions of cell pairs above the {quantile} quantile, by pair category ({args.dataset})')
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-greater-{args.dataset}.png',
            dpi=300)

    # export data to csv
    datalist = [cellcats_quantile_frac_pre_early_late_meanlist_greater]
    datalist_renamed = []
    for l, name in zip(datalist, ['greater', 'smaller', 'greater divided by smaller']):
        for stat, period in zip(l, ['pre', 'early', 'late']):
            pairtype_stats = [stat.apply(lambda cell: cell[i]).rename((period, f'{name} quantile {quantile}', pairtype))
                              for i, pairtype in string_pair_mapping.items()]
            datalist_renamed.append(pairtype_stats)

    datalist_flat = [x for quantile_stat in datalist_renamed for x in quantile_stat]
    data_df = pd.concat(datalist_flat, axis=1)

    data_df.to_csv(f'{savestr}/cell-pair-fractions-in-correlation-distribution-quantile-{quantile}.csv')
