#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 02/08/2024 09:19
@author: hheise

Basically script
"https://github.com/flmuk/wahl-colab/blob/e10c6c353c386ff92352970711ad417ee82670bf/code/08012024/all-cells/neural-data-funccon-neurons-alldays-correlation-place-cells-allcells-quantile-pc.py"
from Filippos Github, manually updated to check for place field locations.
Uses all cells (since PC classification is done daily), not tracked cells.
"""

# this script only considers stable, and nonstable cell categories (all pairs of categories)
# importantly, only cells that are measured on all days are present.
import pickle

import numpy as np

from time import perf_counter
from datetime import timedelta
import itertools

import os
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

import pandas as pd
import seaborn as sns

from schema import common_mice, common_img, hheise_placecell
from util import helper

from preprint.Filippo.utilities import plot_quantiles_with_data, remove_unwanted_mice, df_corr_result, get_correlation_vector, \
    avg_over_columns, divide_pre_early_late, avg_over_columns_nanmean

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
        data_arrays = [statistic.loc[group].values.flatten() for group in mouse_groups_coarse]

        axs[i] = plot_quantiles_with_data(axs[i], categories_str_list,
                                          data_arrays, titlestr=titlestr)

    return fig


def counts_from_uniques(cell, values):
    return np.array([np.sum(cell == v) for v in values])


def map_days_pre_early_late(day):
    if day <= 0:
        return 'pre'
    elif 0 < day < 7:
        return 'early'
    else:
        return 'late'


def map_mouse_behaviour_group_coarse(mouse):
    if mouse in control:
        return 'sham'
    elif mouse in stroke:
        return 'stroke'


def map_mouse_behaviour_group_fine(mouse):
    if mouse in sham:
        return 'sham'
    elif mouse in recovery:
        return 'recovery'
    elif mouse in no_recovery:
        return 'no recovery'


def plot_grouped_boxplots(df, suptitle, baseline):
    # Set the figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey='row')
    fig.suptitle(suptitle)

    # Columns to plot
    data_columns = [
        'fraction_nc_nc_above_div_totalcount',
        'fraction_nc_pc_above_div_totalcount',
        'fraction_pc_pc_above_div_totalcount'
    ]
    y_labels = ['nc-nc', 'nc-pc', 'pc-pc']  # Short labels for the y-axis

    # Periods for the columns in the grid
    periods = ['pre', 'early', 'late']

    # Iterate over the DataFrame rows (data_columns) and columns (periods) in the plot grid
    for i, column in enumerate(data_columns):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            # Filter data for the current period
            period_data = df[df['period'] == period]
            ax.hlines([baseline], 0, 3, linestyle='dashed', lw=1, color='gray')
            # Plot the boxplot with seaborn
            sns.boxplot(x='mouse_group_fine', y=column, data=period_data, ax=ax, color='#D3D3D3',
                        width=0.5)  # Lighter grey and narrower boxes
            sns.stripplot(x='mouse_group_fine', y=column, data=period_data, ax=ax, color='red', jitter=True, size=4,
                          alpha=0.7)

            # Set titles and labels
            if i == 0:
                ax.set_title(period)
            if j == 0:
                ax.set_ylabel(y_labels[i])
            else:
                ax.set_ylabel('')

            # Remove x-axis labels
            ax.set_xlabel('')

    plt.tight_layout()
    return fig, axes

def get_cell_pair_mask_ids(cormat_trace_df):
    """ Create a DF with same structure as traces_correlation_vectors (one 2D np.array per session) with mask_ids for
    each cell pair (in two axes), to keep track of cell identities of paired cells. """
    # Make DF with same shape as original
    mask_id_df = cormat_trace_df.copy()
    mask_id_df.loc[:] = np.nan

    # Process each session
    for mouse_id in mask_id_df.index:
        # Get surgery date for the current mouse to compute dates from relative days
        curr_surg_date = (common_mice.Surgery & f'mouse_id={mouse_id}' & 'username="hheise"' &
                          'surgery_type="Microsphere injection"').fetch1('surgery_date').date()
        for rel_day in mask_id_df.columns:
            if not np.all(np.isnan(cormat_trace_df.loc[mouse_id, rel_day])):
                # We found a session with data
                n_cells = cormat_trace_df.loc[mouse_id, rel_day].shape[0]
                abs_date = curr_surg_date + timedelta(days=rel_day)

                # Make sure that one session exists
                assert len(common_img.Segmentation & f'mouse_id={mouse_id}' & f'day="{abs_date}"') == 1
                mask_ids = (common_img.Segmentation.ROI & f'mouse_id={mouse_id}' & f'day="{abs_date}"' & 'accepted=1').fetch('mask_id')

                # Put mask_ids into a cross-correlation matrix and get lower triangle (like in Filippos utilities.py)
                result = np.empty((n_cells, n_cells), dtype=object)
                for k in range(n_cells):
                    for j in range(n_cells):
                        result[k, j] = (mask_ids[k], mask_ids[j])
                mask_id_pairs = result[np.tril_indices_from(result, k=-1)]

                # Enter flattened array into DF
                mask_id_df.loc[mouse_id, rel_day] = np.vstack(mask_id_pairs)
    return mask_id_df


def check_place_fields(corr_masks, mask_vecs, pc_vecs, control=False):

    unique_pfs = corr_masks.copy()
    unique_pfs.loc[:] = np.nan
    pairwise_pfs = unique_pfs.copy()

    for mouse_id in corr_masks.index:
        # Get surgery date for the current mouse to compute dates from relative days
        curr_surg_date = (common_mice.Surgery & f'mouse_id={mouse_id}' & 'username="hheise"' &
                          'surgery_type="Microsphere injection"').fetch1('surgery_date').date()
        for rel_day in corr_masks.columns:
            if not np.all(np.isnan(corr_masks.loc[mouse_id, rel_day])):
                abs_date = curr_surg_date + timedelta(days=rel_day)
                curr_corr = corr_masks.loc[mouse_id, rel_day]
                curr_ids = mask_vecs.loc[mouse_id, rel_day]
                curr_pcs = pc_vecs.loc[mouse_id, rel_day]

                # Get mask IDs of highly correlating place cell pairs
                pc_ids = curr_ids[(curr_pcs == 2) & curr_corr]

                if len(pc_ids) == 0:
                    continue

                db_mask_ids = (hheise_placecell.PlaceCell.PlaceField & 'username="hheise"' & f'mouse_id={mouse_id}' & f'day="{abs_date}"'
                               & 'corridor_type=0' & 'large_enough=1' & 'strong_enough=1' & 'transients=1').fetch('mask_id')

                # Sanity check that all pc_ids also have place fields
                assert np.all(np.isin(np.unique(pc_ids), db_mask_ids))

                if control:
                    # Adaptation: get all PCs that are not part of pc-pc pairs (as control)
                    control_ids = np.unique(db_mask_ids)[~np.isin(np.unique(db_mask_ids), np.unique(pc_ids))]
                    if len(control_ids) == 0:
                        continue
                    pf_coms = (hheise_placecell.PlaceCell.PlaceField & 'username="hheise"' & f'mouse_id={mouse_id}' &
                               f'day="{abs_date}"' & 'corridor_type=0' & 'large_enough=1' & 'strong_enough=1' &
                               'transients=1' & f'mask_id in {helper.in_query(control_ids)}').fetch('com')

                    # Make pairs out of control IDs
                    pc_ids = np.array([pair for pair in itertools.combinations(control_ids, 2)])

                else:
                    # Get center of mass of all place cells part of a highly-correlating PC pair
                    pf_coms = (hheise_placecell.PlaceCell.PlaceField & 'username="hheise"' & f'mouse_id={mouse_id}' &
                               f'day="{abs_date}"' & 'corridor_type=0' & 'large_enough=1' & 'strong_enough=1' &
                               'transients=1' & f'mask_id in {helper.in_query(np.unique(pc_ids))}').fetch('com')
                unique_pfs.loc[mouse_id, rel_day] = pf_coms

                # Get center of mass of all PC pairs, associated together in 2D array (like mask_vecs). If a PC has more
                # than one place field, take the one with the smallest standard deviation of its CoM
                pf_com_pairs = np.zeros(pc_ids.shape)
                for m_id in np.unique(pc_ids):
                    curr_com = (hheise_placecell.PlaceCell.PlaceField & 'username="hheise"' & f'mouse_id={mouse_id}' &
                                f'day="{abs_date}"' & 'corridor_type=0' & 'large_enough=1' & 'strong_enough=1' &
                                'transients=1' & f'mask_id={m_id}').fetch('com', order_by='com_sd', limit=1)[0]
                    pf_com_pairs[pc_ids == m_id] = curr_com
                pairwise_pfs.loc[mouse_id, rel_day] = pf_com_pairs

    return unique_pfs, pairwise_pfs


def time_code(n_iter):
    times = np.ones(n_iter)
    for i in range(n_iter):
        start = perf_counter()
        test = np.array([list(tup) for tup in mask_id_pairs])
        # test2 = np.vstack(mask_id_pairs)
        times[i] = perf_counter() - start
    print(f"Mean execution time: {np.mean(times):.4f} +/- {np.std(times):.4f} s")


class Arguments:
    # for debugging
    def __init__(self, dset, quantile):
        self.dataset = dset
        self.quantile = quantile


if __name__ == '__main__':

    # os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    # scriptname = os.path.basename(__file__)[:-3]
    #
    # # command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', type=str, default='dff', help='Use Delta F/F or deconvolved activity',
    #                     choices=['dff', 'decon', 'sa'])
    # parser.add_argument('-q', '--quantile', type=float, default='0.8',
    #                     help='Neuron activity correlations threshold (quantile) of all correlations for a session')
    # args = parser.parse_args()

    args = Arguments('dff', 0.8) #for debugging

    # savestr_base = f'code/08012024/all-cells/outputs/{scriptname}'
    # if not os.path.isdir(savestr_base):
    #     os.mkdir(savestr_base)
    #
    # savestr = f'{savestr_base}/{args.dataset}'
    # if not os.path.isdir(savestr):
    #     os.mkdir(savestr)
    #
    # plt.style.use('code/08012024/plotstyle.mplstyle')

    # dataset selection:
    if args.dataset == 'dff':
        traces_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-dff.pkl'
    elif args.dataset == 'decon':
        traces_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-decon.pkl'
    elif args.dataset == 'sa':
        traces_path = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\correlation_matrices/correlation-mat-unsorted-sa.pkl'
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

    # Create a DF with same structure as traces_correlation_vectors (1 flattened  np.array per session) with tuples of
    # mask_ids for each cell pair, to keep track of cell identities of paired cells
    mask_id_vectors = get_cell_pair_mask_ids(cormat_trace_df=filtered_corrmat_traces_df)

    # remap the cell pair categories to place cell and non-place cell
    remapped_final_pc_vec = pc_classes_matrix.applymap(get_correlation_vector, na_action='ignore')
    # compute dataframe of vectors for correlation statistic for every pair category:
    unique_categories = get_unique_cell_pair_categories(
        remapped_final_pc_vec)  # the ordering of these unique values applies to all contents of the
    cellpair_type_counts = remapped_final_pc_vec.applymap(
        lambda cell: np.array([sum(cell == i) for i in unique_categories]), na_action='ignore')

    string_pair_mapping = {0: 'non-coding-non-coding', 1: 'non-coding-place-cell', 2: 'place-cell-place-cell'}


    # derivative of the apply_function_to_cells function

    # calculate quantiles of correlation vectors (equivalently of correlation matrices)
    def get_quantile_means_of_distribution_pc_counts(quant, tcv, pc_vec, qfunction=lambda x, y: x > y):
        correlation_vec_quantiles = tcv.applymap(lambda cell: np.quantile(cell, quant),
                                                 na_action='ignore')
        corr_greater_than_quantile = apply_function_to_cells(tcv, correlation_vec_quantiles,
                                                             qfunction, ignore_nan=True)

        """
        vec_quant_95 = correlation_vec_quantiles.loc[95, -11]
        corr_greater_95 = corr_greater_than_quantile.loc[95, -11]
        mask_95 = mask_id_vec.loc[95, -11]
        pc_vec_95 = pc_vec.loc[95, -11]
        """

        # calculate fraction of cells that place cells and greater than the 0.8 quantile
        cellpairs_greater_than_quantile = apply_function_to_cells(corr_greater_than_quantile, pc_vec,
                                                                  lambda x, y: y[x], ignore_nan=True)

        cellcats_greater_quantile_counts = cellpairs_greater_than_quantile.applymap(
            lambda x: counts_from_uniques(x, unique_categories), na_action='ignore')
        cellcats_greater_quantile_fractions = cellcats_greater_quantile_counts.applymap(lambda x: x / x.sum(),
                                                                                                  na_action='ignore')

        # split fractions according to pre, early late
        cellcats_quantile_frac_pre_early_late_division = divide_pre_early_late(cellcats_greater_quantile_fractions)

        # calculate mean fractions by pair category over pre, early and late poststroke
        cellcats_quantile_frac_pre_early_late_meanlist = [
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[0], last3=True),
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[1]),
            avg_over_columns(cellcats_quantile_frac_pre_early_late_division[2])]  # results should sum to 1

        return cellcats_quantile_frac_pre_early_late_meanlist, cellcats_greater_quantile_fractions, cellcats_greater_quantile_counts, corr_greater_than_quantile


    quantile = args.quantile
    cellcats_quantile_frac_pre_early_late_meanlist_greater, cellcats_greater_than_quantile_fractions, cellcats_greater_than_quantile_counts, corr_greater_quant = get_quantile_means_of_distribution_pc_counts(
        quant=quantile, tcv=traces_correlation_vectors, pc_vec=remapped_final_pc_vec)

    ############################################################
    ### CHECK PLACE FIELDS OF HIGHLY-CORRELATING PLACE CELLS ###
    ############################################################

    unique_fields, pairwise_fields = check_place_fields(corr_masks=corr_greater_quant, mask_vecs=mask_id_vectors, pc_vecs=remapped_final_pc_vec, control=True)
    unique_fields.to_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_unique_fields_control.pkl')
    pairwise_fields.to_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_pairwise_fields_control.pkl')

    #######################################################################################################################################################
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
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-greater-{args.dataset}.svg',
            dpi=300)

    # make the same plot for the whole distribution to check if the distribution of place cells above the 80th percentile is different than below the 80th
    # percentile

    cellcats_quantile_frac_pre_early_late_meanlist_smaller, cellcats_smaller_than_quantile_fractions, cellcats_smaller_than_quantile_counts = get_quantile_means_of_distribution_pc_counts(
        quantile, traces_correlation_vectors,
        remapped_final_pc_vec, lambda x, y: x < y)

    # make plots for coarse and for fine division
    for mouse_grouping, mouse_group_list, division_name in zip(
            [['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                           [no_recovery, recovery, sham]],
            ['coarse', 'fine']):
        # coarse division
        fig_coarse, axs_coarse = plt.subplots(length, 3, figsize=(16, 18), sharex=True, sharey=True)
        for i in range(length):
            fig_coarse = make_boxplots_pre_early_late(fig_coarse, axs_coarse[i], [arr.apply(lambda x: x[i]) for arr in
                                                                                  cellcats_quantile_frac_pre_early_late_meanlist_smaller],
                                                      mouse_group_list, categories_str_list=mouse_grouping)

            axs_coarse[i, 0].set_ylabel(ylabel_list[i])

        axs_coarse[0, 0].set_title('Prestroke')
        axs_coarse[0, 1].set_title('Early ps')
        axs_coarse[0, 2].set_title('Late ps')

        fig_coarse.tight_layout(rect=(0, 0, 1, 0.95))
        fig_coarse.suptitle(f'Fractions of cell pairs below the {quantile} quantile, by pair category ({args.dataset})')
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-lower-{args.dataset}.png',
            dpi=300)
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-lower-{args.dataset}.svg',
            dpi=300)

        # compute fraction of means above and below the 0.8th quantile
    frac = apply_function_to_cells(cellcats_greater_than_quantile_fractions, cellcats_smaller_than_quantile_fractions,
                                   lambda x, y: x / y)
    div_frac = divide_pre_early_late(frac)
    pre_early_late_frac_means = [avg_over_columns_nanmean(div_frac[0], last3=True),
                                 avg_over_columns_nanmean(div_frac[1]), avg_over_columns_nanmean(div_frac[2])]

    for mouse_grouping, mouse_group_list, division_name in zip(
            [['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                           [no_recovery, recovery, sham]],
            ['coarse', 'fine']):

        fig_coarse, axs_coarse = plt.subplots(length, 3, figsize=(16, 18), sharex=True, sharey=True)
        for i in range(length):
            fig_coarse = make_boxplots_pre_early_late(fig_coarse, axs_coarse[i],
                                                      [arr.apply(lambda x: x[i]) for arr in pre_early_late_frac_means],
                                                      mouse_group_list, categories_str_list=mouse_grouping)

            axs_coarse[i, 0].set_ylabel(ylabel_list[i])

        axs_coarse[0, 0].set_title('Prestroke')
        axs_coarse[0, 1].set_title('Early ps')
        axs_coarse[0, 2].set_title('Late ps')

        fig_coarse.tight_layout(rect=(0, 0, 1, 0.95))
        fig_coarse.suptitle(
            f'Mean fractions of cell pairs (above divided by below) the {quantile} quantile, by pair category ({args.dataset}).')
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-lower-{args.dataset}.png',
            dpi=300)
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-lower-{args.dataset}.svg',
            dpi=300)

    # fractions of counts above quantile divided by total count by cell pair type
    above_fractions_div_totalcount = apply_function_to_cells(cellcats_greater_than_quantile_counts,
                                                             cellpair_type_counts, lambda x, y: x / y, ignore_nan=True)
    div_above_fractions_div_totalcount = divide_pre_early_late(above_fractions_div_totalcount)
    # old, wrong cell affected by nans of arrays in cells: pre_early_late_div_above_fractions_div_totalcount = [avg_over_columns(div_above_fractions_div_totalcount[0], last3=True), avg_over_columns(div_above_fractions_div_totalcount[1]), avg_over_columns(div_above_fractions_div_totalcount[2])]
    pre_early_late_div_above_fractions_div_totalcount = [
        avg_over_columns_nanmean(div_above_fractions_div_totalcount[0], last3=True),
        avg_over_columns_nanmean(div_above_fractions_div_totalcount[1]),
        avg_over_columns_nanmean(div_above_fractions_div_totalcount[2])]

    for mouse_grouping, mouse_group_list, division_name in zip(
            [['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                           [no_recovery, recovery, sham]],
            ['coarse', 'fine']):

        fig_coarse, axs_coarse = plt.subplots(length, 3, figsize=(16, 18), sharex=True, sharey=True)
        for i in range(length):
            fig_coarse = make_boxplots_pre_early_late(fig_coarse, axs_coarse[i], [arr.apply(lambda x: x[i]) for arr in
                                                                                  pre_early_late_div_above_fractions_div_totalcount],
                                                      mouse_group_list, categories_str_list=mouse_grouping)

            axs_coarse[i, 0].set_ylabel(ylabel_list[i])

        axs_coarse[0, 0].set_title('Prestroke')
        axs_coarse[0, 1].set_title('Early ps')
        axs_coarse[0, 2].set_title('Late ps')

        fig_coarse.tight_layout(rect=(0, 0, 1, 0.95))
        fig_coarse.suptitle(
            f'Mean fractions of cell pairs (above count divided by total count) the {quantile} quantile, by pair category ({args.dataset}).')
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}.png',
            dpi=300)
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}.svg',
            dpi=300)

    # export data to csv
    datalist = [cellcats_quantile_frac_pre_early_late_meanlist_greater,
                cellcats_quantile_frac_pre_early_late_meanlist_smaller, pre_early_late_frac_means,
                pre_early_late_div_above_fractions_div_totalcount]
    datalist_renamed = []
    for l, name in zip(datalist, ['greater', 'smaller', 'greater divided by smaller',
                                  'above count divided by total count by cell type']):
        for stat, period in zip(l, ['pre', 'early', 'late']):
            pairtype_stats = [stat.apply(lambda cell: cell[i]).rename((period, f'{name} quantile {quantile}', pairtype))
                              for i, pairtype in string_pair_mapping.items()]
            datalist_renamed.append(pairtype_stats)

    datalist_flat = [x for quantile_stat in datalist_renamed for x in quantile_stat]
    data_df = pd.concat(datalist_flat, axis=1)

    data_df.to_csv(f'{savestr}/cell-pair-fractions-in-correlation-distribution-quantile-{quantile}-{args.dataset}.csv')

    # for the statistics:
    ncnc_above_div_totalcount = above_fractions_div_totalcount.applymap(lambda cell: cell[0], na_action='ignore')
    ncpc_above_div_totalcount = above_fractions_div_totalcount.applymap(lambda cell: cell[1], na_action='ignore')
    pcpc_above_div_totalcount = above_fractions_div_totalcount.applymap(lambda cell: cell[2], na_action='ignore')

    melted_ncnc = ncnc_above_div_totalcount.reset_index().rename(columns={'index': 'mouse'}).melt(id_vars='mouse',
                                                                                                  var_name='day',
                                                                                                  value_name=f'fraction_nc_nc_above_div_totalcount')
    melted_ncpc = ncpc_above_div_totalcount.reset_index().rename(columns={'index': 'mouse'}).melt(id_vars='mouse',
                                                                                                  var_name='day',
                                                                                                  value_name=f'fraction_nc_pc_above_div_totalcount')
    melted_pcpc = pcpc_above_div_totalcount.reset_index().rename(columns={'index': 'mouse'}).melt(id_vars='mouse',
                                                                                                  var_name='day',
                                                                                                  value_name=f'fraction_pc_pc_above_div_totalcount')

    fractions_merged_df = pd.merge(pd.merge(melted_ncnc, melted_ncpc), melted_pcpc)
    fractions_merged_df['period'] = fractions_merged_df['day'].apply(map_days_pre_early_late)
    fractions_merged_df['mouse_group_coarse'] = fractions_merged_df['mouse'].apply(map_mouse_behaviour_group_coarse)
    fractions_merged_df['mouse_group_fine'] = fractions_merged_df['mouse'].apply(map_mouse_behaviour_group_fine)
    fractions_merged_df.to_csv(
        f'{savestr}/correlation-by-cellpairs-ncnc-ncpc-pcpc-cellpairs-above-{quantile}-quantile-div-totalcount-{args.dataset}.csv')

    above_div_total_cellpair_list = [ncnc_above_div_totalcount, ncpc_above_div_totalcount, pcpc_above_div_totalcount]

    # no mean
    for mouse_grouping, mouse_group_list, division_name in zip(
            [['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                           [no_recovery, recovery, sham]],
            ['coarse', 'fine']):
        # coarse division
        fig_coarse, axs_coarse = plt.subplots(length, 3, figsize=(16, 18), sharex=True, sharey=True)
        for i in range(length):
            fig_coarse = make_boxplots_pre_early_late(fig_coarse, axs_coarse[i],
                                                      divide_pre_early_late(above_div_total_cellpair_list[i]),
                                                      mouse_group_list, categories_str_list=mouse_grouping)

            axs_coarse[i, 0].set_ylabel(ylabel_list[i])

        axs_coarse[0, 0].set_title('Prestroke')
        axs_coarse[0, 1].set_title('Early ps')
        axs_coarse[0, 2].set_title('Late ps')

        fig_coarse.tight_layout(rect=(0, 0, 1, 0.95))
        fig_coarse.suptitle(
            f'Fraction of cells above {args.quantile} quantile divided by total count, no mean ({args.dataset})')
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}-nomean.png',
            dpi=300)
        fig_coarse.savefig(
            f'{savestr}/{division_name}-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}-nomean.svg',
            dpi=300)

    fig_nomean, axs_nomean = plot_grouped_boxplots(fractions_merged_df,
                                                   f'Fraction of cells above {args.quantile} quantile, no mean ({args.dataset})',
                                                   1 - args.quantile)
    fig_nomean.savefig(
        f'{savestr}/fine-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}-nomean-new.png')
    fig_nomean.savefig(
        f'{savestr}/fine-allcats-{quantile}-correlation-cell-distribution-upper-div-total-{args.dataset}-nomean-new.svg')
