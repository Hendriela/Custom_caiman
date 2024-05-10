#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 10/05/2024 17:38
@author: hheise

"""

# script to reproduce figure 1 from Yue Kris Wu's 2020 PNAS paper Homeostatic mechanisms regulate distinct aspects ofcortical circuit dynamics
# based on neural-data-funccon-neurons-alldays-correlation-scatters.py, but with as many datapoints as possible!
# NOTE: the data in this file excludes mice 121, 63 (too few contiguous cell pairs) and 112 (no calcium measurements).
# also, the prestroke averages are computed over all available prestroke sessions, not over the last 3 prestroke sessions! to achieve this,
# look at the definition of avg_over_columns in utilities.py

import pickle
import numpy as np

import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import scipy as sp
import seaborn as sns
import sys

# matplotlib.use('Agg')
sys.path.append('../')
from preprint.Filippo.utilities import df_corr_result, ax_plot_coor_fit, get_group_results_slope_corr, plot_quantiles_with_data, \
    avg_over_columns, divide_pre_early_late, avg_over_columns_series
import argparse
from statannotations.Annotator import Annotator

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

categories_coarse = ['control', 'stroke']
categories_fine = ['no recovery', 'recovery', 'sham']

mouse_coarse_mapping = {}
mouse_fine_mapping = {}

for mouse in control + stroke:
    # coarse mapping
    if mouse in control:
        mouse_coarse_mapping[mouse] = 'control'
    else:
        mouse_coarse_mapping[mouse] = 'stroke'

    # fine mapping
    if mouse in sham:
        mouse_fine_mapping[mouse] = 'sham'
    elif mouse in recovery:
        mouse_fine_mapping[mouse] = 'recovery'
    elif mouse in no_recovery:
        mouse_fine_mapping[mouse] = 'no recovery'


# take the lower (or upper, does not matter, just be consistent!) of the unclustered correlation matrices, excluding the diagonal elements, and flatten them
def get_correlation_vector(cell):
    return cell[np.tril_indices_from(cell,
                                     k=-1)]  # distance vector, i.e. flattened lower triangle of symmetric correlation matrix, excluding diagonal elements


#####################################
def remove_unwanted_mice(dict, mice):
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy


def get_correlation_matrix_for_columns(df):
    session_cmats = {}
    for col, sess in df.items():
        session_cmats[col] = np.corrcoef(np.vstack(sess))
    return pd.Series(session_cmats)


def calc_corrmat_maxneurons_pre_early_late(dff):
    sessions = ['pre', 'early', 'late']

    correlation_pair_df = {}

    for m, df in dff.items():
        divided_pre_early_late = divide_pre_early_late(df)

        common_neuron_id_list = []
        # pre-early, pre-late, early-late
        for (p1, p2), (p1n, p2n) in zip(itertools.combinations(divided_pre_early_late, 2),
                                        itertools.combinations(sessions, 2)):
            common_neuron_id_list.append(np.intersect1d(p1.dropna().index, p2.dropna().index))

        # now have all indices of corresponding periods. need to filter the dataframes and calculate
        # the correlation matrices for every column accordingly
        period_corrmat_pairs = {}
        for idx, (p1, p2), (p1n, p2n) in zip(common_neuron_id_list, itertools.combinations(divided_pre_early_late, 2),
                                             itertools.combinations(sessions, 2)):
            cmat_p1 = get_correlation_matrix_for_columns(p1.loc[idx])
            cmat_p2 = get_correlation_matrix_for_columns(p2.loc[idx])

            period_corrmat_pairs[(p1n, p2n)] = (cmat_p1, cmat_p2)
            period_corrmat_pairs[f'{p1n}-{p2n}-common-ids'] = idx

        correlation_pair_df[m] = period_corrmat_pairs

    return correlation_pair_df


def get_avg_correlation_vector_by_period(correlation_matrix_pair_result):
    sessions = ['pre', 'early', 'late']
    mean_correlationvecs = {}

    for m, period_corrmat_pairs in correlation_matrix_pair_result.items():

        mouse_mean_correlationvecs = {}
        for (p1n, p2n) in itertools.combinations(sessions, 2):
            p1, p2 = period_corrmat_pairs[(p1n, p2n)]
            last3_1 = False
            last3_2 = False
            if p1n == 'pre':
                last3_1 = True
            if p2n == 'pre':
                last3_1 = True

            p1v, p2v = p1.apply(get_correlation_vector), p2.apply(get_correlation_vector)  # get the offdiagonals!
            p1vm = avg_over_columns_series(p1v, last3=last3_1)
            p2vm = avg_over_columns_series(p2v, last3=last3_2)

            mouse_mean_correlationvecs[(p1n, p2n)] = (p1vm, p2vm)  # pair of average correlation offdiagonals!
        mean_correlationvecs[m] = mouse_mean_correlationvecs

    return mean_correlationvecs


def ax_plot_coor_fit_with_vectors(axes, mousecount, mouse, row, xvec, yvec, xrange=np.array([-0.2, 0.8]), xy=None):
    # function to plot scatters and make both linear fit and correlation value of the point cloud
    x = xrange
    if xy == None:
        xy = (-0.1, 0.6)
    axes[mousecount][row].scatter(xvec, yvec, s=1, color='r')
    axes[mousecount][row].set_title(f'Mouse {mouse}')
    axes[mousecount][row].plot(x, x, color='k')
    axes[mousecount][row].set_aspect('equal')
    fit = sp.stats.linregress(xvec, yvec)
    axes[mousecount][row].plot(x, x * fit.slope + fit.intercept, color='green')
    try:
        coef = sp.stats.pearsonr(xvec, yvec)
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\nr = {coef.statistic:.2f}\np = {coef.pvalue:.4f}',
                                       xy=xy, fontsize=10)
        return fit, coef
    except:
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\ntoo few data points', xy=xy, fontsize=10)
        return fit, np.nan


def ax_plot_histo_with_vectors(axes, mousecount, mouse, row, xvec, yvec, bins=100):
    try:
        histdata = 1 / np.sqrt(2) * (yvec - xvec)  # from rotation matrix of pi/2
    except:
        histdata = [np.nan, np.nan]
    sns.histplot(histdata, bins=bins, color='grey',
                 stat='probability', ax=axes[mousecount][row],
                 element='step', fill=True)
    axes[mousecount][row].set_title(f'Mouse {mouse}')
    skewness = sp.stats.skew(histdata)
    std = np.std(histdata)

    # x = np.quantile(hist[1], 0.2)
    # y = np.quantile(hist[0], 0.9)
    # axes[mousecount][row].annotate(f'skewness = {skewness:.2f}\nstd = {std:.2f}', xy = (x, y), fontsize = 10)
    return skewness, std


def get_group_results_histo_skew_std(group, pre_early_late_skew_std):
    # outputs 3 numpy arrays, where the row corresponds to the mouse id as set in 'group' and the columns correspond to different pairings,
    # i.e. pre-early, pre-late and early-late

    res_skew = np.array([(pre_early_late_skew_std[m]['skew_pre_early'], pre_early_late_skew_std[m]['skew_pre_late'],
                          pre_early_late_skew_std[m]['skew_early_late']) for m in group])

    res_std = []
    for m in group:
        try:
            res_std.append((pre_early_late_skew_std[m]['std_pre_early'], pre_early_late_skew_std[m]['std_pre_late'],
                            pre_early_late_skew_std[m]['std_early_late']))
        except:
            res_std.append((np.nan, np.nan, np.nan))
    res_std = np.array(res_std)

    return res_skew, res_std


def mask_non_significant(res_df):
    """
    Masks entries in DataFrame columns with NaN where corresponding '_pval' columns have values > 0.05.

    Args:
    res_df (pd.DataFrame): The DataFrame with measurement and '_pval' columns.

    Returns:
    pd.DataFrame: The DataFrame with non-significant measurements masked as NaN.
    """
    fit_columns = res_df.columns[res_df.columns.str.contains("fit")]
    fitslopes = res_df[fit_columns]
    corr_columns = res_df.columns[res_df.columns.str.contains("corr")]
    corrs = res_df[corr_columns]
    mask_pvals = res_df.loc[:, res_df.columns.str.contains("pval")].applymap(lambda cell: cell > 0.05)

    fitslopes.mask(mask_pvals.rename(columns={c: newc for c, newc in zip(mask_pvals.columns, fitslopes.columns)}),
                   inplace=True)
    corrs.mask(mask_pvals.rename(columns={c: newc for c, newc in zip(mask_pvals.columns, corrs.columns)}), inplace=True)

    res_df.loc[:, fit_columns] = fitslopes
    res_df.loc[:, corr_columns] = corrs

    return res_df


def plot_boxplots_with_significance(df, group_col, value_col, ax=None):
    # Check if an axis is provided, if not, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plots with specific component colors
    boxplot = sns.boxplot(x=group_col, y=value_col, data=df, ax=ax, color='lightgrey', fliersize=0, width=0.3)

    # Set individual color components
    for i, artist in enumerate(ax.artists):
        # Set the facecolor of the boxes
        artist.set_facecolor('lightgrey')
        # Each artist has 6 associated Line2D objects (whiskers, caps, and median line)
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color('black')
            line.set_mfc('black')
            line.set_mec('black')

    # Add scatter plot for individual data points
    sns.stripplot(x=group_col, y=value_col, data=df, color='red', jitter=0.2, size=5, edgecolor='gray', linewidth=0.5,
                  ax=ax)

    # Initialize the Annotator with pairs of groups you want to test
    pairs = [(group1, group2) for i, group1 in enumerate(df[group_col].unique())
             for group2 in df[group_col].unique()[i + 1:]]
    annotator = Annotator(ax, pairs, data=df, x=group_col, y=value_col)

    # Add annotations using a specific test, like a T-test
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()

    # Set titles and labels within the function or return axis to modify outside
    ax.set_title(f'{value_col} by {group_col} with Statistical Significance')
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)

    return ax  # Return the axis for further modification


def generate_boxplot_grid(res_df, data_prefix, ylabel):
    """
    Generates a grid of box plots with scatter plots overlayed, filtered by data_prefix,
    with one row for 'mouse_group_coarse' and another for 'mouse_group_fine'.
    Each axis will have a specific title from a provided list, no x-labels, and no y-labels.

    Args:
    res_df (pd.DataFrame): DataFrame containing the data.
    data_prefix (str): Prefix of the columns to filter ('corr' or 'fit').

    Returns:
    None: Displays the matplotlib plot.
    """
    # Filter columns that start with the specified prefix and do not end in '_pval'
    measurement_columns = [col for col in res_df.columns if col.startswith(data_prefix) and not col.endswith('_pval')]
    titles = ['Early poststroke vs prestroke', 'Late poststroke vs prestroke', 'Late poststroke vs early poststroke']

    # Setup the figure and axes grid
    n_cols = len(measurement_columns)
    fig, axs = plt.subplots(nrows=2, ncols=n_cols, figsize=(18, 12), constrained_layout=True)

    pairs_coarse = [('control', 'stroke')]
    pairs_coarse = [('sham', 'no recovery'), ('sham', 'recovery'), ('recovery', 'no recovery')]

    # Loop through each measurement column and plot the boxplots for each group category
    for idx, col in enumerate(measurement_columns):
        for i, group_col in enumerate(['mouse_group_coarse', 'mouse_group_fine']):
            ax = axs[i, idx]
            # Create the box plot with a lighter grey color and narrower width
            ax = plot_boxplots_with_significance(res_df, group_col=group_col, value_col=col, ax=ax)

            # Set specific title from the list, cycling through the titles if necessary
            ax.set_title(titles[idx % len(titles)])
            ax.set_xlabel('')  # No x-label
            ax.set_ylabel('')  # No y-label

    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    return fig, axs


class Arguments:
    # for debugging
    def __init__(self, dset):
        self.dataset = dset


if __name__ == '__main__':

    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dff', help='Use Delta F/F or deconvolved activity',
                        choices=['dff', 'decon', 'sa'])
    args = parser.parse_args()

    # args = Arguments('dff') #for debugging

    savestr_base = f'code/08012024/tracked-cells/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)

    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/08012024/plotstyle.mplstyle')

    if args.dataset == 'dff':
        traces_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\dff_tracked_normal.pkl'
    elif args.dataset == 'decon':
        traces_path = 'data/neural-data/neural-data-clean/decon_tracked_normal.pkl'
    elif args.dataset == 'sa':
        traces_path = 'data/neural-data/spatial_activity_maps_dff.pkl'
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')

    pc_classes_binary_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\is_pc.pkl'
    coords_path_binary = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\cell_coords.pkl'

    unwanted_mice = [121, 112]

    with open(traces_path, 'rb') as file:
        traces = pickle.load(file)
    with open(pc_classes_binary_path, 'rb') as file:
        pc_classes_binary = pickle.load(file)
    with open(coords_path_binary, 'rb') as file:
        coords_binary = pickle.load(file)

    traces = remove_unwanted_mice(traces, unwanted_mice)
    pc_classes_binary = remove_unwanted_mice(pc_classes_binary, unwanted_mice)
    coords_binary = remove_unwanted_mice(coords_binary, unwanted_mice)

    correlation_matrices_by_period_pair = calc_corrmat_maxneurons_pre_early_late(traces)
    mean_correlation_vector_by_period_pair = get_avg_correlation_vector_by_period(correlation_matrices_by_period_pair)

    # make a grid of scatterplots, 3 for every mouse
    nmice = len(mean_correlation_vector_by_period_pair.keys())  # rows
    nints = 3  # number of columns, 1 plot each: (pre-early, pre-late, early-late)
    sessions = ['pre', 'early', 'late']
    session_combos = itertools.combinations(sessions, 2)

    fig_scatters, axes = plt.subplots(nmice, nints, figsize=(3.5 * nints, 3 * nmice), sharex=True, sharey=True,
                                      dpi=300)
    pre_early_late_fit_corrcoef = {}

    for mousecount, mouse in enumerate(mean_correlation_vector_by_period_pair.keys()):
        pre_early_late_fit_corrcoef[mouse] = {}
        for i, ((p1n, p2n), (p1vm, p2vm)) in enumerate(mean_correlation_vector_by_period_pair[mouse].items()):
            fit, corrcoef = ax_plot_coor_fit_with_vectors(axes, mousecount, mouse, i, p1vm, p2vm,
                                                          xrange=np.array([np.min(p1vm), np.max(p1vm)]))
            pre_early_late_fit_corrcoef[mouse][f'corr_{p1n}_{p2n}'] = corrcoef
            pre_early_late_fit_corrcoef[mouse][f'fit_{p1n}_{p2n}'] = fit

        axes[mousecount][0].set_ylabel('Early post corr')
        axes[mousecount][1].set_ylabel('Late post corr')
        axes[mousecount][2].set_ylabel('Late post corr')
        axes[mousecount][0].set_xlabel('Pre corr')
        axes[mousecount][1].set_xlabel('Pre corr')
        axes[mousecount][2].set_xlabel('Early corr')

    fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig_scatters.suptitle(f'Correlations of neuron pairs, prestroke, early and late poststroke ({args.dataset})')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-scatter-{args.dataset}.png')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-scatter-{args.dataset}.svg')

    control_slope, control_corr, control_pval = get_group_results_slope_corr(control, pre_early_late_fit_corrcoef)
    stroke_slope, stroke_corr, stroke_pval = get_group_results_slope_corr(stroke, pre_early_late_fit_corrcoef)

    no_recovery_slope, no_recovery_corr, no_recovery_pval = get_group_results_slope_corr(no_recovery,
                                                                                         pre_early_late_fit_corrcoef)
    recovery_slope, recovery_corr, recovery_pval = get_group_results_slope_corr(recovery, pre_early_late_fit_corrcoef)
    sham_slope, sham_corr, sham_pval = get_group_results_slope_corr(sham, pre_early_late_fit_corrcoef)

    # export fit results and correlations to csv file!
    resdict = {}
    for mouse, res in pre_early_late_fit_corrcoef.items():
        stats_pval_dict = {statname: stat_obj[0] for statname, stat_obj in res.items()}
        pvals = {statname + '_pval': getattr(res[statname], 'pvalue') for statname in
                 filter(lambda name: "corr" in name, res.keys())}
        stats_pval_dict = stats_pval_dict | pvals
        resdict[mouse] = stats_pval_dict

    res_df = pd.DataFrame(resdict).T
    res_df['mouse_group_coarse'] = [mouse_coarse_mapping[m] for m in res_df.index]
    res_df['mouse_group_fine'] = [mouse_fine_mapping[m] for m in res_df.index]
    res_df = mask_non_significant(res_df)
    # res_df.reset_index().rename()
    res_df.to_csv(f'{savestr}/fit-slopes-correlations-pointclouds-{args.dataset}.csv', index_label='mouse')

    # correlations
    f_corr, axs_corr = generate_boxplot_grid(res_df, 'corr', 'Pearson r')
    # f_corr.tight_layout(rect = (0.05, 0, 1, 0.985))
    f_corr.suptitle(f'Boxplots of pointcloud correlations ({args.dataset})')
    f_corr.savefig(f'{savestr}/corr-coarse-fine-boxplots-{args.dataset}.png')
    f_corr.savefig(f'{savestr}/corr-coarse-fine-boxplots-{args.dataset}.svg')

    # slopes
    f_slope, axs_slope = generate_boxplot_grid(res_df, 'fit', 'Slope')
    # f_slope.tight_layout(rect = (0.05, 0, 1, 0.985))
    f_slope.suptitle(f'Boxplots of pointcloud fit slopes ({args.dataset})')
    f_slope.savefig(f'{savestr}/slopes-coarse-fine-boxplots-{args.dataset}.png')
    f_slope.savefig(f'{savestr}/slopes-coarse-fine-boxplots-{args.dataset}.svg')

    #######################################################################################
    # skewness and standard deviation analysis

    fig_scatters, axes = plt.subplots(nmice, nints, figsize=(3.5 * nints, 3 * nmice), sharex=True, sharey=False,
                                      dpi=300)
    pre_early_late_skew_std = {}

    for mousecount, mouse in enumerate(mean_correlation_vector_by_period_pair.keys()):
        pre_early_late_skew_std[mouse] = {}
        for i, ((p1n, p2n), (p1vm, p2vm)) in enumerate(mean_correlation_vector_by_period_pair[mouse].items()):
            skew, std = ax_plot_histo_with_vectors(axes, mousecount, mouse, i, p1vm, p2vm, bins=200)
            pre_early_late_skew_std[mouse][f'skew_{p1n}_{p2n}'] = skew
            pre_early_late_skew_std[mouse][f'std_{p1n}_{p2n}'] = std

            axes[mousecount][i].set_xlabel('Pearson r')

    fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig_scatters.suptitle(f'Correlations of neuron pairs, prestroke, early and late poststroke ({args.dataset})')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-correlation-histograms-{args.dataset}.png')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-correlation-histograms-{args.dataset}.svg')

    control_skew, control_std = get_group_results_histo_skew_std(control, pre_early_late_skew_std)
    stroke_skew, stroke_std = get_group_results_histo_skew_std(stroke, pre_early_late_skew_std)

    no_recovery_skew, no_recovery_std = get_group_results_histo_skew_std(no_recovery, pre_early_late_skew_std)
    recovery_skew, recovery_std = get_group_results_histo_skew_std(recovery, pre_early_late_skew_std)
    sham_skew, sham_std = get_group_results_histo_skew_std(sham, pre_early_late_skew_std)

    # Coarse division
    categories_coarse = ['control', 'stroke']
    categories_fine = ['no recovery', 'recovery', 'sham']

    f_skews, axs = plt.subplots(2, 3, figsize=(18, 14), dpi=300)

    for i, titlestr in enumerate(['Early poststroke vs prestroke',
                                  'Late poststroke vs prestroke',
                                  'Late poststroke vs early poststroke']):
        data_arrays_skew_coarse = [control_skew[:, i], np.array(list(filter(lambda x: x == x, stroke_skew[:, i])))]
        axs[0, i] = plot_quantiles_with_data(axs[0, i], categories_coarse, data_arrays_skew_coarse, titlestr=titlestr)

        data_arrays_skew_fine = [np.array(list(filter(lambda x: x == x, no_recovery_skew[:, i]))), recovery_skew[:, i],
                                 sham_skew[:, i]]
        axs[1, i] = plot_quantiles_with_data(axs[1, i], categories_fine, data_arrays_skew_fine, titlestr=titlestr)

    axs[0, 0].set_ylabel('Histogram skewness')
    axs[1, 0].set_ylabel('Histogram skewness')
    f_skews.tight_layout(rect=(0, 0, 1, 0.95))
    f_skews.suptitle(f'Boxplots for projected histogram skewness of cell pairs ({args.dataset})')
    f_skews.savefig(f'{savestr}/histo-skewness-coarse-fine-boxplots-{args.dataset}.png')
    f_skews.savefig(f'{savestr}/histo-skewness-coarse-fine-boxplots-{args.dataset}.svg')
    f_skews.show()

    # correlations
    f_std, axs = plt.subplots(2, 3, figsize=(18, 14), dpi=300)

    for i, titlestr in enumerate(['Early poststroke vs prestroke',
                                  'Late poststroke vs prestroke',
                                  'Late poststroke vs early poststroke']):
        data_arrays_std_coarse = [control_std[:, i], np.array(list(filter(lambda x: x == x, stroke_std[:, i])))]
        axs[0, i] = plot_quantiles_with_data(axs[0, i], categories_coarse, data_arrays_std_coarse, titlestr=titlestr)

        data_arrays_std_fine = [np.array(list(filter(lambda x: x == x, no_recovery_std[:, i]))), recovery_std[:, i],
                                sham_std[:, i]]
        axs[1, i] = plot_quantiles_with_data(axs[1, i], categories_fine, data_arrays_std_fine, titlestr=titlestr)
    axs[0, 0].set_ylabel('Histogram std')
    axs[1, 0].set_ylabel('Histogram std')
    f_std.tight_layout(rect=(0, 0, 1, 0.95))
    f_std.suptitle(f'Boxplots for projected histogram std of cell pairs ({args.dataset})')
    f_std.savefig(f'{savestr}/histo-std-coarse-fine-boxplots-{args.dataset}.png')
    f_std.savefig(f'{savestr}/histo-std-coarse-fine-boxplots-{args.dataset}.svg')
    f_std.show()

    res_df = pd.DataFrame(pre_early_late_skew_std).T
    res_df['mouse_group_coarse'] = [mouse_coarse_mapping[m] for m in res_df.index]
    res_df['mouse_group_fine'] = [mouse_fine_mapping[m] for m in res_df.index]
    res_df = mask_non_significant(res_df)
    res_df.to_csv(f'{savestr}/pointcloud-histograms-skew-std-{args.dataset}.csv', index_label='mouse')

    ###########################
    ###### For the figure:
    toplot_mice = {mouse: mean_correlation_vector_by_period_pair[mouse] for mouse in [91, 113, 69]}
    # scatter:
    fig_scatters, axes = plt.subplots(3, nints, figsize=(3.5 * nints, 3 * 3), sharex=True, sharey=True,
                                      dpi=300)
    pre_early_late_fit_corrcoef_figure = {}

    for mousecount, mouse in enumerate(toplot_mice.keys()):
        pre_early_late_fit_corrcoef_figure[mouse] = {}
        for i, ((p1n, p2n), (p1vm, p2vm)) in enumerate(toplot_mice[mouse].items()):
            fit, corrcoef = ax_plot_coor_fit_with_vectors(axes, mousecount, mouse, i, p1vm, p2vm,
                                                          xrange=np.array([np.min(p1vm), np.max(p1vm)]), xy=(-0.1, 0.5))
            pre_early_late_fit_corrcoef_figure[mouse][f'corr_{p1n}_{p2n}'] = corrcoef
            pre_early_late_fit_corrcoef_figure[mouse][f'fit_{p1n}_{p2n}'] = fit
            axes[mousecount][i].locator_params(axis='both', nbins=2)
            # axes[mousecount][i].set_title(f'Mouse {mouse}', fontsize = 12)
        axes[mousecount][0].set_ylabel('Early post corr')
        axes[mousecount][1].set_ylabel('Late post corr')
        axes[mousecount][2].set_ylabel('Late post corr')
        axes[mousecount][0].set_xlabel('Pre corr')
        axes[mousecount][1].set_xlabel('Pre corr')
        axes[mousecount][2].set_xlabel('Early corr')

    fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.92))
    fig_scatters.suptitle(f'Correlations of neuron pairs, prestroke, early and late poststroke (dff)')
    fig_scatters.savefig(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\neuron_pair_pointclouds\grid-pre-early-late-scatter-dff-paper-figure_dots.png')
    fig_scatters.savefig(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\neuron_pair_pointclouds\grid-pre-early-late-scatter-dff-paper-figure_dots.svg')

    # histograms:
    fig_scatters, axes = plt.subplots(3, nints, figsize=(3.5 * nints, 3 * 3), sharex=True, sharey=True,
                                      dpi=300)
    pre_early_late_skew_std = {}
    for mousecount, mouse in enumerate(toplot_mice.keys()):
        pre_early_late_skew_std[mouse] = {}
        for i, ((p1n, p2n), (p1vm, p2vm)) in enumerate(toplot_mice[mouse].items()):
            skew, std = ax_plot_histo_with_vectors(axes, mousecount, mouse, i, p1vm, p2vm, bins=200)
            pre_early_late_skew_std[mouse][f'skew_{p1n}_{p2n}'] = skew
            pre_early_late_skew_std[mouse][f'std_{p1n}_{p2n}'] = std

            axes[mousecount][i].set_xlabel('Pearson r')
    axes[0, 0].set_title('Pre, mouse 91')
    axes[0, 1].set_title('Early, mouse 91')
    axes[0, 2].set_title('Late, mouse 91')
    fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig_scatters.suptitle(f'Correlations of neuron pairs, prestroke, early and late poststroke ({args.dataset})')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-correlation-histograms-{args.dataset}-paper-figure.png')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-correlation-histograms-{args.dataset}-paper-figure.svg')

