#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 31/10/2023 17:15
@author: hheise

"""

# script to reproduce figure 1 from Yue Kris Wu's 2020 PNAS paper Homeostatic mechanisms regulate distinct aspects ofcortical circuit dynamics
import pickle
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

import pandas as pd
from sklearn.cluster import KMeans

os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
scriptname = os.path.basename(__file__)[:-3]

savestr = f'code/18102023/outputs/{scriptname}'
if not os.path.isdir(savestr):
    os.mkdir(savestr)

plt.style.use('code/18102023/plotstyle.mplstyle')

dff_path = 'data/neural-data/dff.pkl'

with open(dff_path, 'rb') as file:
    dff = pickle.load(file)


def filter_neurons_always_present(dict):
    # return a dict, such as decon or dff, that contains for every mouse
    # only neurons that were imaged on all sessions across time.

    filtered = {k: v.dropna() for k, v in dict.items()}
    return filtered


dff = filter_neurons_always_present(dff)


def get_rastergram_coords(df, session_id, thr):
    points_list = []
    for i in df[session_id].index:
        if df.loc[i, session_id] is not np.NaN:
            thresholded = df.loc[i, session_id] >= thr
            threshold_idx = np.where(thresholded)[0]
            for idx in threshold_idx:
                points_list.append((i, idx))

    return np.array(points_list)  # first column: neuron id; second column: time id


def spike_rastergram(df, session_id, thr, **kwargs):
    imaging_frequency = 30  # Hz
    scatter_array = get_rastergram_coords(df, session_id, thr)
    f = plt.figure(figsize=(12, 6))
    plt.scatter(scatter_array[:, 1] / imaging_frequency, scatter_array[:, 0], marker='+', s=1, color='black')
    plt.xlabel('Time [sec]')
    plt.ylabel('Neuron id')
    return f


def correlation_cluster(session, metric='euclidean', method='average'):
    # calculate correlation of neural activity for one day (session). return the sorted labels and distance matrix
    correlation_matrix = np.corrcoef(np.vstack(session))
    distances = sp.spatial.distance.pdist(correlation_matrix,
                                          metric=metric)  # compute the distance matrix of correlation vectors
    linkage_matrix = sp.cluster.hierarchy.linkage(distances, method=method)
    dendrogram = sp.cluster.hierarchy.dendrogram(linkage_matrix, no_plot=True)
    leaves = dendrogram['leaves']  # clustered indices of mouse id

    distance_matrix = sp.spatial.distance.squareform(distances)
    clustered_correlation_matrix = correlation_matrix[leaves, :][:, leaves]

    return correlation_matrix, clustered_correlation_matrix, distance_matrix, leaves


def neuraldata_cluster_corr(dset):
    # function that iterates over all mice in a dataset (either deconvolved or dff or other dicts with int:dataframe)
    # for every dataframe, compute the correlation of all arrays stored in every row of a single column. this returns
    # a correlation matrix for a single column, a clustered version of the correlation matrix, a distance matrix, as
    # well as the ordering

    # the output is a dict, where every key is a mouse, and every value is another dict, with the session as key.
    # this inner dict has as values another dict, with following keys:
    # corr: correlation matrix
    # clustcorr: clustered correlation matrix
    # dist: distance matrix, where column ordering corresponds to ordering of corr
    # order: ordering according to which corr is reordered into clustcorr

    mice = {}

    for k in dset.keys():
        sessions = {}
        for sess in dff[k].columns:
            results_dict = {}
            res = correlation_cluster(dset[k].loc[:, sess].dropna())

            results_dict['sesssion'] = sess
            results_dict['corr'] = res[0]
            results_dict['clustcorr'] = res[1]
            results_dict['dist'] = res[2]
            results_dict['order'] = res[3]

            sessions[sess] = results_dict

        mice[k] = sessions

    return mice


dff_corr_clust = neuraldata_cluster_corr(dff)


# want a function that takes results from neuraldata_cluster_corr and includes a

# correlation_matrices
def df_corr_result(res, key='corr'):
    to_df_dict = {k: pd.Series([res[k][sess][key] for sess in res[k].keys()], index=res[k].keys()) for k in res.keys()}
    df = pd.DataFrame(data=to_df_dict).T
    return df


dff_corr_df = df_corr_result(dff_corr_clust)
dff_corr_clust_df = df_corr_result(dff_corr_clust, key='clustcorr')

# this is the most up-to-date division of mice into groups (24.10.2023)
# 'Stroke' (deficit, Recovery + No Recovery groups): [33, 41, 63, 69, 85, 86, 89, 90, 110, 113, 121]
# 'Control' (no deficit, Sham + No Deficit groups): [83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]
#
# Fine:
# 'Recovery' (performance below "early", but not "late" threshold): [85, 86, 89, 90, 113],
# 'No Recovery' (performance below "early" and "late" threshold): [41, 63, 69, 110, 121],
# 'No Deficit' (No deficit, but n_spheres > detection threshold): [33, 83, 93, 95, 108, 112, 114, 116] ,
# 'Sham' (No deficit, and n_spheres < detection threshold): [91, 111, 115, 122]
# Und die Mäuse, die das Physical Exercise ab Tag 6 bekommen haben: [83, 85, 89, 90, 91]

control = [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]
control_clean = [33, 83, 91, 93, 95, 108, 111, 114, 115, 116, 122]
stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113, 121]

no_deficit = [33, 83, 93, 95, 108, 112, 114, 116]
no_deficit_clean = [33, 83, 93, 95, 108, 114, 116]
no_recovery = [41, 63, 69, 110, 121]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122]


#############################################
# reproduce figure 5 from https://www.pnas.org/cgi/doi/10.1073/pnas.1906595116, i.e. divide data into pre and poststroke, and
# see if there is a substantial change in correlation
# the control mice are mice  [91, 111, 115, 122] (email on October 14, 2023)
# exclude mouse 121!
# exclusion of mice can occur at the end of preprocessing
# use only DFF signal, not spike probability

# early and late post-stroke: early < d7, late > d7
# try this for all mice except for control mice
# try this separately for control mice

##Preprocessing
# calculate correlation matrices for every mouse, for every session
# (see calculation of dff_corr_df)

# take the lower (or upper, does not matter, just be consistent!) of the unclustered correlation matrices, excluding the diagonal elements, and flatten them
def correlation_vector(cell):
    return cell[np.tril_indices_from(cell,
                                     k=-1)]  # distance vector, i.e. flattened lower triangle of symmetric correlation matrix, excluding diagonal elements


def find_closest_columns_to_zero(row, n=3):
    # Find the 3 columns closest to 0
    sorted_columns = sorted(row[row.index <= 0].index)[-n:]
    return sorted_columns


dff_correlation_vectors = dff_corr_df.applymap(correlation_vector, na_action='ignore')


# average over the pre-stroke, early and late post-stroke mice
def avg_over_columns(df):
    meanlist = []
    for mouse in df.index:
        m = df.loc[mouse].dropna().mean()
        meanlist.append(m)
    return pd.Series(meanlist, index=df.index)


def avg_over_columns_prestroke(df):
    meanlist = []
    for mouse in df.index:
        clean_df = df.loc[mouse].dropna()
        last3 = find_closest_columns_to_zero(clean_df)
        meanlist.append(clean_df[last3].mean())
    return pd.Series(meanlist, index=df.index)


# group the distance vectors into pre, early and late post-stroke.
early_correlations = dff_correlation_vectors.loc[:,
                     np.logical_and(dff_correlation_vectors.columns > 0, dff_correlation_vectors.columns < 7)]
late_correlations = dff_correlation_vectors.loc[:, dff_correlation_vectors.columns >= 7]

pre_avg_corr = avg_over_columns_prestroke(
    dff_correlation_vectors)  # compute the average over the last 3 poststroke sessions!
early_avg_corr = avg_over_columns(early_correlations)
late_avg_corr = avg_over_columns(late_correlations)

x = np.array([-0.2, 0.8])


def ax_plot_coor_fit(axes, mousecount, mouse, row, xframe, yframe):
    axes[mousecount][row].scatter(xframe.loc[mouse], yframe.loc[mouse], s=1, marker='x', color='r')
    axes[mousecount][row].set_title(f'Mouse {mouse}')
    axes[mousecount][row].plot(x, x, color='k')
    fit = sp.stats.linregress(xframe.loc[mouse], yframe.loc[mouse])
    axes[mousecount][row].plot(x, x * fit.slope + fit.intercept, color='green')
    try:
        coef = sp.stats.pearsonr(xframe.loc[mouse], yframe.loc[mouse])
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\nr = {coef.statistic:.2f}\np = {coef.pvalue:.4f}',
                                       xy=(-0.1, 0.6), fontsize=10)
        return fit, coef
    except:
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\ntoo few data points', xy=(-0.1, 0.6), fontsize=10)
        return fit, np.nan


# make a grid of scatterplots, 3 for every mouse
nmice = len(pre_avg_corr.index)  # rows
nints = 3  # number of columns, 1 plot each: (pre-early, pre-late, early-late)

fig_scatters, axes = plt.subplots(nmice, nints, figsize=(3.5 * nints, 3 * nmice), sharex=True, sharey=True,
                                  dpi=300)
pre_early_post_fit_corrcoef = {}

for mousecount, mouse in enumerate(pre_avg_corr.index):
    fit_pre_early, corrcoef_pre_early = ax_plot_coor_fit(axes, mousecount, mouse, 0, pre_avg_corr, early_avg_corr)
    pre_early_post_fit_corrcoef[mouse] = {}
    pre_early_post_fit_corrcoef[mouse]['corr_pre_early'] = corrcoef_pre_early
    pre_early_post_fit_corrcoef[mouse]['fit_pre_early'] = fit_pre_early
    axes[mousecount][0].set_ylabel('Early post corr')

    fit_pre_late, corrcoef_pre_late = ax_plot_coor_fit(axes, mousecount, mouse, 1, pre_avg_corr, late_avg_corr)
    pre_early_post_fit_corrcoef[mouse]['corr_pre_late'] = corrcoef_pre_late
    pre_early_post_fit_corrcoef[mouse]['fit_pre_late'] = fit_pre_late
    axes[mousecount][1].set_ylabel('Late post corr')

    fit_early_late, corrcoef_early_late = ax_plot_coor_fit(axes, mousecount, mouse, 2, early_avg_corr, late_avg_corr)
    pre_early_post_fit_corrcoef[mouse]['corr_early_late'] = corrcoef_early_late
    pre_early_post_fit_corrcoef[mouse]['fit_early_late'] = fit_early_late
    axes[mousecount][2].set_ylabel('Late post corr')

axes[mousecount][0].set_xlabel('Pre corr')
axes[mousecount][1].set_xlabel('Pre corr')
axes[mousecount][2].set_xlabel('Early corr')

fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.97))
fig_scatters.suptitle('Correlations of neuron pairs, prestroke, early and late poststroke')
fig_scatters.savefig(f'{savestr}/grid-pre-early-late-scatter.png')


def get_group_results_slope_corr(group):
    # outputs 3 numpy arrays, where the row corresponds to the mouse id as set in 'group' and the columns correspond to different pairings,
    # i.e. pre-early, pre-late and early-late

    res_slope = np.array(
        [(pre_early_post_fit_corrcoef[m]['fit_pre_early'].slope, pre_early_post_fit_corrcoef[m]['fit_pre_late'].slope,
          pre_early_post_fit_corrcoef[m]['fit_early_late'].slope) for m in group])

    res_corr = []
    for m in group:
        try:
            res_corr.append((pre_early_post_fit_corrcoef[m]['corr_pre_early'].statistic,
                             pre_early_post_fit_corrcoef[m]['corr_pre_late'].statistic,
                             pre_early_post_fit_corrcoef[m]['corr_early_late'].statistic))
        except:
            res_corr.append((np.nan, np.nan, np.nan))
    res_corr = np.array(res_corr)

    res_pval = []
    for m in group:
        try:
            res_pval.append((pre_early_post_fit_corrcoef[m]['corr_pre_early'].pvalue,
                             pre_early_post_fit_corrcoef[m]['corr_pre_late'].pvalue,
                             pre_early_post_fit_corrcoef[m]['corr_early_late'].pvalue))
        except:
            res_pval.append((np.nan, np.nan, np.nan))
    res_pval = np.array(res_pval)

    return res_slope, res_corr, res_pval


control_slope, control_corr, control_pval = get_group_results_slope_corr(control_clean)
stroke_slope, stroke_corr, stroke_pval = get_group_results_slope_corr(stroke)

no_deficit_clean_slope, no_deficit_clean_corr, no_deficit_clean_pval = get_group_results_slope_corr(no_deficit_clean)
no_recovery_slope, no_recovery_corr, no_recovery_pval = get_group_results_slope_corr(no_recovery)
recovery_slope, recovery_corr, recovery_pval = get_group_results_slope_corr(recovery)
sham_slope, sham_corr, sham_pval = get_group_results_slope_corr(sham)


def plot_quantiles_with_data(ax, categories, data_arrays, titlestr='', ylabel='', xlabel=''):
    # Define the quantiles you want to plot
    quantiles = [0.25, 0.5, 0.75]

    # Create numerical values for categories
    category_indices = np.arange(len(categories))

    # Store combinations of categories for t-tests
    combinations = [(i, j) for i in category_indices for j in category_indices if i < j]

    # Set box width
    box_width = 0.2

    for category, data in zip(category_indices, data_arrays):
        # Calculate quantiles for the data
        quantile_values = np.quantile(data, quantiles)

        # Plot the quantiles as box plots
        ax.boxplot(data, positions=[category], widths=box_width, labels=[categories[category]],
                   patch_artist=True, boxprops=dict(facecolor='lightgray'),
                   medianprops={'color': 'black'})

        # Plot individual data points as scattered dots in the background
        jittered_x = np.random.normal(category, 0.04, len(data))
        ax.plot(jittered_x, data, 'ro', alpha=0.3)

    # Get the maximum range among all data arrays
    max_range = max([np.max(data) - np.min(data) for data in data_arrays])
    max_value = max([np.max(data) for data in data_arrays])

    # Calculate dynamic line_offset based on the data range
    line_offset = max_range * 0.25  # You can adjust the factor as needed
    # Perform t-tests for all unique combinations of categories
    significance_count = 0
    for i, comb in enumerate(combinations):
        category1, category2 = comb
        data1 = data_arrays[category1]
        data2 = data_arrays[category2]
        _, p_value = sp.stats.ttest_ind(data1, data2)

        # Add significance stars for different combinations
        if p_value < 0.001:
            star = '***'
        elif p_value < 0.01:
            star = '**'
        elif p_value < 0.05:
            star = '*'
        else:
            star = ''
            continue

        # Determine the x-coordinates for the lines
        x1, x2 = category1, category2

        # Calculate the vertical position for the line (above the highest datapoint)
        y = max_value + line_offset * significance_count + 0.1 * line_offset

        # Add lines connecting significantly different combinations
        ax.plot([x1, x2], [y, y], 'k-', lw=0.5)

        # Add significance stars for each combination
        ax.text((x1 + x2) / 2, y + 0.015, star, fontsize=12, horizontalalignment='center')
        significance_count += 1

    # Set x-axis labels to category names
    ax.set_xticks(category_indices)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titlestr)

    return ax


#####################################
# BEWARE!!
# the following plots are not computed with all mice, because some mice don't appear in the
# calcium imaging dataframe or have too few neuron pairs that are measured over all poststroke sessions
# these are mice 63 and 112

#################################
# Coarse division
categories = ['control', 'stroke']

# fit slopes
f_slopes, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, titlestr in zip(range(len(axs)), ['Early poststroke vs prestroke',
                                         'Late poststroke vs prestroke',
                                         'Late poststroke vs early poststroke']):
    data_arrays = [control_slope[:, i], list(filter(lambda x: x == x, stroke_corr[:, i]))]
    axs[i] = plot_quantiles_with_data(axs[i], categories, data_arrays,
                                      titlestr=titlestr)
axs[0].set_ylabel('Fit slopes')
f_slopes.tight_layout(rect=(0, 0, 1, 0.9))
f_slopes.suptitle('Boxplots for fit slopes of correlation pairs')
f_slopes.savefig(f'{savestr}/slopes-coarse-boxplots.png')
f_slopes.show()

# correlations
f_corr, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, titlestr in zip(range(len(axs)), ['Early poststroke vs prestroke',
                                         'Late poststroke vs prestroke',
                                         'Late poststroke vs early poststroke']):
    data_arrays = [control_corr[:, i], list(filter(lambda x: x == x, stroke_corr[:, i]))]
    axs[i] = plot_quantiles_with_data(axs[i], categories, data_arrays,
                                      titlestr=titlestr)
axs[0].set_ylabel('Correlations')
f_corr.tight_layout(rect=(0, 0, 1, 0.9))
f_corr.suptitle('Boxplots for correlations of correlation pairs')
f_corr.savefig(f'{savestr}/correlations-coarse-boxplots.png')
f_corr.show()

# Fine division
categories_fine = ['no deficit', 'no recovery', 'recovery', 'sham']

# fit slopes
f_slopes, axs = plt.subplots(1, 3, figsize=(21, 6))

for i, titlestr in zip(range(len(axs)), ['Early poststroke vs prestroke',
                                         'Late poststroke vs prestroke',
                                         'Late poststroke vs early poststroke']):
    data_arrays = [no_deficit_clean_slope[:, i], list(filter(lambda x: x == x, no_recovery_slope[:, i])),
                   recovery_slope[:, i], sham_slope[:, i]]
    axs[i] = plot_quantiles_with_data(axs[i], categories_fine, data_arrays,
                                      titlestr=titlestr)
axs[0].set_ylabel('Fit slopes')
f_slopes.tight_layout(rect=(0, 0, 1, 0.9))
f_slopes.suptitle('Boxplots for fit slopes of correlation pairs')
f_slopes.savefig(f'{savestr}/slopes-fine-boxplots.png')
f_slopes.show()

# correlations
f_corr, axs = plt.subplots(1, 3, figsize=(21, 6))

for i, titlestr in zip(range(len(axs)), ['Early poststroke vs prestroke',
                                         'Late poststroke vs prestroke',
                                         'Late poststroke vs early poststroke']):
    data_arrays = [no_deficit_clean_corr[:, i], list(filter(lambda x: x == x, no_recovery_corr[:, i])),
                   recovery_corr[:, i], sham_corr[:, i]]
    axs[i] = plot_quantiles_with_data(axs[i], categories_fine, data_arrays,
                                      titlestr=titlestr)
axs[0].set_ylabel('Correlations')
f_corr.tight_layout(rect=(0, 0, 1, 0.9))
f_corr.suptitle('Boxplots for correlations of correlation pairs')
f_corr.savefig(f'{savestr}/corr-fine-boxplots.png')
f_corr.show()

