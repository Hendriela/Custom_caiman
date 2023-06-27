#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 23/06/2023 15:39
@author: hheise

"""
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats as stat

from schema import hheise_behav, hheise_hist
from util import helper

sns.set_context('talk')

no_deficit = [93, 91, 95]   # Exclude 94, never learned the task
no_deficit_flicker = [111, 114, 116]
recovery = [33, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]

# Grouping by number of spheres (extrapolated for whole brain)
low = [38, 91, 111, 115, 122]   # < 50 spheres
mid = [33, 83, 86, 89, 93, 94, 95, 108, 110, 112, 113, 114, 116, 121]  # 50 - 500 spheres
high = [41, 63, 69, 85, 90, 109, 123]   # > 500 spheres

mice = [*no_deficit, *no_deficit_flicker, *recovery, *deficit_no_flicker, *deficit_flicker, *sham_injection]


def compute_behavior_coordinates(behavior_df, n_last_days=3, late_thresh_day=15):

    result = []
    # For each mouse, get avg behavior at two timepoints: until day 3 after surgery, and the last three days.
    # If a mouse has few sessions (less than three sessions after day 15), only the last session is used)
    for mouse in behavior_df.mouse_id.unique():
        # Make sure that data is sorted chronologically
        curr_mouse = behavior_df[behavior_df['mouse_id'] == mouse].sort_values('day')

        # Early timepoint
        early = curr_mouse[(curr_mouse['day'] > 0) & (curr_mouse['day'] <= 3)]['performance'].mean()

        # Late timepoint
        if (curr_mouse['day'] >= late_thresh_day).sum() < n_last_days:
            late = curr_mouse[curr_mouse['day'] >= late_thresh_day]['performance'].mean()
        else:
            late = curr_mouse['performance'].iloc[-n_last_days:].mean()
        result.append(pd.DataFrame([dict(mouse_id=mouse, early=early, late=late)]))

    return pd.concat(result, ignore_index=True)


def plot_scatter(data, ax, title=None, legend=True):
    sns.scatterplot(data=data, x='early', y='late', hue='spheres', palette='flare', hue_norm=LogNorm(), s=100, ax=ax,
                    legend=legend)
    if legend:
        ax.legend(title='Spheres', fontsize='10', title_fontsize='12')
    # Label each point with mouse_id
    for i, point in enumerate(ax.collections):
        # Extract the x and y coordinates of the data point
        x = point.get_offsets()[:, 1]
        y = point.get_offsets()[:, 0]

        # Add labels to the data point
        for j, y_ in enumerate(y):
            ax.text(y[j], x[j] - 0.05, data[data['early'] == y_]['mouse_id'].iloc[0],
                    ha='center', va='bottom', fontsize=12)

    if title is not None:
        ax.set_title(title)


def plot_pca(model, ax, reduced_data=None, mean_offset: Union[np.ndarray, int] = 0):

    def reset_ax_lim(curr_lim, points):
        lower_lim = np.min(points[points < curr_lim[1]]) if np.sum(points < curr_lim[0]) > 0 else curr_lim[0]
        upper_lim = np.max(points[points > curr_lim[1]]) if np.sum(points > curr_lim[1]) > 0 else curr_lim[1]
        return lower_lim, upper_lim

    # Draw arrow in direction of 1st principal component
    # The arrow's direction comes from the PCA vector "components_", and the length is determined by the explained variance
    lims = []
    for length, vector in zip(model.explained_variance_, model.components_):
        v = vector * 3 * np.sqrt(length)        # End coordinates of arrow vector
        origin = model.mean_ + mean_offset      # Origin coordinates of vector (shifted by mean of dataset)
        head = model.mean_ + v + mean_offset    # End coordinates of arrow (shifted by mean of dataset)

        arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
        ax.annotate('', head, origin, arrowprops=arrowprops)

        lims.append(origin)
        lims.append(head)
    lims = np.vstack(lims)

    if reduced_data is not None:
        reduced_data = reduced_data + mean_offset
        ax.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color='g', alpha=0.6)

    ax.set_xlim(reset_ax_lim(curr_lim=ax.get_xlim(), points=lims[:, 0]))
    ax.set_ylim(reset_ax_lim(curr_lim=ax.get_ylim(), points=lims[:, 1]))


# Get normalized performance
metrics = ['binned_lick_ratio', 'si_binned_run', 'distance', 'autocorr']
plot_pca_mapping = False

fig, axes = plt.subplots(nrows=2, ncols=len(metrics), layout='constrained')

datas = []
for j, norm in enumerate([True, False]):    # Plot both normalized and raw performance metrics
    data = {}
    for i, metric in enumerate(metrics):

        out = (hheise_behav.VRPerformance &
               f'mouse_id in {helper.in_query(mice)}').get_normalized_performance(attr=metric, pre_days=5, baseline_days=3,
                                                                                  plotting=False, normalize=norm)

        # Add sphere numbers for marker hue
        sphere_count = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & f'mouse_id in {helper.in_query(mice)}'
                                     & 'metric_name="spheres"').fetch('count_extrap', 'mouse_id', as_dict=True))
        sphere_count = sphere_count.rename(columns={'count_extrap': 'spheres'})

        # Drop a few outlier sessions (usually last session of a mouse that should not be used)
        out = out.drop(out[(out['mouse_id'] == 83) & (out['day'] == 27)].index)
        out = out.drop(out[(out['mouse_id'] == 69) & (out['day'] == 23)].index)

        matrix = compute_behavior_coordinates(out, n_last_days=2, late_thresh_day=15)
        matrix = matrix.merge(sphere_count)
        data[metric] = matrix

        if i == len(metrics)-1:
            legend = True
        else:
            legend = False

        # Perform PCA to find a single dimension, and correlate the position of each mouse on it to its sphere count
        pca = PCA(n_components=1)
        pca_data = matrix[['early', 'late']]    # Extract the 2D data used for PCA
        data_mean = pca_data.mean(axis=0)  # Center data before PCA (subtract mean to get mean of 0)
        data_centered = pca_data - data_mean
        reduced_metric = pca.fit_transform(data_centered)

        # plt.figure()
        # plt.scatter(x=data_centered['early'], y=data_centered['late'])
        # # plt.scatter(x=reduced_metric[:, 0], y=reduced_metric[:, 1])
        # inverse_reduced = pca1.inverse_transform(reduced_metric1)
        # plt.scatter(x=inverse_reduced[:, 0], y=inverse_reduced[:, 1])
        #
        # plt.figure()
        # plt.scatter(x=pca_data['early'], y=pca_data['late'])
        # inverse_reduced = pca1.inverse_transform(reduced_metric1)
        # inverse_raw = inverse_reduced + np.array(data_mean)
        # plt.scatter(x=inverse_raw[:, 0], y=inverse_raw[:, 1])
        #
        # plt.scatter(x=pca_data['early'], y=pca_data['late'], label='raw')
        # plt.scatter(x=pca_data['early'], y=pca_data['late'], label='raw')

        # If primary component is negative (arrow points down-left), invert to make consistent
        if np.all(pca.components_ < 0):
            reduced_metric = -reduced_metric
            pca.components_ = -pca.components_

        sphere_corr = np.corrcoef(reduced_metric.squeeze(), matrix['spheres'])[0, 1]

        # Display stats of PCA in textbox
        props = dict(boxstyle='round', alpha=0)
        text = f'%var: {pca.explained_variance_ratio_[0]:.2f}\nr = {sphere_corr:.2f}'
        axes[j, i].text(0.95, 0.05, text, transform=axes[j, i].transAxes, fontsize=14, verticalalignment='bottom',
                        horizontalalignment='right', bbox=props, fontfamily='monospace')

        # Plot raw, not centered, data
        plot_scatter(data=matrix, ax=axes[j, i], legend=legend, title=metric)
        if plot_pca_mapping:
            plot_pca(model=pca, ax=axes[j, i], reduced_data=pca.inverse_transform(reduced_metric),
                     mean_offset=np.array(data_mean))

    datas.append(data)


### Plot linear regressions

def annotate(x_label, y_label, **kws):

    # Extract data
    X = kws['data'][x_label]
    Y = kws['data'][y_label]

    # Compute correlation
    r, p = stat.pearsonr(X, Y)

    # Compute goodness of fit for linear regression
    x = np.array(X).reshape(-1, 1)
    y = Y
    linreg = LinearRegression().fit(x, y)
    r_sq = linreg.score(x, y)
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}\n$r^2$={:.2f}'.format(r, p, r_sq),
            transform=ax.transAxes)

dfs = []
for data in datas:
    df = pd.concat(data)
    df['metric'] = [key for key in data.keys() for _ in range(len(data[key]))]
    df.reset_index(drop=True, inplace=True)
    dfs.append(df)

dfs[0]['normalized'] = True
dfs[1]['normalized'] = False
df = pd.concat(dfs, ignore_index=True)

g = sns.lmplot(df, x='early', y='late', col='metric', row='normalized', facet_kws=dict(sharex=False, sharey=False))
g.map_dataframe(annotate, 'early', 'late')

# for metric_name, arr in data.items():
#     x = np.array(arr['early']).reshape(-1, 1)
#     y = arr['late']
#     linreg = LinearRegression().fit(x, y)
#     r_sq = linreg.score(x, y)
