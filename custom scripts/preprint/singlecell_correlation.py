#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/08/2023 10:49
@author: hheise

"""

import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

from schema import hheise_placecell, common_match
from preprint import data_cleaning as dc
from util import helper

matplotlib.rcParams['font.sans-serif'] = "Arial"  # Use same font as Prism
plt.rcParams['svg.fonttype'] = 'none'

mouse_ids = [33, 41,  # Batch 3
             63, 69,  # Batch 5
             83, 85, 86, 89, 90, 91, 93, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 121, 122]  # Batch 8

folder = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint'


def spatial_map_correlations_single_cells(spatial_maps=list, align_days=False, save_files=False):
    """
    Correlate spatial maps across sessions of multiple numpy arrays. This function does not require the matching
    and merging of arrays and is more flexible with variable acquisition days.

    Args:
        spatial_maps: Data list, output from get_matched_data().

    Returns:
        A Dataframe with all the cross-session correlations.
    """

    DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)

    df_list = []
    for animal in spatial_maps:
        for net_id, data in animal.items():

            rel_days = np.array(data[1])

            if align_days:
                if 3 not in rel_days:
                    rel_days[(rel_days == 2) | (rel_days == 4)] = 3
                rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
                rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
                rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
                rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
                rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
                rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
                rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
                if 28 not in rel_days:
                    rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27
                rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
                rel_days[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1))+1, 1)

            # Do not analyse day 1
            day1_mask = rel_days != 1
            rel_days = rel_days[day1_mask]
            arr = data[0][:, day1_mask]

            curr_df = pd.DataFrame({'net_id': [net_id]*len(arr)})

            # Loop through days and compute correlation between sessions that are 3 days apart
            for day_idx, day in enumerate(rel_days):
                next_day_idx = np.where(rel_days == day+DAY_DIFF)[0]

                # If a session 3 days later exists, compute the correlation of all cells between these sessions
                # Do not analyze session 1 day after stroke (unreliable data)
                if day+DAY_DIFF != 1 and len(next_day_idx) == 1:
                    curr_corr = [np.corrcoef(arr[cell_id, day_idx], arr[cell_id, next_day_idx[0]])[0, 1]
                                 for cell_id in range(len(arr))]

                    curr_df[rel_days[next_day_idx[0]]] = curr_corr
            df_list.append(curr_df)

    final_df = pd.concat(df_list, ignore_index=True)

    # Sort columns numerically. The column names show the 2nd day of correlation, e.g. the column 0 shows the correlation
    # of activity on day 0 (the day of the surgery) with day -3 (3 days before surgery)
    net_ids = final_df.pop('net_id')
    sorted_days = np.sort(final_df.columns.astype(int))
    final_df = final_df[sorted_days]
    final_df['net_id'] = net_ids

    ### ANALYSIS ###
    # Get average cross-correlation of of pre, early post and late post sessions

    # Prestroke sessions (the index has to be the column names, not a boolean mask)
    pre_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[sorted_days <= 0]]), axis=1))

    # Pre-Post
    pre_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days) & (sorted_days <= DAY_DIFF)]]),
                                      axis=1))

    # Early Post
    early_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days) & (sorted_days <= 7)]]),
                                        axis=1))

    # Late Post
    late_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(7 < sorted_days)]]), axis=1))

    # All Post
    all_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days)]]), axis=1))

    # Construct DataFrame
    avg_df = pd.DataFrame({'pre': pre_avg, 'pre_post': pre_post_avg, 'early_post': early_post_avg,
                           'late_post': late_post_avg, 'all_post': all_post_avg, 'mouse_id': final_df['net_id']})
    avg_df = avg_df.set_index('mouse_id')

    avg_df_rel = avg_df.div(avg_df['pre'], axis=0)
    avg_df_dif = avg_df.sub(avg_df['pre'], axis=0)

    # Drop cells that are not in every period
    avg_df_clean = avg_df.dropna(axis=0)
    # avg_df_clean.to_csv(os.path.join(folder, r'single_cell_corr_sham.csv'))

    # Sort cells by pre-stroke correlation
    # avg_df_clean_sort = avg_df_clean.sort_values(by='pre', ascending=False)
    # avg_df_clean_sort.to_csv(os.path.join(folder, r'single_cell_corr_sorted_allstroke.csv'))
    #
    # avg_df_dif.to_csv(os.path.join(folder, r'single_cell_corr_dif_allstroke.csv'))

    # Reshape DataFrame for Prism (old, not needed anymore)
    # avg_dif_merge = None
    # for mouse_id in avg_df_dif['mouse_id'].unique():
    #     curr_df = avg_df_dif[avg_df_dif.mouse_id == mouse_id][['pre_post', 'early_post', 'late_post', 'all_post']]
    #     curr_df = curr_df.rename(columns={'pre_post': f'pre_post_{mouse_id.split("_")[0]}',
    #                                       'early_post': f'early_post_{mouse_id.split("_")[0]}',
    #                                       'late_post': f'late_post_{mouse_id.split("_")[0]}',
    #                                       'all_post': f'all_post_{mouse_id.split("_")[0]}'})
    #     if avg_dif_merge is None:
    #         avg_dif_merge = curr_df
    #     else:
    #         avg_dif_merge = avg_dif_merge.join(curr_df.reset_index(drop=True), how='outer')


    # Store data sorted by mice
    if save_files:
        for net in final_df['net_id'].unique():
            out = avg_df_dif.loc[final_df['net_id'] == net]
            out.to_csv(os.path.join(folder, f'single_cell_corr_dif_{net}.csv'))

    # # Store total correlation pair data sorted by mice (for scatter plot)
    # avg_df_clean['net_id'] = final_df['net_id']
    # avg_df_totalpost = avg_df_clean.pivot(index='pre', columns='net_id', values='all_post')

    # avg_df_totalpost.to_csv(os.path.join(folder, f'single_cell_corr_totalpost_allmice.csv'))

    return avg_df_clean, avg_df_dif


def correlate_stability(df, plot=False, figname=None):

    df = df.reset_index()
    df['mouse_id'] = df.apply(lambda x: int(x['mouse_id'].split('_')[0]), axis=1)

    # For each mouse, correlate single-neuron cross-session stability across phases and fit linear regression
    results = []
    plot_df = []
    for i, (mouse_id, mouse_data) in enumerate(df.groupby('mouse_id')):
        for j, phase_pair in enumerate((('pre', 'early_post'), ('pre', 'late_post'), ('early_post', 'late_post'))):

            x = mouse_data[phase_pair[0]]
            y = mouse_data[phase_pair[1]]
            corr = stats.pearsonr(x, y)                 # Compute correlation
            slope, intercept = np.polyfit(x, y, 1)      # Linear regression

            results.append(pd.DataFrame([dict(mouse_id=mouse_id, phase_pair=f"{phase_pair[0]} - {phase_pair[1]}",
                                              r=corr.statistic, p=corr.pvalue, slope=slope)]))

            if plot:
                plot_df.append(pd.DataFrame(dict(mouse_id=mouse_id, x=x, y=y, phase_pair=f"{phase_pair[0]} - {phase_pair[1]}")))

    results = pd.concat(results, ignore_index=True)

    if plot:
        plot_df = pd.concat(plot_df, ignore_index=True)

        g = sns.lmplot(data=plot_df, x='x', y='y', col='phase_pair', row='mouse_id', facet_kws=dict(margin_titles=True),
                       height=5, aspect=1)

        for row_idx, row in enumerate(g.row_names):
            for col_idx, col in enumerate(g.col_names):
                curr_ax = g.axes[row_idx, col_idx]

                curr_results = results[(results.mouse_id == row) & (results.phase_pair == col)].iloc[0].to_dict()

                curr_ax.text(0.05, 0.99, f'r={curr_results["r"]:.3f}\n'
                                         f'p={curr_results["p"]:.3f}\n'
                                         f'slope={curr_results["slope"]:.3f}', transform=curr_ax.transAxes,
                             verticalalignment='top', horizontalalignment='left', fontsize=12)

                curr_ax.axvline(0, linestyle='--', c='grey', alpha=0.5)
                curr_ax.axhline(0, linestyle='--', c='grey', alpha=0.5)

        if figname is not None:
            plt.savefig(os.path.join(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\single_cell_corr',
                                     figname))

    return results



#%% Call function
spat_maps = dc.load_data('spat_dff_maps')

df_clean, df_dif = spatial_map_correlations_single_cells(spatial_maps=spat_maps, align_days=False)
df_clean, df_dif = spatial_map_correlations_single_cells(spatial_maps=spat_maps, align_days=True)

result = correlate_stability(df=df_clean, plot=True, figname="cross_session_correlation_pair.png")

result.pivot(index='phase_pair', columns='mouse_id', values='slope').loc[['pre - early_post', 'pre - late_post', 'early_post - late_post']].to_clipboard()
