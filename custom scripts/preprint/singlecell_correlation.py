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
import numpy as np
from schema import hheise_placecell, common_match
from preprint import data_cleaning as dc

matplotlib.rcParams['font.sans-serif'] = "Arial"  # Use same font as Prism
plt.rcParams['svg.fonttype'] = 'none'

mouse_ids = [33, 41,  # Batch 3
             63, 69,  # Batch 5
             83, 85, 86, 89, 90, 91, 93, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 121, 122]  # Batch 8

folder = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint'


def spatial_map_correlations_single_cells(spatial_maps=list):
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
            days = np.array(data[1])
            arr = data[0]

            curr_df = pd.DataFrame({'net_id': [net_id]*len(arr)})

            # Loop through days and compute correlation between sessions that are 3 days apart
            for day_idx, day in enumerate(days):
                next_day_idx = np.where(days == day+DAY_DIFF)[0]

                # If a session 3 days later exists, compute the correlation of all cells between these sessions
                # Do not analyze session 1 day after stroke (unreliable data)
                if day+DAY_DIFF != 1 and len(next_day_idx) == 1:
                    curr_corr = [np.corrcoef(arr[cell_id, day_idx], arr[cell_id, next_day_idx[0]])[0, 1]
                                 for cell_id in range(len(arr))]

                    curr_df[days[next_day_idx[0]]] = curr_corr
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
                           'late_post': late_post_avg, 'all_post': all_post_avg})

    avg_df_rel = avg_df.div(avg_df['pre'], axis=0)
    avg_df_dif = avg_df.sub(avg_df['pre'], axis=0)

    # Drop cells that are not in every period
    avg_df_clean = avg_df.dropna(axis=0)
    # avg_df_clean.to_csv(os.path.join(folder, r'single_cell_corr_sham.csv'))

    # Sort cells by pre-stroke correlation
    avg_df_clean_sort = avg_df_clean.sort_values(by='pre', ascending=False)
    # avg_df_clean_sort.to_csv(os.path.join(folder, r'single_cell_corr_sorted_allstroke.csv'))
    #
    # avg_df_dif.to_csv(os.path.join(folder, r'single_cell_corr_dif_allstroke.csv'))

    # Reshape DataFrame for Prism
    avg_df_dif['mouse_id'] = final_df['net_id']

    avg_dif_merge = None
    for mouse_id in avg_df_dif['mouse_id'].unique():
        curr_df = avg_df_dif[avg_df_dif.mouse_id == mouse_id][['pre_post', 'early_post', 'late_post', 'all_post']]
        curr_df = curr_df.rename(columns={'pre_post': f'pre_post_{mouse_id.split("_")[0]}',
                                          'early_post': f'early_post_{mouse_id.split("_")[0]}',
                                          'late_post': f'late_post_{mouse_id.split("_")[0]}',
                                          'all_post': f'all_post_{mouse_id.split("_")[0]}'})
        if avg_dif_merge is None:
            avg_dif_merge = curr_df
        else:
            avg_dif_merge = avg_dif_merge.join(curr_df.reset_index(drop=True), how='outer')


    # Store data sorted by mice
    for net in final_df['net_id'].unique():
        out = avg_df_dif.loc[final_df['net_id'] == net]
        out.to_csv(os.path.join(folder, f'single_cell_corr_dif_{net}.csv'))

    # Store total correlation pair data sorted by mice (for scatter plot)
    avg_df_clean['net_id'] = final_df['net_id']
    avg_df_totalpost = avg_df_clean.pivot(index='pre', columns='net_id', values='all_post')

    # avg_df_totalpost.to_csv(os.path.join(folder, f'single_cell_corr_totalpost_allmice.csv'))

    return avg_df_clean_sort, avg_df_dif, avg_df_totalpost

# #%% Load data
#
# queries = (
#            (common_match.MatchedIndex & 'mouse_id=33'),       # 407 cells
#            # (common_match.MatchedIndex & 'mouse_id=38'),
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),   # 246 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=63' & 'day<="2021-03-23"'),     # 350 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),     # 350 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=83'),   # 270 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),   # 250 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=86'),   # 86 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=89'),   # 183 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),   # 131 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=91'),   # 299 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=93'),   # 397 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=95'),   # 350 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=108' & 'day<"2022-09-09"'),     # 316 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & 'day<"2022-09-09"'),     # 218 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=111' & 'day<"2022-09-09"'),     # 201 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=113' & 'day<"2022-09-09"'),     # 350 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=114' & 'day<"2022-09-09"'),     # 307 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),     # 331 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=116' & 'day<"2022-09-09"'),     # 350 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'),     # 401 cells
#            (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),     # 791 cells
# )
#
# is_pc = []
# pfs = []
# spatial_maps = []
# match_matrices = []
# spat_dff_maps = []
#
# for i, query in enumerate(queries):
#     if i == 2:
#         match_matrices.append(query.construct_matrix(start_with_ref=True))
#     else:
#         match_matrices.append(query.construct_matrix(start_with_ref=False))
#
#     is_pc.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
#                                         extra_restriction=dict(corridor_type=0, place_cell_id=2),
#                                         return_array=True, relative_dates=True, surgery='Microsphere injection'))
#
#     pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
#                                       extra_restriction=dict(corridor_type=0, place_cell_id=2),
#                                       return_array=False, relative_dates=True, surgery='Microsphere injection'))
#
#     spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
#                                                extra_restriction=dict(corridor_type=0, place_cell_id=2),
#                                                return_array=True, relative_dates=True,
#                                                surgery='Microsphere injection'))
#
#     spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
#                                                 extra_restriction=dict(corridor_type=0, place_cell_id=2),
#                                                 return_array=True, relative_dates=True,
#                                                 surgery='Microsphere injection'))
#
# folder = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
# with open(os.path.join(folder, f'spatial_activity_maps_spikerate.pkl'), 'wb') as file:
#     pickle.dump(spatial_maps, file)
#%% Call function

spatial_maps = dc.load_data('spat_dff_maps')

df_clean, df_dif, df_totalpost = spatial_map_correlations_single_cells(spatial_maps)
