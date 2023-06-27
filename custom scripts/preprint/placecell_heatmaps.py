#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/06/2023 11:33
@author: hheise

"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from schema import hheise_placecell, common_match, hheise_behav

#%% Load matched/fetched data
dir = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
with open(os.path.join(dir, 'match_matrices.pickle'), "r") as output_file:
    match_matrices = pickle.load(output_file)
with open(os.path.join(dir, 'is_pc.pickle'), "r") as output_file:
    is_pc = pickle.load(output_file)
with open(os.path.join(dir, 'pfs.pickle'), "r") as output_file:
    pfs = pickle.load(output_file)
with open(os.path.join(dir, 'spatial_maps.pickle'), "r") as output_file:
    spatial_maps = pickle.load(output_file)
with open(os.path.join(dir, 'spat_dff_maps.pickle'), "r") as output_file:
    spat_dff_maps = pickle.load(output_file)

#%% Make a place-cell-heatmap from deficit mice (41+69), recovery mice (85+90) and sham mice (115+122) --> FIRST ATTEMPT


def get_place_cells(is_pc_arr, spat_dff_arr, pc_day: int, select_days=None, ignore_missing=True):

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_dff_arr = spat_dff_arr[0]

    # Find indices of selected days
    filter_date = [np.where(dates == sel_day)[0][0] for sel_day in select_days]
    is_pc_filt = is_pc_arr[:, filter_date]

    # Only include cells that are found on every selected day
    if ignore_missing:
        filter_missing = np.isnan(is_pc_filt).sum(axis=1) == 0
    else:
        filter_missing = np.ones(len(is_pc_arr), dtype=bool)
    is_pc_filt = is_pc_filt[filter_missing]

    # Only include cells that are place cells on a given day
    filter_pc = is_pc_filt[:, np.where(np.array(select_days) == pc_day)[0][0]] == 1

    # Apply filters to spatial activity maps to only return the dFF of the requested cells
    spat_dff_filt = spat_dff_arr[:, filter_date]
    spat_dff_filt = spat_dff_filt[filter_missing]
    spat_dff_filt = spat_dff_filt[filter_pc]

    return spat_dff_filt


def get_noncoding(is_pc_arr, spat_dff_arr, select_days=None, ignore_missing=True, return_cell_mask=True):
    """ Only take cells that occur in all sessions and are not place cells in prestroke. """

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_dff_arr = spat_dff_arr[0]

    # Find indices of selected days
    filter_date = [np.where(dates == sel_day)[0][0] for sel_day in select_days]
    is_pc_filt = is_pc_arr[:, filter_date]
    spat_dff_filt = spat_dff_arr[:, filter_date]

    # Only include cells that are found on every selected day
    if ignore_missing:
        filter_missing = np.isnan(is_pc_filt).sum(axis=1) == 0
    else:
        filter_missing = np.ones(len(is_pc_arr), dtype=bool)

    # Only include cells that were not place cells in prestroke days (negative relative date)
    noncoding_mask = np.nansum(is_pc_arr[:, :np.argmax(dates[np.where(dates <= 0)])+1], axis=1) == 0

    # Apply filters to spatial activity maps to only return the dFF of the requested cells
    spat_dff_filt = spat_dff_filt[filter_missing & noncoding_mask]

    if return_cell_mask:
        return spat_dff_filt, filter_missing & noncoding_mask
    else:
        return spat_dff_filt


def get_stab(spat_dff_arr, num_sessions=3):
    """ Session-session correlation. """
    corrcoef = []
    for cell in spat_dff_arr:
        corrcoef.append(np.nanmean([np.corrcoef(cell[i], cell[i+1])[0, 1] for i in range(num_sessions)]))
    corrcoef = np.array(corrcoef)
    corrcoef = np.nan_to_num(corrcoef, nan=-1)
    return np.array(corrcoef)


def sort_neurons(spat_dff_arr, day_idx=3):
    # Sort neurons by maximum activity location in last prestroke session
    sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(spat_dff_arr[:, day_idx])]
    sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    return spat_dff_arr[sort_key]


def sort_noncoding_neurons(spat_dff_arr, is_pc_list, noncoding_list, last_pre_idx=2):
    """ Sort neurons that are not coding in prestroke by the maximum of the first day where they are PCs. """

    # Todo

    return


spat_dff = []
# spat_dff.append(get_place_cells(is_pc[1]['41_1'], spat_dff_maps[1]['41_1'])[:, :-1])
# spat_dff.append(get_place_cells(is_pc[2]['69_1'], spat_dff_maps[2]['69_1']))
spat_dff.append(get_place_cells(is_pc[3]['85_1'], spat_dff_maps[3]['85_1']))
spat_dff.append(get_place_cells(is_pc[4]['90_1'], spat_dff_maps[4]['90_1']))
spat_dff = np.vstack(spat_dff)

spat_dff_noncoding = []
# spat_dff_noncoding.append(get_noncoding(is_pc[1]['41_1'], spat_dff_maps[1]['41_1'])[:, :-1])
# spat_dff_noncoding.append(get_noncoding(is_pc[2]['69_1'], spat_dff_maps[2]['69_1']))
spat_dff_noncoding.append(get_place_cells(is_pc[3]['85_1'], spat_dff_maps[3]['85_1']))
spat_dff_noncoding.append(get_place_cells(is_pc[4]['90_1'], spat_dff_maps[4]['90_1']))
spat_dff_noncoding = np.vstack(spat_dff_noncoding)

# split dataset into stable (upper 50%) and unstable (lower 50%) cells
corr = get_stab(spat_dff)
spat_dff_stable = spat_dff[corr > np.median(corr)]
spat_dff_unstable = spat_dff[corr <= np.median(corr)]

# Sort by maximum activity location
spat_dff_stable = sort_neurons(spat_dff_stable)
spat_dff_unstable = sort_neurons(spat_dff_unstable)
spat_dff_noncoding = sort_neurons(spat_dff_noncoding)


fig, ax = plt.subplots(3, 7, sharex='all', sharey='row', layout='constrained')
for i in range(2, 9):
    sns.heatmap(spat_dff_stable[:, i], ax=ax[0, i-2], cbar=False, cmap='turbo')
    sns.heatmap(spat_dff_unstable[:, i], ax=ax[1, i-2], cbar=False, cmap='turbo')
    sns.heatmap(spat_dff_noncoding[:, i], ax=ax[2, i-2], cbar=False, cmap='turbo')
    ax[0, i - 2].set_title(f'Day {is_pc[2]["69_1"][1][i]}')
ax[0, 0].set_ylabel('High-Correlation Place Cells')
ax[1, 0].set_ylabel('Low-Correlation Place Cells')
ax[2, 0].set_ylabel('Non-coding cells in pre')


#%% SECOND ATTEMPT
"""
Plotting:
- Three-part plot like in first attempt: All place cells on last prestroke day from several mice with deficit
- Sort by average stability (correlation) in prestroke phase and split 50-50
    (alternative keep top and bottom 30% and discard middle 30% to increase contrast between them)
- Plot EITHER selected single days (e.g. days -2, -1, 0, 3, 9, 15)
    OR
  Averaged traces across sessions within one period (pre, early, late)
  
Analysis (per network):
- In prestroke, get ratio between PCs/non-coding cells, sort PCs by average stability across period and split 50-50
- Do the same analysis for all subsequent days: split into PCs for each day, compute stability across period (early/late),
    use 50-50 threshold from prestroke to split PCs into stable/unstable
- Track ratio of non-coding - unstable PCs - stable PCs across time
- Construct transition matrix:
    - For each period/day?, how many cells stay in their group, how many move to a different group    
    - Include percentage of lost cells, that were not recorded in the next periods?
"""


def draw_heatmap_single_days(data_arrays, titles=None, draw_empty_row=True, draw_zone_borders=True):

    zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # Concatenate spatial maps and plot in single heatmap to give all cells the same row width
    if draw_empty_row:
        empty_row = np.zeros((1, *spat_dff_stable_sort.shape[1:])) * np.nan
        stacked_data = [np.vstack([x, empty_row]) for x in data_arrays[:-1]]
        stacked_data.append(data_arrays[-1])
        stacked_data = np.vstack(stacked_data)
    else:
        stacked_data = np.vstack(data_arrays)

    fig, ax = plt.subplots(1, stacked_data.shape[1], sharex='all', sharey='row', layout='constrained')
    for i in range(stacked_data.shape[1]):

        # Plot data in common heatmap
        sns.heatmap(stacked_data[:, i], ax=ax[i], cbar=False, cmap='turbo')

        # Set subplot titles to relative day
        if titles is not None:
            ax[i].set_title(f'Day {titles[i]}')

        if not draw_empty_row:
            # Draw horizontal lines to divide groups
            ax[i].axhline(len(spat_dff_stable_sort), c='red', linestyle='--')
            ax[i].axhline(len(spat_dff_stable_sort) + len(spat_dff_unstable_sort), c='red', linestyle='--')

        if draw_zone_borders:
            for z in zones:
                ax[i].axvline(z[0], linestyle='--', c='green')
                ax[i].axvline(z[1], linestyle='--', c='green')


# Deficit (recovery or no recovery) mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33'),
           # (common_match.MatchedIndex & 'mouse_id=38'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),
           # (common_match.MatchedIndex & 'mouse_id=93'),
           # (common_match.MatchedIndex & 'mouse_id=108' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=110' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=115' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=122' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=114' & 'day<"2022-09-09"')
)

# Sham mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33' & 'day<="2020-08-24"'),
           (common_match.MatchedIndex & 'mouse_id=121' & 'day<="2022-08-12"'),
           (common_match.MatchedIndex & 'mouse_id=115' & 'day<="2022-08-09"'),
           (common_match.MatchedIndex & 'mouse_id=122' & 'day<="2022-08-09"'),
           # (common_match.MatchedIndex & 'mouse_id=114' & 'day<="2022-08-09"')
)

is_pc = []
pfs = []
spatial_maps = []
match_matrices = []
spat_dff_maps = []

for query in queries:
    match_matrices.append(query.construct_matrix())

    is_pc.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
                                        extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                        return_array=True, relative_dates=True, surgery='Microsphere injection'))

    pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                      extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                      return_array=False, relative_dates=True, surgery='Microsphere injection'))

    # spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
    #                                            extra_restriction=dict(corridor_type=0, place_cell_id=2),
    #                                            return_array=True, relative_dates=True,
    #                                            surgery='Microsphere injection'))

    spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                                extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                return_array=True, relative_dates=True,
                                                surgery='Microsphere injection'))

# Stack together data from different mice
spat_dff = []
spat_dff.append(get_place_cells(is_pc_arr=is_pc[0]['41_1'], spat_dff_arr=spat_dff_maps[0]['41_1'], pc_day=-1, select_days=[-5, -4, -1, 2, 8, 17]))
spat_dff.append(get_place_cells(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 15]))
spat_dff.append(get_place_cells(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], pc_day=0, select_days=[-2, -1, 0, 3, 8, 17]))
spat_dff.append(get_place_cells(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], pc_day=0, select_days=[-2, -1, 0, 3, 8, 17]))
spat_dff.append(get_place_cells(is_pc_arr=is_pc[4]['121_1'], spat_dff_arr=spat_dff_maps[4]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
spat_dff = np.vstack(spat_dff)

spat_dff_noncoding = []
spat_dff_noncoding.append(get_noncoding(is_pc[0]['41_1'], spat_dff_maps[0]['41_1'], select_days=[-5, -4, -1, 2, 8, 17]))
spat_dff_noncoding.append(get_noncoding(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], select_days=[-2, -1, 0, 3, 9, 15]))
spat_dff_noncoding.append(get_noncoding(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], select_days=[-2, -1, 0, 3, 8, 17]))
spat_dff_noncoding.append(get_noncoding(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], select_days=[-2, -1, 0, 3, 8, 17]))
spat_dff_noncoding.append(get_noncoding(is_pc[4]['121_1'], spat_dff_maps[4]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
spat_dff_noncoding, noncoding_masks = list(map(list, zip(*spat_dff_noncoding)))
spat_dff_noncoding = np.vstack(spat_dff_noncoding)

# split dataset into stable (upper 50%) and unstable (lower 50%) cells
upper_percentile = 50
lower_percentile = 50

corr = get_stab(spat_dff)
upper_thresh = np.percentile(corr, upper_percentile)
lower_thresh = np.percentile(corr, lower_percentile)

spat_dff_stable = spat_dff[corr > upper_thresh]
spat_dff_unstable = spat_dff[corr <= lower_thresh]

# Sort by maximum activity location
spat_dff_stable_sort = sort_neurons(spat_dff_stable, day_idx=2)
spat_dff_unstable_sort = sort_neurons(spat_dff_unstable, day_idx=2)
spat_dff_noncoding_sort = sort_neurons(spat_dff_noncoding, day_idx=2)

# sort_noncoding_neurons(spat_dff_arr=spat_dff_noncoding, is_pc_list=is_pc, noncoding_list=noncoding_masks)

draw_heatmap_single_days(data_arrays=[spat_dff_stable_sort, spat_dff_unstable_sort, spat_dff_noncoding_sort],
                         titles=[-2, -1, 0, 3, 9, 18], draw_empty_row=False, draw_zone_borders=True)


######################### PLOT AVERAGE SPATIAL ACTIVITY MAPS FOR EACH PERIOD (PRE, EARLY, LATE) ########################

"""
Todo: Does this make sense? We would have to decide how often a place cell should be a place cell to be included.
And the quantification will most likely be done on individual days as well, so the plotting should reflect that.
"""

########################################################################################################################


