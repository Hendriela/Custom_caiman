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
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from schema import hheise_placecell, common_match, hheise_behav

# #%% Load matched/fetched data
# dir = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
# with open(os.path.join(dir, 'match_matrices.pickle'), "r") as output_file:
#     match_matrices = pickle.load(output_file)
# with open(os.path.join(dir, 'is_pc.pickle'), "r") as output_file:
#     is_pc = pickle.load(output_file)
# with open(os.path.join(dir, 'pfs.pickle'), "r") as output_file:
#     pfs = pickle.load(output_file)
# with open(os.path.join(dir, 'spatial_maps.pickle'), "r") as output_file:
#     spatial_maps = pickle.load(output_file)
# with open(os.path.join(dir, 'spat_dff_maps.pickle'), "r") as output_file:
#     spat_dff_maps = pickle.load(output_file)

#%% Functions


def get_place_cells(is_pc_arr, spat_arr, pc_day: int, select_days=None, ignore_missing=True):

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_arr = spat_arr[0]

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
    spat_filt = spat_arr[:, filter_date]
    spat_filt = spat_filt[filter_missing]
    spat_filt = spat_filt[filter_pc]

    return spat_filt

def get_place_cells_multiday(is_pc_arr, spat_arr, pc_frac, first_post=True, stab_in_pre=True):
    """ Get cells that are a PC at least 'pc_frac' % of prestroke sessions. Return average spatial activity maps
     of four periods: Prestroke, first poststroke session, other early poststroke sessions (<= d9), late sessions. """

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_arr = spat_arr[0]
    date_masks = []     # Masks for different periods

    # Get period masks
    pre_mask = dates <= 0
    date_masks.append(pre_mask)
    if first_post:
        first_post_mask = dates == np.min(np.where(dates > 0, dates, np.inf))
        early_mask = (dates > 0) & (dates <= 9) & ~first_post_mask      # If first poststroke session is plotted separately, exclude it from early
        date_masks.append(first_post_mask)
    else:
        early_mask = (dates > 0) & (dates <= 9)
    late_mask = dates > 9
    date_masks.append(early_mask)
    date_masks.append(late_mask)

    # For each cell, average spatial activity maps across each period
    avg_spat_arr = np.zeros((spat_arr.shape[0], len(date_masks), spat_arr.shape[2]))
    for i, cell_act in enumerate(spat_arr):
        for j, period in enumerate(date_masks):
            avg_spat_arr[i, j] = np.nanmean(cell_act[period], axis=0)

    # Select cells that occur at least once in every period
    occ_mask = np.sum(np.isnan(avg_spat_arr[:, :, 0]), axis=1) == 0

    # Select cells that are PCs in enough prestroke sessions
    pc_mask = np.nansum(is_pc_arr[:, pre_mask], axis=1) / np.sum(~np.isnan(is_pc_arr[:, pre_mask]), axis=1) >= pc_frac

    final_cell_mask = np.logical_and(occ_mask, pc_mask)

    # Apply masks
    avg_spat_arr_filt = avg_spat_arr[final_cell_mask]

    # Compute stability. Stability can be computed for the prestroke phase, or for all sessions
    if stab_in_pre:
        stab_sessions = np.argmax(np.where(dates < 0, dates, -np.inf))
    else:
        stab_sessions = None

    stab = get_stab(spat_arr=spat_arr[final_cell_mask], num_sessions=stab_sessions)

    # Return the average spatial activity maps of place cells, as well as computed stability for each cell
    return avg_spat_arr_filt, stab


def get_noncoding(is_pc_arr, spat_arr, select_days=None, ignore_missing=True, return_cell_mask=True):
    """ Only take cells that occur in all sessions and are not place cells in prestroke. """

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_arr = spat_arr[0]

    # Find indices of selected days
    filter_date = [np.where(dates == sel_day)[0][0] for sel_day in select_days]
    is_pc_filt = is_pc_arr[:, filter_date]
    spat_dff_filt = spat_arr[:, filter_date]

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


def get_noncoding_multiday(is_pc_arr, spat_arr, first_post):

    dates = np.array(is_pc_arr[1])
    is_pc_arr = is_pc_arr[0]
    spat_arr = spat_arr[0]
    date_masks = []     # Masks for different periods

    # Get period masks
    pre_mask = dates <= 0
    date_masks.append(pre_mask)
    if first_post:
        first_post_mask = dates == np.min(np.where(dates > 0, dates, np.inf))
        early_mask = (dates > 0) & (dates <= 9) & ~first_post_mask      # If first poststroke session is plotted separately, exclude it from early
        date_masks.append(first_post_mask)
    else:
        early_mask = (dates > 0) & (dates <= 9)
    late_mask = dates > 9
    date_masks.append(early_mask)
    date_masks.append(late_mask)

    # For each cell, average spatial activity maps across each period
    avg_spat_arr = np.zeros((spat_arr.shape[0], len(date_masks), spat_arr.shape[2]))
    for i, cell_act in enumerate(spat_arr):
        for j, period in enumerate(date_masks):
            avg_spat_arr[i, j] = np.nanmean(cell_act[period], axis=0)

    # Select cells that occur at least once in every period
    occ_mask = np.sum(np.isnan(avg_spat_arr[:, :, 0]), axis=1) == 0

    # Select cells that are PCs in enough prestroke sessions
    pc_mask = np.nansum(is_pc_arr[:, pre_mask], axis=1) / np.sum(~np.isnan(is_pc_arr[:, pre_mask]), axis=1) == 0

    final_cell_mask = np.logical_and(occ_mask, pc_mask)

    # Apply masks
    avg_spat_arr_filt = avg_spat_arr[final_cell_mask]
    return final_cell_mask, avg_spat_arr_filt


def get_stab(spat_arr, num_sessions=None):
    """ Session-session correlation. Correlate neighbouring sessions, irrespective of time distance. """
    corrcoef = []

    if num_sessions is None:
        num_sessions = spat_arr.shape[1] - 1

    for cell in spat_arr:
        corrcoef.append(np.nanmean([np.corrcoef(cell[i], cell[i+1])[0, 1] for i in range(num_sessions)]))
    corrcoef = np.array(corrcoef)

    # If a cell was not found in neighbouring sessions, its correlation will be nan. For these cells, check if they
    # were found in more than one session. If yes, compute stability between all session combinations. Otherwise,
    # the correlation remains nan.
    nan_cells = np.where(np.isnan(corrcoef))[0]
    for nan_cell in nan_cells:
        if np.sum(~np.isnan(spat_arr[nan_cell, :num_sessions+1, 0])) > 1:
            non_nan_sessions = np.where(~np.isnan(spat_arr[nan_cell, :num_sessions+1, 0]))[0]
            corrcoef[nan_cell] = np.nanmean([np.corrcoef(spat_arr[nan_cell, non_nan_sessions[i]],
                                                         spat_arr[nan_cell, non_nan_sessions[i + 1]])[0, 1]
                                             for i in range(len(non_nan_sessions)-1)])
    # corrcoef = np.nan_to_num(corrcoef, nan=-1)

    return np.array(corrcoef)


def sort_neurons(spat_dff_arr, day_idx=3):
    # Sort neurons by maximum activity location in last prestroke session
    sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(spat_dff_arr[:, day_idx])]
    sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    return spat_dff_arr[sort_key]


# def sort_noncoding_neurons(spat_dff_arr, is_pc_list, noncoding_list, last_pre_idx=2):
#     """ Sort neurons that are not coding in prestroke by the maximum of the first day where they are PCs. """
#
#     # Todo
#
#     return


#%% Calling functions
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


def draw_heatmap_across_days(data_arrays, titles=None, draw_empty_row=True, draw_zone_borders=True):
    """
    Plot a series of heatmaps. One heatmap per day, multiple days horizontally, sets of cells split vertically in each
    heatmap.

    Args:
        data_arrays: List of 3D numpy arrays with shape (n_cells, n_sessions, n_bins). One array for one type of
            cell (e.g. stable cells, unstable cells, noncoding cells)
        titles:
        draw_empty_row:
        draw_zone_borders:

    Returns:

    """

    if draw_zone_borders:
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # Concatenate spatial maps and plot in single heatmap to give all cells the same row width
    if draw_empty_row:
        empty_row = np.zeros((1, *data_arrays[0].shape[1:])) * np.nan
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
            ax[i].axhline(len(data_arrays[0]), c='red', linestyle='--')
            ax[i].axhline(len(data_arrays[0]) + len(data_arrays[1]), c='red', linestyle='--')

        if draw_zone_borders:
            for z in zones:
                ax[i].axvline(z[0], linestyle='--', c='green')
                ax[i].axvline(z[1], linestyle='--', c='green')


def transition_heatmap(classes: pd.DataFrame, spat_arr: list, to_period: str, place_cells: bool):
    """ Class_df from quantify_stability_split(), already selected all mice whose cells should be included."""
    # Transform list into dict for easier querying
    spat_dict = {k: v for dic in spat_arr for k, v in dic.items()}

    if to_period == 'early':
        from_period = 'pre'
    elif to_period == 'late':
        from_period = 'early'
    else:
        raise NameError(f'Argument "to_period" has invalid value: {to_period}')

    from_act = []
    to_stable = []
    to_unstable = []
    to_noncoding = []
    for curr_mouse in classes['mouse_id'].unique():

        # Make date masks
        curr_dates = np.array(spat_dict[curr_mouse][1])
        if to_period == 'early':
            from_dates = curr_dates <= 0
            to_dates = (curr_dates > 0) & (curr_dates <= 9)
        else:
            from_dates = (curr_dates > 0) & (curr_dates <= 9)
            to_dates = curr_dates > 9

        # Heatmap 1: All PCs (classes 2+3) in phase 1 (pre or early)
        from_classes = classes[(classes['mouse_id'] == curr_mouse) & (classes['period'] == from_period)]['classes'].iloc[0]
        if place_cells:
            cell_mask = (from_classes == 2) | (from_classes == 3)
        else:
            cell_mask = from_classes == 1

        # Average activity across the whole to_period for the first heatmap
        curr_act = spat_dict[curr_mouse][0][cell_mask]
        curr_act = curr_act[:, from_dates]
        from_act.append(np.nanmean(curr_act, axis=1))

        # Heatmap 2: All PCs that remained stable in phase 2 (early or late)
        to_classes = classes[(classes['mouse_id'] == curr_mouse) & (classes['period'] == to_period)]['classes'].iloc[0]
        curr_act = spat_dict[curr_mouse][0][cell_mask & (to_classes == 3)]
        to_stable.append(np.nanmean(curr_act[:, to_dates], axis=1))

        # Heatmap 3: All PCs that became unstable in phase 2
        curr_act = spat_dict[curr_mouse][0][cell_mask & (to_classes == 2)]
        to_unstable.append(np.nanmean(curr_act[:, to_dates], axis=1))

        # Heatmap 4: All PCs that became non-coding in phase 2
        curr_act = spat_dict[curr_mouse][0][cell_mask & (to_classes == 1)]
        to_noncoding.append(np.nanmean(curr_act[:, to_dates], axis=1))

    output = {'from_act': np.concatenate(from_act), 'to_stable': np.concatenate(to_stable),
              'to_unstable': np.concatenate(to_unstable), 'to_noncoding': np.concatenate(to_noncoding)}

    return output


def plot_transition_heatmaps(early_maps, late_maps, cmap='turbo', title=None):

    def plot_transition_heatmap(maps, f, gs, from_title=None, to_title=None):

        def sort_array(arr):
            sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(arr)]
            sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
            return arr[sort_key]

        gs1 = gs.subgridspec(1, 2)      # Create inner Gridspec (1 row, 2 columns)

        ax_from = f.add_subplot(gs1[0])      # Axis for "from" large heatmap
        ax_to = f.add_subplot(gs1[1])        # Axis for to_stable heatmap

        # Plot "from" data sorted by maximum
        sns.heatmap(sort_array(maps['from_act']), ax=ax_from, cbar=False, cmap=cmap)

        # Sort "to" data by maximum separately for each group, and concatenate to single array, separated by single row
        to_arr = [sort_array(x) for x in [maps['to_stable'], maps['to_unstable'], maps['to_noncoding']]]
        empty_row = np.zeros((1, *to_arr[0].shape[1:])) * np.nan
        stacked_data = [np.vstack([x, empty_row]) for x in to_arr[:-1]]
        stacked_data.append(to_arr[-1])
        stacked_data.append(np.zeros((len(maps['from_act'])-len(np.vstack(to_arr))-2, *to_arr[0].shape[1:])) * np.nan)   # Append empty rows for lost neurons
        stacked_data = np.vstack(stacked_data)

        sns.heatmap(stacked_data, ax=ax_to, cbar=False, cmap=cmap)

        # Use Y-axis labels for cell numbers
        ax_from.set_yticks([])
        ax_from.set_xticklabels(np.floor(ax_from.get_xticks()).astype(int), rotation=0)
        y_pos = 0
        y_tick_pos = []
        y_tick_label = []
        for a in to_arr:
            y_tick_pos.append(len(a)//2 + y_pos)
            y_tick_label.append(f'{len(a)}\n{(len(a)/len(maps["from_act"]))*100:.0f}%')
            y_pos += len(a)
        ax_to.set(yticks=y_tick_pos, yticklabels=y_tick_label)
        ax_to.set_xticklabels(np.floor(ax_to.get_xticks()).astype(int), rotation=0)

        if from_title is not None:
            ax_from.set_title(from_title)
        if to_title is not None:
            ax_to.set_title(to_title)

    fig = plt.figure()
    gs_outer = gridspec.GridSpec(1, 2, figure=fig)   # Splits figure into two halves (early and late)

    plot_transition_heatmap(maps=early_maps, f=fig, gs=gs_outer[0], from_title='Pre', to_title='Early')
    plot_transition_heatmap(maps=late_maps, f=fig, gs=gs_outer[1], from_title='Early', to_title='Late')

    if title is not None:
        fig.suptitle(title)

#%%
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
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=115' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=122' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=114' & 'day<"2022-09-09"')
)

# Sham mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33' & 'day<="2020-08-24"'),
           (common_match.MatchedIndex & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'mouse_id=122' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'mouse_id=121' & 'day<"2022-09-09"'),
    # (common_match.MatchedIndex & 'mouse_id=114' & 'day<="2022-08-09"')
)

# All mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33'),
           # (common_match.MatchedIndex & 'mouse_id=38'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=93'),
           # (common_match.MatchedIndex & 'mouse_id=108' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=110' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=114' & 'day<"2022-09-09"')
)

is_pc = []
pfs = []
# spatial_maps = []
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

separate_days = False

if separate_days:
    # Stack together data from different mice (deficit)
    spat = []
    spat.append(get_place_cells(is_pc_arr=is_pc[0]['41_1'], spat_arr=spat_dff_maps[0]['41_1'], pc_day=-1, select_days=[-5, -4, -1, 2, 8, 17]))
    spat.append(get_place_cells(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 15]))
    spat.append(get_place_cells(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], pc_day=0, select_days=[-2, -1, 0, 3, 8, 17]))
    spat.append(get_place_cells(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], pc_day=0, select_days=[-2, -1, 0, 3, 8, 17]))
    # spat_dff.append(get_place_cells(is_pc_arr=is_pc[4]['121_1'], spat_dff_arr=spat_dff_maps[4]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat = np.vstack(spat)

    spat_noncoding = []
    spat_noncoding.append(get_noncoding(is_pc[0]['41_1'], spat_dff_maps[0]['41_1'], select_days=[-5, -4, -1, 2, 8, 17]))
    spat_noncoding.append(get_noncoding(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], select_days=[-2, -1, 0, 3, 9, 15]))
    spat_noncoding.append(get_noncoding(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], select_days=[-2, -1, 0, 3, 8, 17]))
    spat_noncoding.append(get_noncoding(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], select_days=[-2, -1, 0, 3, 8, 17]))
    # spat_dff_noncoding.append(get_noncoding(is_pc[4]['121_1'], spat_dff_maps[4]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    spat_noncoding, noncoding_masks = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

    # Stack together data from different mice (sham)
    spat = []
    spat.append(get_place_cells(is_pc_arr=is_pc[0]['115_1'], spat_arr=spat_dff_maps[0]['115_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat.append(get_place_cells(is_pc_arr=is_pc[1]['122_1'], spat_arr=spat_dff_maps[1]['122_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat.append(get_place_cells(is_pc_arr=is_pc[2]['121_1'], spat_arr=spat_dff_maps[2]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat = np.vstack(spat)

    spat_noncoding = []
    spat_noncoding.append(get_noncoding(is_pc[0]['115_1'], spat_dff_maps[0]['115_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    spat_noncoding.append(get_noncoding(is_pc[1]['122_1'], spat_dff_maps[1]['122_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    spat_noncoding.append(get_noncoding(is_pc[2]['121_1'], spat_dff_maps[2]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    spat_noncoding, noncoding_masks = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

else:
    # AVERAGE DIFFERENT PERIODS

    first_poststroke_separate = True    # Whether the first poststroke session should be shown separately or included in early phase
    pc_fraction = 0.2   # Fraction of sessions that a cell has to be a PC to be included

    # Stack together data from different mice (deficit)
    spat = []
    spat.append(get_place_cells_multiday(is_pc_arr=is_pc[0]['41_1'], spat_arr=spat_dff_maps[0]['41_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(get_place_cells_multiday(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(get_place_cells_multiday(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(get_place_cells_multiday(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    # spat.append(get_place_cells_multiday(is_pc_arr=is_pc[4]['121_1'], spat_arr=spat_dff_maps[4]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat, pc_stab = list(map(list, zip(*spat)))
    spat = np.vstack(spat)
    pc_stab = np.concatenate(pc_stab)

    spat_noncoding = []
    spat_noncoding.append(get_noncoding_multiday(is_pc[0]['41_1'], spat_dff_maps[0]['41_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(get_noncoding_multiday(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(get_noncoding_multiday(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(get_noncoding_multiday(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], first_post=first_poststroke_separate))
    # spat_dff_noncoding.append(get_noncoding(is_pc[4]['121_1'], spat_dff_maps[4]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    noncoding_masks, spat_noncoding = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

    # Stack together data from different mice (sham)
    spat = []
    spat.append(get_place_cells_multiday(is_pc_arr=is_pc[4]['115_1'], spat_arr=spat_dff_maps[4]['115_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(get_place_cells_multiday(is_pc_arr=is_pc[5]['122_1'], spat_arr=spat_dff_maps[5]['122_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(get_place_cells_multiday(is_pc_arr=is_pc[6]['121_1'], spat_arr=spat_dff_maps[6]['121_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat, pc_stab = list(map(list, zip(*spat)))
    spat = np.vstack(spat)
    pc_stab = np.concatenate(pc_stab)

    spat_noncoding = []
    spat_noncoding.append(get_noncoding_multiday(is_pc[4]['115_1'], spat_dff_maps[4]['115_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(get_noncoding_multiday(is_pc[5]['122_1'], spat_dff_maps[5]['122_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(get_noncoding_multiday(is_pc[6]['121_1'], spat_dff_maps[6]['121_1'], first_post=first_poststroke_separate))
    noncoding_masks, spat_noncoding = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)


# split dataset into stable (upper 50%) and unstable (lower 50%) cells
upper_percentile = 50
lower_percentile = 50

if separate_days:
    pc_stab = get_stab(spat)
    upper_thresh = np.percentile(pc_stab, upper_percentile)
    lower_thresh = np.percentile(pc_stab, lower_percentile)
else:
    upper_thresh = np.nanpercentile(pc_stab, upper_percentile)
    lower_thresh = np.nanpercentile(pc_stab, lower_percentile)

spat_stable = spat[pc_stab > upper_thresh]
spat_unstable = spat[pc_stab <= lower_thresh]
spat_unstable = np.concatenate((spat_unstable, spat[np.isnan(pc_stab)]))    # Add cells with only 1 prestroke session (no stability) to unstable

# Sort by maximum activity location
spat_stable_sort = sort_neurons(spat_stable, day_idx=0)
spat_unstable_sort = sort_neurons(spat_unstable, day_idx=0)
spat_noncoding_sort = sort_neurons(spat_noncoding, day_idx=0)

# sort_noncoding_neurons(spat_dff_arr=spat_dff_noncoding, is_pc_list=is_pc, noncoding_list=noncoding_masks)

# DEFICIT: Filter out some bad neurons
spat_noncoding_sort = np.delete(spat_noncoding_sort, [133, 228, 428], axis=0)

draw_heatmap_across_days(data_arrays=[spat_stable_sort, spat_unstable_sort, spat_noncoding_sort],
                         titles=['pre', 'first post', 'early', 'late'], draw_empty_row=True, draw_zone_borders=True)

#%%
############################################### QUANTIFICATION #########################################################


def quantify_stability_split(is_pc_arr, spat_dff_arr, rel_days, day_diff = 3, stab_thresh = None, mouse_id=None):
    """
    Split cells into non-coding (never a PC) and coding (at least once a PC). Compute stability across prestroke
    sessions for all coding cells. Baseline threshold for the network is the median correlation of all PCs.
    Get ratio of non-coding - unstable - stable place cells.
    Called once per mouse.

    Args:
        is_pc_arr: is_pc output for one mouse. Already restricted for a certain time period.
        spat_dff_arr: spat_dff output for one mouse.  Already restricted for a certain time period.
    """

    # Filter cells that are place cells at least once
    filter_pc = np.nansum(is_pc_arr, axis=1) > 0

    filter_idx = np.where(filter_pc)[0]   # Indices of place cells to keep track of their attributes

    # Compute stability across all session pairs of all place cells
    stab = []
    for cell_data in spat_dff_arr[filter_pc]:
        cell_corr = []
        for day_idx, day in enumerate(rel_days):
            next_day_idx = np.where(rel_days == day + day_diff)[0]
            if len(next_day_idx) == 1:
                cell_corr.append(np.corrcoef(cell_data[day_idx], cell_data[next_day_idx[0]])[0, 1])
        stab.append(np.nanmean(cell_corr))

    # Some cells did not occur in days with distance day_diff, they have to be excluded as well
    filter_idx = filter_idx[~np.isnan(stab)]
    stab = np.array(stab)[~np.isnan(stab)]

    # If not provided (for prestroke period), get median of stability (threshold for further stability classifications)
    if stab_thresh is None:
        stab_thresh = np.median(stab)
        print(f'Mouse {mouse_id}: {stab_thresh:.4f} stability threshold.')
        return_stab = True
    else:
        # print(f'Using provided stability threshold: {stab_thresh:.4f}')
        return_stab = False

    # Get numbers and ratio of non-coding, unstable and stable cells, as well as classification mask
    stable_mask = np.zeros(len(is_pc_arr), dtype=bool)
    stable_mask[filter_idx[stab > stab_thresh]] = 1
    unstable_mask = np.zeros(len(is_pc_arr), dtype=bool)
    unstable_mask[filter_idx[stab <= stab_thresh]] = 1

    noncoding_mask = np.nansum(is_pc_arr, axis=1) == 0
    noncoding_mask[np.isnan(is_pc_arr).sum(axis=1) == is_pc_arr.shape[1]] = False   # Exclude cells that were not present at all

    """    
    Construct single mask with cell classification:
        Class 0: Rare cells that did not appear in enough sessions to be analysed (cells that did not appear in period 
            at all, and place cells that did not appear on days with distance day_diff).
        Class 1: Non-coding cells (cells that are never a PC in the period)
        Class 2: Cells that are a PC in at least one session during the period and have an average cross-session 
            stability below the threshold.
        Class 3: Cells that are a PC in at least one session during the period and have an average cross-session 
            stability above the threshold.
    """
    class_mask = np.zeros(len(is_pc_arr), dtype=int)
    class_mask[noncoding_mask] = 1
    class_mask[unstable_mask] = 2
    class_mask[stable_mask] = 3

    if return_stab:
        return class_mask, stab_thresh
    else:
        return class_mask


def summarize_quantification(class_mask, period='None'):

    return pd.DataFrame([dict(n0=np.sum(class_mask == 0), n1=np.sum(class_mask == 1),
                              n2=np.sum(class_mask == 2), n3=np.sum(class_mask == 3),
                              n0_r=(np.sum(class_mask == 0)/len(class_mask))*100,
                              n1_r=(np.sum(class_mask == 1)/len(class_mask))*100,
                              n2_r=(np.sum(class_mask == 2)/len(class_mask))*100,
                              n3_r=(np.sum(class_mask == 3)/len(class_mask))*100,
                              period=period)])


def transition_matrix(mask1, mask2, num_classes=4, percent=True):

    # Create empty transition matrix
    trans = np.zeros((num_classes, num_classes), dtype=int)

    for prev_class, curr_class in zip(mask1, mask2):
        trans[prev_class, curr_class] += 1

    if percent:
        trans = (trans/len(mask1)) * 100

    return trans


deficit = ['41_1', '69_1', '85_1', '90_1']
no_recovery = ['41_1', '69_1']
recovery = ['85_1', '90_1']
sham = ['115_1', '122_1', '121_1']
mice = ['41_1', '69_1', '85_1', '90_1', '115_1', '122_1', '121_1']
dfs = []
for i, mouse in enumerate(mice):
    rel_dates = np.array(is_pc[i][mouse][1])
    mask_pre = rel_dates <= 0
    mask_early = (0 < rel_dates) & (rel_dates <= 9)
    mask_late = rel_dates > 9

    classes_pre, stability_thresh = quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_pre],
                                                             spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_pre],
                                                             rel_days=rel_dates[mask_pre], mouse_id=mouse)
    classes_early = quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_early],
                                             spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_early],
                                             rel_days=rel_dates[mask_early], stab_thresh=stability_thresh)
    classes_late = quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_late],
                                            spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_late],
                                            rel_days=rel_dates[mask_late], stab_thresh=stability_thresh)

    df = pd.concat([summarize_quantification(classes_pre, 'pre'),
                    summarize_quantification(classes_early, 'early'),
                    summarize_quantification(classes_late, 'late')])
    df['mouse_id'] = mouse
    df['classes'] = [classes_pre, classes_early, classes_late]
    dfs.append(df)
class_df = pd.concat(dfs, ignore_index=True)


# Plot transition matrices
def label_row(label, axis, pad=5):
    axis.annotate(label, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0), xycoords=axis.yaxis.label,
                  textcoords='offset points', size='large', ha='right', va='center')


fig, ax = plt.subplots(2, len(class_df.mouse_id.unique()), layout='constrained')
matrices = []
for i, mouse in enumerate(class_df.mouse_id.unique()):
    mask_pre = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'pre')]['classes'].iloc[0]
    mask_early = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'early')]['classes'].iloc[0]
    mask_late = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'late')]['classes'].iloc[0]

    pre_early_trans = transition_matrix(mask_pre, mask_early)
    early_late_trans = transition_matrix(mask_early, mask_late)

    sns.heatmap(pre_early_trans, ax=ax[0, i], square=True, annot=True, cbar=False)
    sns.heatmap(early_late_trans, ax=ax[1, i], square=True, annot=True, cbar=False)
    ax[0, i].set_title(mouse)

    matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early_trans, early_late=early_late_trans)]))
matrices = pd.concat(matrices, ignore_index=True)

# label_row('Pre->Early', ax[0, 0])
# label_row('Early->Late', ax[1, 0])
ax[0, 0].set_xlabel('To Cell Class Early')
ax[0, 0].set_ylabel('From Cell Class Pre')
ax[1, 0].set_xlabel('To Cell Class Late')
ax[1, 0].set_ylabel('From Cell Class Early')

# Make average transition matrices
deficit_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(deficit)]['pre_early'])), axis=0)
deficit_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(deficit)]['early_late'])), axis=0)

no_recovery_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(no_recovery)]['pre_early'])), axis=0)
no_recovery_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(no_recovery)]['early_late'])), axis=0)

recovery_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(recovery)]['pre_early'])), axis=0)
recovery_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(recovery)]['early_late'])), axis=0)

sham_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(sham)]['pre_early'])), axis=0)
sham_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(sham)]['early_late'])), axis=0)

fig, ax = plt.subplots(2, 4, layout='constrained')
titles = ['deficit all', 'no recovery', 'recovery', 'sham']
for i, mat in enumerate([[deficit_early, deficit_late], [no_recovery_early, no_recovery_late],
                         [recovery_early, recovery_late], [sham_early, sham_late]]):
    sns.heatmap(mat[0], ax=ax[0, i], square=True, annot=True, cbar=False)
    sns.heatmap(mat[1], ax=ax[1, i], square=True, annot=True, cbar=False)
    ax[0, i].set_title(titles[i])
ax[0, 0].set_xlabel('To Cell Class Early')
ax[0, 0].set_ylabel('From Cell Class Pre')
ax[1, 0].set_xlabel('To Cell Class Late')
ax[1, 0].set_ylabel('From Cell Class Early')


### Transition heatmaps
# Cell numbers in early and late dont sum up because there are some cells that transitioned from noncoding to PC
def_heat_early = transition_heatmap(classes=class_df[class_df['mouse_id'].isin(deficit)], spat_arr=spatial_maps,
                                    to_period='early', place_cells=True)
def_heat_late = transition_heatmap(classes=class_df[class_df['mouse_id'].isin(deficit)], spat_arr=spatial_maps,
                                   to_period='late', place_cells=True)

sham_heat_early = transition_heatmap(classes=class_df[class_df['mouse_id'].isin(sham)], spat_arr=spatial_maps,
                                     to_period='early', place_cells=True)
sham_heat_late = transition_heatmap(classes=class_df[class_df['mouse_id'].isin(sham)], spat_arr=spatial_maps,
                                    to_period='late', place_cells=True)

## Plot heatmaps
plot_transition_heatmaps(early_maps=def_heat_early, late_maps=def_heat_late)     # Deficit place cells
plot_transition_heatmaps(early_maps=sham_heat_early, late_maps=sham_heat_late)     # Deficit place cells

#%% Transition matrix visualizations
### Visualize transition matrices with hmmviz
from hmmviz import TransGraph

# Use pandas for cross-tabulation (transition matrix)
classes = ['lost', 'non-coding', 'unstable', 'stable']
deficit_early_tab = pd.DataFrame(deficit_early, columns=classes, index=classes)
deficit_late_tab = pd.DataFrame(deficit_late, columns=classes, index=classes)
no_recovery_early_tab = pd.DataFrame(no_recovery_early, columns=classes, index=classes)
no_recovery_late_tab = pd.DataFrame(no_recovery_late, columns=classes, index=classes)
recovery_early_tab = pd.DataFrame(recovery_early, columns=classes, index=classes)
recovery_late_tab = pd.DataFrame(recovery_late, columns=classes, index=classes)
sham_early_tab = pd.DataFrame(sham_early, columns=classes, index=classes)
sham_late_tab = pd.DataFrame(sham_late, columns=classes, index=classes)

graph = TransGraph(deficit_early_tab/100)
fig = plt.figure(figsize=(6, 6))
graph.draw(edgelabels=True)

# Export classes for Plotly Sankey Diagram
matrices = []
for i, mouse in enumerate(class_df.mouse_id.unique()):
    mask_pre = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'pre')]['classes'].iloc[0]
    mask_early = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'early')]['classes'].iloc[0]
    mask_late = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'late')]['classes'].iloc[0]

    pre_early_trans = transition_matrix(mask_pre, mask_early, percent=False)
    early_late_trans = transition_matrix(mask_early, mask_late, percent=False)

    matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early_trans, early_late=early_late_trans)]))
matrices = pd.concat(matrices, ignore_index=True)

# Make average transition matrices
deficit_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(deficit)]['pre_early'])), axis=0)
deficit_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(deficit)]['early_late'])), axis=0)

no_recovery_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(no_recovery)]['pre_early'])), axis=0)
no_recovery_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(no_recovery)]['early_late'])), axis=0)

recovery_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(recovery)]['pre_early'])), axis=0)
recovery_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(recovery)]['early_late'])), axis=0)

sham_early = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(sham)]['pre_early'])), axis=0)
sham_late = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(sham)]['early_late'])), axis=0)


def unravel_matrix(mat_early, mat_late):
    source_early = [it for sl in [[i]*4 for i in range(4)] for it in sl]  # "From" class
    target_early = [4,5,6,7]*4                      # "To" classes (shifted by 4)
    source_late = np.array(source_early) + 4
    target_late = np.array(target_early) + 4        # Classes from early->late have different labels (shifted by 4 again)

    early_flat = mat_early.flatten()
    late_flat = mat_late.flatten()

    # Concatenate everything into three rows (source, target, value)
    out = np.array([[*source_early, *source_late], [*target_early, *target_late], [*early_flat, *late_flat]])
    return out


np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\pc_heatmaps\sankey_matrices\deficit.csv',
           unravel_matrix(deficit_early, deficit_late), fmt="%d", delimiter=',')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\pc_heatmaps\sankey_matrices\no_recovery.csv',
           unravel_matrix(no_recovery_early, no_recovery_late), fmt="%d", delimiter=',')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\pc_heatmaps\sankey_matrices\recovery.csv',
           unravel_matrix(recovery_early, recovery_late), fmt="%d", delimiter=',')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\pc_heatmaps\sankey_matrices\sham.csv',
           unravel_matrix(sham_early, sham_late), fmt="%d", delimiter=',')