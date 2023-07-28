#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 09/07/2023 10:41
@author: hheise


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats

from schema import hheise_behav


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
        early_mask = (dates > 0) & (dates <= 6) & ~first_post_mask      # If first poststroke session is plotted separately, exclude it from early
        date_masks.append(first_post_mask)
    else:
        early_mask = (dates > 0) & (dates <= 6)
    late_mask = dates > 6
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
        early_mask = (dates > 0) & (dates <= 6) & ~first_post_mask      # If first poststroke session is plotted separately, exclude it from early
        date_masks.append(first_post_mask)
    else:
        early_mask = (dates > 0) & (dates <= 6)
    late_mask = dates > 6
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
        sns.heatmap(stacked_data[:, i], ax=ax[i], cbar=False, cmap='turbo', vmin=0)

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
            to_dates = (curr_dates > 0) & (curr_dates <= 6)
        else:
            from_dates = (curr_dates > 0) & (curr_dates <= 6)
            to_dates = curr_dates > 6

        # Heatmap 1: All PCs (classes 2+3) in phase 1 (pre or early)
        from_classes = \
        classes[(classes['mouse_id'] == curr_mouse) & (classes['period'] == from_period)]['classes'].iloc[0]
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


def plot_transition_heatmaps(early_maps, late_maps, cmap='turbo', title=None, draw_zone_borders=True, n_empty=1):
    def plot_transition_heatmap(maps, f, gs, from_title=None, to_title=None, rw_zones=None):

        def sort_array(arr):
            sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(arr)]
            sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
            return arr[sort_key]

        gs1 = gs.subgridspec(1, 2)  # Create inner Gridspec (1 row, 2 columns)

        ax_from = f.add_subplot(gs1[0])  # Axis for "from" large heatmap
        ax_to = f.add_subplot(gs1[1])  # Axis for to_stable heatmap

        # Plot "from" data sorted by maximum
        sns.heatmap(sort_array(maps['from_act']), ax=ax_from, cbar=False, cmap=cmap, vmin=0)

        # Sort "to" data by maximum separately for each group, and concatenate to single array, separated by single row
        to_arr = [sort_array(x) for x in [maps['to_stable'], maps['to_unstable'], maps['to_noncoding']]]
        empty_row = np.zeros((n_empty, *to_arr[0].shape[1:])) * np.nan
        stacked_data = [np.vstack([x, empty_row]) for x in to_arr[:-1]]
        stacked_data.append(to_arr[-1])
        try:
            stacked_data.append(np.zeros((len(maps['from_act']) - len(np.vstack(to_arr)) - 2,
                                          *to_arr[0].shape[1:])) * np.nan)  # Append empty rows for lost neurons
        except ValueError:
            pass    # If no neurons are lost, dont add empty rows
        stacked_data = np.vstack(stacked_data)

        sns.heatmap(stacked_data, ax=ax_to, cbar=False, cmap=cmap, vmin=0)

        # Use Y-axis labels for cell numbers
        ax_from.set_yticks([])
        ax_from.set_xticklabels(np.floor(ax_from.get_xticks()).astype(int), rotation=0)
        y_pos = 0
        y_tick_pos = []
        y_tick_label = []
        for a in to_arr:
            y_tick_pos.append(len(a) // 2 + y_pos)
            y_tick_label.append(f'{len(a)}\n{(len(a) / len(maps["from_act"])) * 100:.0f}%')
            y_pos += len(a)
        ax_to.set(yticks=y_tick_pos, yticklabels=y_tick_label)
        ax_to.set_xticklabels(np.floor(ax_to.get_xticks()).astype(int), rotation=0)

        if rw_zones is not None:
            for z in rw_zones:
                ax_to.axvline(z[0], linestyle='--', c='green')
                ax_to.axvline(z[1], linestyle='--', c='green')
                ax_from.axvline(z[0], linestyle='--', c='green')
                ax_from.axvline(z[1], linestyle='--', c='green')

        if from_title is not None:
            ax_from.set_title(from_title)
        if to_title is not None:
            ax_to.set_title(to_title)

    if draw_zone_borders:
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)
    else:
        zones = None

    fig = plt.figure()
    gs_outer = gridspec.GridSpec(1, 2, figure=fig)  # Splits figure into two halves (early and late)

    plot_transition_heatmap(maps=early_maps, f=fig, gs=gs_outer[0], from_title='Pre', to_title='Early', rw_zones=zones)
    plot_transition_heatmap(maps=late_maps, f=fig, gs=gs_outer[1], from_title='Early', to_title='Late', rw_zones=zones)

    if title is not None:
        fig.suptitle(title)

#%%
############################################### QUANTIFICATION #########################################################


def pivot_classdf_prism(df, percent=True, col_order=None):

    if percent:
        value_columns = ['n0_r', 'n1_r', 'n2_r', 'n3_r']
    else:
        value_columns = ['n0', 'n1', 'n2', 'n3']

    if col_order is None:
        col_order = ['pre', 'early', 'late']

    mouse_ids = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 112, 113, 114, 115, 116, 121, 122]
    cols = [f'{col}_{m_id}' for col in col_order for m_id in mouse_ids]

    # Transform dataframe to long format before pivoting
    df_melt = pd.melt(df, id_vars=['period', 'mouse_id'], value_vars=value_columns)
    df_melt['mouse_id'] = df_melt['mouse_id'].str[:-2].astype(int)

    df_new = pd.DataFrame(columns=cols, index=value_columns)

    for i, row in df_melt.iterrows():
        df_new.at[row['variable'], f'{row["period"]}_{row["mouse_id"]}'] = row['value']

    return df_new


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


def export_trans_matrix(mat1, mat2):

    export = np.zeros((np.max([len(mat1), len(mat2)]), (mat1.shape[1] * mat1.shape[2])*2)) * np.nan
    col = 0
    for i in range(mat1.shape[1]):
        for j in range(mat1.shape[2]):
            export[:len(mat1[:, i, j]), col] = mat1[:, i, j]
            export[:len(mat2[:, i, j]), col + 1] = mat2[:, i, j]
            col += 2
    return export


def transition_matrix_ttest(mat1, mat2):

    ttest_p = np.zeros(mat1.shape[1:])
    for i in range(mat1.shape[1]):
        for j in range(mat1.shape[2]):
            ttest_p[i, j] = stats.ttest_ind(mat1[:, i, j], mat2[:, i, j]).pvalue
    return ttest_p

