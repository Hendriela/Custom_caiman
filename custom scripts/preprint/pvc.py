#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/04/2023 14:00
@author: hheise

"""
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from schema import hheise_placecell, common_match, hheise_behav, hheise_pvc, hheise_grouping
import preprint.data_cleaning as dc


def plot_pvc_curve(y_vals, session_stdev, bin_size=5, show=False):
    """Plots the pvc curve

        Parameters
        ----------
        y_vals : array-like
            data points of the pvc curve (idx = bin distance)
        bin_size : bool, optional
        show : bool, optional

       Returns
       -------
       fig: figure object
           a figure object of the pvc curve
    """
    fig = plt.figure()
    x_axis = np.arange(0., len(y_vals) * bin_size, bin_size)  # bin size
    plt.errorbar(x_axis, y_vals, session_stdev, figure=fig)
    plt.ylim(bottom=0); plt.ylabel('Mean PVC')
    plt.xlim(left=0); plt.xlabel('Offset Distances (cm)')
    if show:
        plt.show(block=True)
    return fig


def pvc_curve(session1, session2, plot=True, max_delta_bins=79, circular=False):
    """Calculate the mean pvc curve between two sessions.

        Parameters
        ----------
        activity_matrix : 2D array containing (float, dim1 = bins, dim2 = neurons)
        plot: bool, optional
        max_delta_bins: int, optional
            max difference in bin distance. Default is entire corridor.

       Returns
       -------
       curve_yvals:
           array of mean pvc curve (idx = delta_bin)
    """

    # Filter out neurons that are not active in both sessions
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask].T
    session2 = session2[neuron_mask].T
    logging.info(f'{np.sum(neuron_mask)} neurons available')

    num_bins = np.size(session1, 0)
    num_neurons = np.size(session1, 1)
    curve_yvals = np.empty(max_delta_bins + 1)
    curve_stdev = np.empty(max_delta_bins + 1)

    pvc_mat = np.zeros((max_delta_bins + 1, num_bins)) * np.nan

    for delta_bin in range(max_delta_bins + 1):
        pvc_vals = []

        if circular:
            max_offset = num_bins
        else:
            max_offset = num_bins - delta_bin

        for offset in range(max_offset):
            idx_x = offset

            if circular:
                idx_y = offset + (delta_bin - num_bins)     # This wraps around the end of the corridor
            else:
                idx_y = offset + delta_bin  # This only yields the next bin in the same corridor (no wrapping)

            pvc_xy_num = pvc_xy_den_term1 = pvc_xy_den_term2 = 0
            for neuron in range(num_neurons):
                pvc_xy_num += session1[idx_x][neuron] * session2[idx_y][neuron]
                pvc_xy_den_term1 += session1[idx_x][neuron] * session1[idx_x][neuron]
                pvc_xy_den_term2 += session2[idx_y][neuron] * session2[idx_y][neuron]
            pvc_xy = pvc_xy_num / (np.sqrt(pvc_xy_den_term1 * pvc_xy_den_term2))
            pvc_vals.append(pvc_xy)

        if circular:
            pvc_mat[delta_bin] = pvc_vals
        else:
            # If not wrapping around, the matrix for all delta_bin > 0 do not fill the array completely
            pvc_mat[delta_bin, :len(pvc_vals)] = pvc_vals

        mean_pvc_delta_bin = np.mean(pvc_vals)
        stdev_delta_bin = np.std(pvc_vals)
        curve_yvals[delta_bin] = mean_pvc_delta_bin
        curve_stdev[delta_bin] = stdev_delta_bin

    if plot:
        plot_pvc_curve(curve_yvals, curve_stdev, show=True)

    return curve_yvals, curve_stdev, pvc_mat


def pvc_across_sessions(session1, session2, plot_heatmap=False, plot_in_ax=None, plot_zones=False,):
    """
    Compute PVC across sessions (see Shuman2020 Fig 3c).
    For each position bin X, compute population vector correlation of all neurons between two sessions.
    The resulting matrix has position bins of session1 on x-axis, and position bins of session2 on y-axis.
    Mean PVC curves are computed by comparing not the same position bin X between both sessions, but offsetting the
    position bin in one session by N centimeters, and average across all position-bin-pairs with the same offset.
    """

    # Filter out neurons that are not active in both cells
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask]
    session2 = session2[neuron_mask]
    logging.info(f'{np.sum(neuron_mask)} neurons available')

    ### Compute PVC across all positions between the two sessions
    pvc_matrix = np.zeros((session1.shape[1], session2.shape[1])) * np.nan
    for x in range(session1.shape[1]):
        for y in range(session2.shape[1]):
            # Multiply spatial maps of the same neuron of both sessions and sum across neurons (numerator)
            numerator = np.sum(session1[:, x] * session2[:, y])
            # Multiply spatial maps of the same neuron of both sessions with each other, sum across neurons, and multiply (denominator)
            denominator = np.sum(session1[:, x] * session1[:, x]) * np.sum(session2[:, y] * session2[:, y])

            # Compute PVC for this position combination
            pvc_matrix[x, y] = numerator / np.sqrt(denominator)

    if plot_heatmap:
        if plot_in_ax is not None:
            curr_ax = plot_in_ax
        else:
            plt.figure()
            curr_ax = plt.gca()
        sns.heatmap(pvc_matrix, ax=curr_ax, vmin=0, vmax=1, cbar=False, cmap='turbo')
        if plot_zones:
            zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
            for zone in zone_borders:
                curr_ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    return pvc_matrix


def draw_single_mouse_heatmaps(mouse_dict, day_diff=3, v_min=None, v_max=None, cmap='turbo', plot_last_cbar=True,
                               draw_zone_borders=True, verbose=False, only_return_matrix=False,
                               directory=None):
    """
    Draw PVC heatmaps across time (pre, pre-post, early, late). One figure per mouse.

    Args:
        mouse_dict: Dict with one entry (key is mouse_id). Value is a list with 2 elements -
            3D array with shape (n_neurons, n_sessions, n_bins) of spatial activity maps for a single mouse
            List of days relative to microsphere injection per session.
        day_diff: day difference at which to compare sessions.
        v_min: minimum scale of the heatmap. If both are set, dont draw color bar.
        v_max: maximum scale of the heatmap. If both are set, dont draw color bar.

    Returns:

    """

    if verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO, force=True)

    if v_min is not None and v_max is not None:
        plot_cbar = False
    else:
        plot_cbar = True

    if len(mouse_dict) != 1:
        raise KeyError('Only accepting single-key dicts.')

    mouse_id = list(mouse_dict.keys())[0]
    days = np.array(mouse_dict[mouse_id][1])
    last_pre_day = days[np.searchsorted(days, 0, side='right') - 1]  # Find last prestroke day
    spat_map = mouse_dict[mouse_id][0]

    prestroke = []
    pre_poststroke = []
    early_stroke = []
    late_stroke = []
    for day_idx, day in enumerate(days):
        next_day_idx = np.where(days == day + day_diff)[0]

        # After stroke, ignore small differences between sessions (do not have to be 3 days apart, sometimes 2 days)
        # In that case, use the next session irrespective of distance
        if (len(next_day_idx) == 0) and (1 < day < np.max(days)):
            next_day_idx = [day_idx + 1]

        # If a session 3 days later exists, compute the correlation of all cells between these sessions
        # Do not analyze session 1 day after stroke (unreliable data)
        if len(next_day_idx) == 1:
            curr_mat = pvc_across_sessions(session1=spat_map[:, day_idx],
                                           session2=spat_map[:, next_day_idx[0]],
                                           plot_heatmap=False)
            logging.info(f'Day {day} - next_day {days[next_day_idx[0]]}')
            if day < last_pre_day:
                prestroke.append(curr_mat)
                logging.info('\t-> Pre')
            elif day == last_pre_day:
                pre_poststroke.append(curr_mat)
                logging.info('\t-> Pre-Post')
            # elif last_pre_day < days[next_day_idx[0]] <= 6:
            elif last_pre_day < day < 6:
                early_stroke.append(curr_mat)
                logging.info('\t-> Early Post')
            elif 6 <= day:
                late_stroke.append(curr_mat)
                logging.info('\t-> Late Post')

    avg_prestroke = np.mean(np.stack(prestroke), axis=0)
    pre_poststroke = pre_poststroke[0]
    avg_early_stroke = np.mean(np.stack(early_stroke), axis=0)
    avg_late_stroke = np.mean(np.stack(late_stroke), axis=0)

    if only_return_matrix:
        return {'pre': avg_prestroke, 'pre_post': pre_poststroke, 'early': avg_early_stroke, 'late': avg_late_stroke}

    fig, axes = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(23, 6), layout='constrained')
    sns.heatmap(avg_prestroke, ax=axes[0], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(pre_poststroke, ax=axes[1], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(avg_early_stroke, ax=axes[2], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(avg_late_stroke, ax=axes[3], vmin=v_min, vmax=v_max, square=True, cbar=plot_last_cbar, cmap=cmap)

    # Make plot pretty
    titles = ['Prestroke', 'Pre-Post', 'Early Post', 'Late Post']
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Session 2')
    axes[0].set_ylabel('Session 1')

    if draw_zone_borders:
        zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
        for curr_ax in axes:
            for zone in zone_borders:
                # curr_ax.axvspan(zone[0], zone[1], facecolor='gray', alpha=0.4)
                curr_ax.axvline(zone[0], color='black', linestyle='--')
                curr_ax.axvline(zone[1], color='black', linestyle='--')

    fig.canvas.manager.set_window_title(mouse_id)

    if directory is not None:
        plt.savefig(os.path.join(directory, f'{mouse_id}.png'))
        plt.close()

    # Re-set logging level
    if verbose:
        logging.basicConfig(level=logging.WARNING, force=True)


def figure_plots(matrices, vmin=0, vmax=1, cmap='turbo', draw_zone_borders=True, directory=None):

    curves = []
    for row, (mouse_id, mouse_mats) in enumerate(matrices.items()):
        for col, (phase, mat) in enumerate(mouse_mats.items()):

            if directory is not None:

                # Plot PVC matrix
                plt.figure(layout='constrained', figsize=(12, 12), dpi=300)
                ax = sns.heatmap(mat, vmin=vmin, vmax=vmax, square=True, cbar=False, cmap=cmap)
                ax.set(xticks=[], yticks=[])

                plt.savefig(os.path.join(directory, f"{mouse_id}_{phase}.png"))

                if draw_zone_borders:
                    zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
                    # zone_borders = np.round(zone_borders)
                    for zone in zone_borders:
                        # curr_ax.axvspan(zone[0], zone[1], facecolor='gray', alpha=0.4)
                        ax.axvline(zone[0], color='black', linestyle='--')
                        ax.axvline(zone[1], color='black', linestyle='--')
                    plt.savefig(os.path.join(directory, f"{mouse_id}_{phase}_zones.png"))
                plt.close()

            # Query, compute and plot average PVC curve
            curve = np.nanmean(np.stack((hheise_pvc.PvcCrossSessionEval * hheise_pvc.PvcCrossSession &
                                         'locations="all"' & 'circular=0' & f'mouse_id={mouse_id}' &
                                        f'phase="{phase}"').fetch('pvc_curve')), axis=0)

            curves.append(pd.DataFrame(dict(mouse_id=mouse_id, phase=phase, pvc=curve,
                                            pos=np.linspace(5, 400, 80).astype(int))))
    return pd.concat(curves, ignore_index=True)

#%% PVC Heatmaps

spatial_maps = dc.load_data('decon_maps')
spatial_maps = dc.load_data('dff_maps')

####################################################################################################################
####### First example plot for a single mouse. For each network, compute PVC matrices between two sessions #########
fig, axes = plt.subplots(1, 4, sharex='all', sharey='all')
pvc_pre = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 1],
                              session2=spatial_maps[1]['121_1'][0][:, 4],
                              plot_heatmap=True, plot_in_ax=axes[0], plot_zones=True)
pvc_pre_post = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 4],
                                   session2=spatial_maps[1]['121_1'][0][:, 5],
                                   plot_heatmap=True, plot_in_ax=axes[1], plot_zones=True)
pvc_early = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 6],
                                session2=spatial_maps[1]['121_1'][0][:, 7],
                                plot_heatmap=True, plot_in_ax=axes[2], plot_zones=True)
pvc_late = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 10],
                               session2=spatial_maps[1]['121_1'][0][:, 11],
                               plot_heatmap=True, plot_in_ax=axes[3], plot_zones=True)

# Make plot pretty
titles = ['Prestroke (-3 -> 0)', 'Pre-Post (0 -> 3)', 'Early Post (6 -> 9)', 'Late Post (18 -> 21)']
for ax, title in zip(axes, titles):
    ax.set_title(title)
####################################################################################################################

### Take average of all pre, early and post pvc matrices
DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)
vmin = 0
vmax = 1

for mouse in spatial_maps:
    draw_single_mouse_heatmaps(mouse_dict=mouse, day_diff=DAY_DIFF, v_min=vmin, v_max=vmax, verbose=False,
                               directory=r"W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\PVC\avg_matrices")


# Make final overview for figure 3: One mouse per group + avg PVC curves
sham_mouse = 91
nodef_mouse = 114
rec_mouse = 90
norec_mouse = 41

plot_matrices = {91: draw_single_mouse_heatmaps(spatial_maps[9], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            114: draw_single_mouse_heatmaps(spatial_maps[16], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            90: draw_single_mouse_heatmaps(spatial_maps[8], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            41: draw_single_mouse_heatmaps(spatial_maps[1], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True)
            }
curves = figure_plots(plot_matrices, directory=r"W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\PVC\avg_matrices\figure3")

all_matrices = {int(list(dic.keys())[0].split('_')[0]): draw_single_mouse_heatmaps(dic, only_return_matrix=True) for dic in spatial_maps}
curves = figure_plots(all_matrices)

curves[curves.phase == 'late'].pivot(index='pos', columns='mouse_id', values='pvc').to_clipboard(index=False, header=False)

#%% Calculate and plot PVC curves

spatial_maps = dc.load_data('decon_maps')

DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)
days = np.array(spatial_maps[-1]['121_1'][1])
pre = []
pre_post = []
early = []
late = []

rz_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(80)
rz_borders[:, 0] = np.floor(rz_borders[:, 0])
rz_borders[:, 1] = np.ceil(rz_borders[:, 1])
rz_mask = np.zeros(80, dtype=bool)
for b in rz_borders.astype(int):
    rz_mask[b[0]:b[1]] = True

for day_idx, day in enumerate(days):
    next_day_idx = np.where(days == day + DAY_DIFF)[0]

    # If a session 3 days later exists, compute the correlation of all cells between these sessions
    # Do not analyze session 1 day after stroke (unreliable data)
    if day + DAY_DIFF != 1 and len(next_day_idx) == 1:

        curr_curve2, curr_std2, mat2 = pvc_curve(session1=spatial_maps[-1]['121_1'][0][:, day_idx],
                                              session2=spatial_maps[-1]['121_1'][0][:, next_day_idx[0]],
                                              plot=False, circular=False, max_delta_bins=30)

        curve_all = np.nanmean(mat, axis=1)
        curve_rz = np.nanmean(mat[:, rz_mask], axis=1)
        curve_nonrz = np.nanmean(mat[:, ~rz_mask], axis=1)

        if day < 0:
            pre.append(np.array([curr_curve, curr_std]))
            print('\r -> pre')
        elif day == 0:
            pre_post.append(np.array([curr_curve, curr_std]))
            print('\r -> pre-post')
        elif 0 < day < 9:
            early.append(np.array([curr_curve, curr_std]))
            print('\r -> early')
        elif 9 < day:
            late.append(np.array([curr_curve, curr_std]))
            print('\r -> late')

avg_pre = np.mean(np.stack(pre), axis=0)
avg_pre_post = pre_post[0]
avg_early = np.mean(np.stack(early), axis=0)
avg_late = np.mean(np.stack(late), axis=0)

## Plot example heatmaps of non-averaged PVC curves
fig, ax = plt.subplots(1, 2, sharey='all', sharex='all')
sns.heatmap(pvc_mat_upper, ax=ax[0], cbar=False)
sns.heatmap(pvc_mat_all, ax=ax[1], cbar=False)
for axis in ax:
    axis.set_aspect(1)
    axis.set_ylabel('$\Delta$X (diff between compared locations)')
    axis.set_xlabel('Corridor position bin')
ax[0].set_title('Corridor not wrapping around (straight)')
ax[1].set_title('Corridor wrapping around (circular)')

#%% DataJoint

metrics = ['min_slope', 'max_pvc', 'min_pvc', 'pvc_dif', 'pvc_rel_dif', 'slope_std', 'avg_prominence', 'avg_rel_prominence',
           'avg_slope', 'q1_diff', 'q1_rel_diff', 'q1_prominence', 'q1_rel_prominence', 'q1_slope', 'q2_diff',
           'q2_rel_diff', 'q2_prominence', 'q2_rel_prominence', 'q2_slope', 'q1_q2_diff', 'q1_q2_rel_diff',
           'q1_q2_prom_dif', 'q1_q2_rel_prom']
useful_metrics = ['max_pvc', 'min_pvc', 'min_slope', 'pvc_rel_dif', 'slope_std', 'q1_rel_diff', 'q1_rel_prominence',
                  'q2_rel_diff', 'q2_rel_prominence', 'q1_q2_diff', 'q1_q2_rel_diff', 'q1_q2_prom_dif']
data = pd.DataFrame((hheise_pvc.PvcCrossSessionEval() * hheise_pvc.PvcCrossSession & 'circular=0').fetch(as_dict=True))
# data = pd.DataFrame((hheise_pvc.PvcPrestrokeEval * hheise_pvc.PvcPrestroke & 'circular=0').fetch(as_dict=True))

coarse = (hheise_grouping.BehaviorGrouping & 'grouping_id = 0' & 'cluster = "coarse"').get_groups()
fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 0' & 'cluster = "fine"').get_groups()

data = data.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
data = data.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

data['day'] = pd.to_datetime(data.day)
data['slope_std'] = data['slope'].map(lambda x: np.nanstd(x))
data['q1_q2_diff'] = data.apply(lambda x:  (x['max_pvc'] + x['q2_diff']) - (x['max_pvc'] + x['q1_diff']), axis=1)
data['q1_q2_rel_diff'] = data.apply(lambda x:  ((x['max_pvc'] + x['q2_diff']) - (x['max_pvc'] + x['q1_diff'])) / (x['max_pvc'] + x['q1_diff']), axis=1)
data['q1_q2_prom_dif'] = data.apply(lambda x:  x['q2_prominence'] - x['q1_prominence'], axis=1)
data['q1_q2_rel_prom'] = data.apply(lambda x:  (x['q2_prominence'] - x['q1_prominence']) / x['q2_prominence'], axis=1)

#%% Plot example curve
plt.figure()
plt.plot(np.linspace(0, 400, 80)[:63],
         data[(data.mouse_id == 121) & (data.day == "2022-08-12") & (data.locations == 'all')]['pvc_curve'].values[0][:63])
plt.ylim(0)
plt.xlabel('$\Delta$X [cm] (location shift)')
plt.ylabel('PVC')

# Plot example curves for separate locations
quad_size = 64/3*5
half_win = 5*5
cols = ['green', 'grey']
fig, axes = plt.subplots(2, 1, sharey='all', sharex='all')
for i, loc in enumerate(['rz', 'non_rz']):
    axes[i].plot(np.linspace(0, 400, 80)[:55],
                 data[(data.mouse_id == 121) & (data.day == "2022-08-12") & (data.locations == loc)]['pvc_curve'].values[0][:55])
    axes[i].set_ylim(0)
    axes[i].set_ylabel('PVC')

    axes[i].axvspan(0, half_win, color=cols[i], alpha=0.25)
    axes[i].axvspan(quad_size - half_win, quad_size + half_win, color=cols[i], alpha=0.25)
    axes[i].axvspan(quad_size*2 - half_win, quad_size*2 + half_win, color=cols[i], alpha=0.25)

axes[-1].set_xlabel('$\Delta$X [cm] (location shift)')

#%% Correlate metrics with each other
corr_mat = data[metrics].corr()
sns.heatmap(corr_mat, mask=np.triu(np.ones_like(corr_mat, dtype=bool)), annot=True, center=0,
            xticklabels=corr_mat.columns, yticklabels=corr_mat.index, cmap='vlag')

#%% Test if standard deviation of derivative (slope) is a good measure for PVC "smoothness" --> yes, kind of
data = pd.DataFrame((hheise_pvc.PvcCrossSessionEval() * hheise_pvc.PvcCrossSession & 'circular=0').fetch('mouse_id', 'rel_day', 'pvc_curve', 'slope', as_dict=True))
data['slope_std'] = data['slope'].map(lambda x: np.nanstd(x)*100)

data_sort = data.sort_values('slope_std')


def plot_trace(row, a):
    a.plot(row['pvc_curve'])
    a.set_title(f'$\sigma$ = {row["slope_std"]:.3f}  (M{row["mouse_id"]}; day {row["rel_day"]})')
    a.spines[['right', 'top']].set_visible(False)


fig, axes = plt.subplots(7, 2, sharey='all', sharex='all', figsize=(12, 10))
for i in range(7):
    plot_trace(data_sort.iloc[i], axes[i, 0])
    plot_trace(data_sort.iloc[-(i+1)], axes[i, 1])


#%% Compare location-specific curves
data_melt = data[data.phase == 'pre'].melt(id_vars=['mouse_id', 'day', 'locations', 'coarse', 'fine'], value_vars=useful_metrics)
g = sns.FacetGrid(data_melt, col='variable', col_wrap=4)
g.map(sns.boxplot, 'locations', 'value', 'coarse')
g.add_legend()

# Prism export
mouse_avg = data_melt.groupby(['mouse_id', 'locations', 'variable'])['value'].mean().reset_index()
out = mouse_avg.pivot(index='variable', columns=['locations', 'mouse_id'], values='value')

out = data_melt.pivot(index='variable', columns=['locations', 'day', 'mouse_id'], values='value')
out['non_rz'].to_clipboard(header=False, index=False)

#%% Metrics for groups over time
avg = data[data.locations == 'all'].groupby(['mouse_id', 'phase'])[metrics].mean(numeric_only=True)
avg = avg.join(data[data.locations == 'all'][['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()

avg_prism = avg.pivot(index='phase', columns='mouse_id', values='avg_rel_prominence').loc[['pre', 'pre_post', 'early', 'late']]
avg_prism.to_clipboard(header=False, index=False)

# Normalize against prestroke baseline
norm_data = []
for mouse_id, mouse in data.groupby(['mouse_id', 'locations']):
    mouse_norm = pd.DataFrame({metric+'_norm': mouse[metric] / mouse[mouse.phase == 'pre'][metric].mean() for metric in useful_metrics})
    norm_data.append(mouse.join(mouse_norm))
norm_data = pd.concat(norm_data)
avg_norm = norm_data[norm_data.locations == 'all'].groupby(['mouse_id', 'phase']).mean(numeric_only=True)
avg_norm = avg_norm.join(norm_data[norm_data.locations == 'all'][['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()

avg_norm_prism = avg_norm.pivot(index='phase', columns='mouse_id', values='q1_q2_prom_dif_norm').loc[['pre', 'pre_post', 'early', 'late']]
avg_norm_prism.to_clipboard(header=False, index=False)

# Plot average PVC curves per period between groups
avg_curves = data.groupby(['mouse_id', 'locations', 'phase'])['pvc_curve'].mean().reset_index()
avg_curves = avg_curves.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
avg_curves = avg_curves.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})
avg_curves['$\Delta$x'] = [np.linspace(0, 395, 80)]*len(avg_curves)

avg_curves_melt = avg_curves.explode(['pvc_curve', '$\Delta$x'])
avg_curves_melt['$\Delta$x'] = avg_curves_melt['$\Delta$x'].astype(int)
with sns.plotting_context(context='notebook', font_scale=1.5):
    g = sns.FacetGrid(avg_curves_melt[avg_curves_melt['$\Delta$x'] <= 250], row='locations', col='phase',
                      # hue='coarse', hue_order=['Control', 'Stroke'], palette={'Control': 'green', 'Stroke': 'red'},
                      hue='fine', hue_order=['Sham', 'No Deficit', 'Recovery', 'No Recovery'], palette={'Sham': 'grey', 'No Deficit': 'green', 'Recovery': 'blue', 'No Recovery': 'red'},
                      col_order=['pre', 'pre_post', 'early', 'late'], row_order=['all', 'rz', 'non_rz'],
                      margin_titles=True)
    g.map(sns.lineplot, '$\Delta$x', 'pvc_curve')
    # g.axes[-1][-1].legend()
    g.add_legend()

# Export avg curves to Prism
data_prism = avg_curves_melt[(avg_curves_melt.locations == 'non_rz') & (avg_curves_melt.phase == 'pre') &
                             (avg_curves_melt['$\Delta$x'] <= 250)].pivot(index='$\Delta$x', columns='mouse_id', values='pvc_curve')
data_prism.to_clipboard(header=False, index=False)


#%% Test if there is improvement across sessions in healthy condition --> learning effect?

# add column for prestroke session num
dfs = []
for m_id, mouse in data.groupby('mouse_id'):
    unique_days = np.sort(np.unique(np.concatenate([mouse['rel_day'], mouse['rel_tar_day']])))
    rel_sessions = np.arange(-len(unique_days)+1, 1)
    mapper = pd.DataFrame({'rel_day': unique_days, 'rel_sess': rel_sessions})
    dfs.append(pd.merge(mouse, mapper, on='rel_day'))
data = pd.concat(dfs, ignore_index=True)

data_prism = data[data.locations == 'all'].pivot(index='rel_sess', columns='mouse_id', values='pvc_rel_dif')
data_prism.to_clipboard(header=False, index=False)


#%% RZ vs non-RZ locations
avg = data.groupby(['mouse_id', 'locations', 'phase'])[metrics].mean(numeric_only=True)
avg = avg.join(data[['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()

metric = 'q1_rel_prominence'
avg_loc_rz = avg[avg.locations == 'rz'].pivot(index='phase', columns='mouse_id', values=metric).loc[['pre', 'pre_post', 'early', 'late']]
avg_loc_nonrz = avg[avg.locations == 'non_rz'].pivot(index='phase', columns='mouse_id', values=metric).loc[['pre', 'pre_post', 'early', 'late']]
avg_loc_rz.join(avg_loc_nonrz, lsuffix='_rz', rsuffix='_nonrz').to_clipboard(header=False, index=False)

# Metric differences (RZ - non_RZ)
rz_diff_data = []
for idx, df in data.groupby(['mouse_id', 'day']):
    pks = df[['mouse_id', 'day', 'phase', 'rel_day', 'rel_tar_day', 'coarse', 'fine']].iloc[0]
    diff = df[df.locations == 'rz'][metrics].iloc[0] - df[df.locations == 'non_rz'][metrics].iloc[0]
    rz_diff_data.append(pd.DataFrame([dict(**pks, **diff)]))
rz_diff_data = pd.concat(rz_diff_data, ignore_index=True)

avg = rz_diff_data.groupby(['mouse_id', 'phase'])[metrics].mean(numeric_only=True)
avg = avg.join(rz_diff_data[['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()
avg_prism = avg.pivot(index='phase', columns='mouse_id', values='slope_std').loc[['pre', 'pre_post', 'early', 'late']]
avg_prism.to_clipboard(header=False, index=False)


rz_rel_diff_data = []
for idx, df in data.groupby(['mouse_id', 'day']):
    pks = df[['mouse_id', 'day', 'phase', 'rel_day', 'rel_tar_day', 'coarse', 'fine']].iloc[0]
    diff = (df[df.locations == 'rz'][metrics].iloc[0] - df[df.locations == 'non_rz'][metrics].iloc[0]) / df[df.locations == 'non_rz'][metrics].iloc[0]
    rz_rel_diff_data.append(pd.DataFrame([dict(**pks, **diff)]))
rz_rel_diff_data = pd.concat(rz_rel_diff_data, ignore_index=True)

# One-sample t-tests
avg_ttest = avg[['mouse_id', 'phase', *metrics]].melt(id_vars=['mouse_id', 'phase'])
big_pivot = avg_ttest.pivot(index='mouse_id', columns=['phase', 'variable'], values='value')
big_pivot = big_pivot.reindex(columns=big_pivot.columns.reindex(['pre', 'pre_post', 'early', 'late'], level=0)[0])
big_pivot.columns = [' '.join(col).strip() for col in big_pivot.columns.values]
big_pivot.to_clipboard(header=True, index=False)

#%%
plt.figure()
sns.barplot(avg, x='phase', y='min_pvc', hue='coarse', order=['pre', 'pre_post', 'early', 'late'])


