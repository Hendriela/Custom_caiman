#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 02/08/2024 12:51
@author: hheise

Script to analyse place fields of highly correlating pc-pc pairs from percentiles_nc_pc_pf_locations.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from schema import hheise_behav, hheise_grouping

# Load data
unique_fields = pd.read_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_unique_fields.pkl')
pairwise_fields = pd.read_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_pairwise_fields.pkl')
unique_fields_ctrl = pd.read_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_unique_fields_control.pkl')
pairwise_fields_ctrl = pd.read_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\place_field_locations\95th_perc_pc-pc_pairwise_fields_control.pkl')

fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
zones = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)

#%%  Get distribution of unique place fields within place cells that are part of a highly correlating pc-pc pair

# Reshape data
df_melt = unique_fields_ctrl.melt(var_name='day', ignore_index=False, value_name='coord').reset_index(names='mouse_id').explode('coord', ignore_index=True).dropna(axis='index')
df_melt = df_melt[df_melt.day != 1]
df_melt['period'] = 'early'
df_melt.loc[df_melt.day <= 0, 'period'] = 'pre'
df_melt.loc[df_melt.day > 7, 'period'] = 'late'
df_melt = df_melt.merge(fine, how='left', on='mouse_id')
df_melt.loc[df_melt.group == 'No Deficit', 'group'] = 'Sham'
df_melt = df_melt.sort_values(['day', 'mouse_id'])

# Transform place field coordinates to distance from last reward zone
def get_distance(centers, borders):
    """ Get distance in bin coordinates from last RZ end. """
    def comp_dist(cent, b):
        dist = cent - b
        return np.min(dist[dist > 0])
    try:
        return [comp_dist(c, borders) for c in np.array(centers)]
    except TypeError:
        return comp_dist(centers, borders)

end2end = zones[1, 1]-zones[0, 1]
start2end = zones[1, 1]-zones[1, 0]
df_melt['dist_from_rz_end'] = df_melt['coord'].apply(get_distance, borders=np.append(zones[0, 1]-end2end, zones[:, 1]))

# Filter out border cells (place cells with field center within 10 cm of start/end of the corridor), and mouse 89
df_melt_filt = df_melt[(df_melt.coord > 2) & (df_melt.coord < 77) & (df_melt.mouse_id != 89)]


# Plot data of groups for each period as histograms (distribution of place fields across corridor)
color_dict = {'Sham': (149/255, 152/255, 156/255), 'Recovery': (100/255, 152/255, 208/255), 'No Recovery': (238/255, 125/255, 128/255)}
color_dict = {'Sham': 'grey', 'Recovery': 'blue', 'No Recovery': 'red'}

fig, axes = plt.subplots(nrows=3, sharex='all', layout='constrained')
for ax, period in zip(axes, ['pre', 'early', 'late']):
    for z in zones:
        ax.axvspan(z[0], z[1], color='green', alpha=0.2)

    sns.histplot(data=df_melt[(df_melt.period == period) & (df_melt.mouse_id != 89)], x='coord', bins=80, binrange=(0, 80), hue='group', stat='percent',
                 common_norm=False, element='step', palette=color_dict, discrete=False, ax=ax)
    ax.set_ylabel(period)
    if period != 'pre':
        ax.get_legend().remove()

# Separate groups
g = sns.displot(data=df_melt_filt, x='coord', bins=80, binrange=(0, 80), col='period', row='group', stat='percent',
                common_norm=False, element='step', discrete=False, height=2, aspect=3,
                facet_kws={'margin_titles': True, 'sharey': True},
                )
for ax in g.axes.flatten():
    for z in zones:
        ax.axvspan(z[0], z[1], color='green', alpha=0.2)


# Plot distance from last reward zone
fig, axes = plt.subplots(nrows=3, sharex='all', layout='constrained')
for ax, period in zip(axes, ['pre', 'early', 'late']):
    ax.axvspan(end2end-start2end, end2end, color='green', alpha=0.2)
    sns.histplot(data=df_melt_filt[df_melt_filt.period == period], x='dist_from_rz_end', bins=22, hue='group', stat='percent',
                 common_norm=False, element='step', palette=color_dict, ax=ax, # multiple='stack',
                 hue_order=['No Recovery', 'Recovery', 'Sham'])
    ax.set_ylabel(period)
    if period != 'pre':
        ax.get_legend().remove()

# Plot all Recovery mice singularly, maybe the trend is determined by a few mice
rec_mice = np.unique(df_melt_filt.loc[df_melt_filt.group == 'Sham', 'mouse_id'])
fig, axes = plt.subplots(nrows=len(rec_mice), ncols=3, sharex='all', layout='constrained', sharey='row')
for ax_row, mouse in zip(axes, rec_mice):
    for ax, period in zip(ax_row, ['pre', 'early', 'late']):
        ax.axvspan(end2end-start2end, end2end, color='green', alpha=0.2)
        sns.histplot(data=df_melt_filt[(df_melt_filt.period == period) & (df_melt_filt.mouse_id == mouse)], x='dist_from_rz_end', bins=22, stat='percent',
                     common_norm=False, element='step', ax=ax, # multiple='stack',
                     )
        if period == 'pre':
            ax.set_ylabel(mouse)
        if mouse == rec_mice[0]:
            ax.set_title(period)


#%% Pairwise place fields
color_dict = {'Sham': 'grey', 'Recovery': 'cornflowerblue', 'No Recovery': 'salmon'}
color_dict = {'control': 'grey', 'top 5%': 'lawngreen'}
color_dict = {'top 5%': {'Sham': 'grey', 'Recovery': 'cornflowerblue', 'No Recovery': 'salmon'},
              'control': {'Sham': 'lightgrey', 'Recovery': 'powderblue', 'No Recovery': 'mistyrose'}}

# Reshape data
df_melt = pairwise_fields.melt(var_name='day', ignore_index=False, value_name='coord').reset_index(names='mouse_id').explode('coord', ignore_index=True).dropna(axis='index')
df_melt['dataset'] = 'top 5%'
df_melt_ctrl = pairwise_fields_ctrl.melt(var_name='day', ignore_index=False, value_name='coord').reset_index(names='mouse_id').explode('coord', ignore_index=True).dropna(axis='index')
df_melt_ctrl['dataset'] = 'control'
df_melt = pd.concat([df_melt, df_melt_ctrl])
df_melt[['coord_a', 'coord_b']] = pd.DataFrame(df_melt['coord'].tolist(), index=df_melt.index)
df_melt = df_melt.drop(columns=['coord'])
df_melt = df_melt[df_melt.day != 1]
df_melt['period'] = 'early'
df_melt.loc[df_melt.day <= 0, 'period'] = 'pre'
df_melt.loc[df_melt.day > 7, 'period'] = 'late'
df_melt = df_melt.merge(fine, how='left', on='mouse_id')
df_melt.loc[df_melt.group == 'No Deficit', 'group'] = 'Sham'
df_melt = df_melt.sort_values(['day', 'mouse_id'])


def circularize_quadrants(positions, quadrant_size=64 / 3):
    """
    Transform an array of position bins into a version where each quadrant is circularized, reflecting
    the periodicity of the corridor.

    Args:
        positions: 1D numpy array with shape (n_frames) in standard corridor coordinates. Default for training corridor.
        quadrant_size:  Size of each quadrant in standard corridor coordinates

    Returns:
        1D array with same shape as 'positions', transformed into circular quadrant coordinates
    """

    # Rescale positions to a single quadrant and take cosine to map it to a circle (one circle/period per quadrant)
    pos_cos = np.cos(positions / quadrant_size * 2 * np.pi)

    # Rescale the circular positions to corridor coordinates (peak distance is 10 cm (half quadrant size)
    pos_quad = np.arccos(pos_cos) * quadrant_size / np.pi / 2

    return pos_quad


# Circularize locations and compute absolute (quadrant-insensitive) distance between place fields
df_melt['circ_a'] = circularize_quadrants(df_melt['coord_a'])
df_melt['circ_b'] = circularize_quadrants(df_melt['coord_b'])
df_melt['circ_dist'] = np.abs(df_melt['circ_a'] - df_melt['circ_b'])
df_melt['dist'] = np.abs(df_melt['coord_a'] - df_melt['coord_b'])

# Plot distributions
sns.catplot(data=df_melt, x='group', y='circ_dist', col='period', kind='violin', hue='dataset', palette=color_dict, cut=0, split=True, bw_adjust=.5)
sns.catplot(data=df_melt, x='group', y='dist', col='period', kind='violin', hue='dataset', palette=color_dict, cut=0, split=True, bw_adjust=.5)



