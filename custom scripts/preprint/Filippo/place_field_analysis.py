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
unique_fields = pd.read_pickle(r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations\95th_perc_pc-pc_unique_fields.pkl')

fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
zones = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)

#%%  Get distribution of unique place fields within place cells that are part of a highly correlating pc-pc pair

# Reshape data
df_melt = unique_fields.melt(var_name='day', ignore_index=False, value_name='coord').reset_index(names='mouse_id').explode('coord', ignore_index=True).dropna(axis='index')
df_melt = df_melt[df_melt.day != 1]
df_melt['period'] = 'early'
df_melt.loc[df_melt.day <= 0, 'period'] = 'pre'
df_melt.loc[df_melt.day > 7, 'period'] = 'late'
df_melt = df_melt.merge(fine, how='left', on='mouse_id')
df_melt.loc[df_melt.group == 'No Deficit', 'group'] = 'Sham'
df_melt = df_melt.sort_values(['day', 'mouse_id'])

# Plot data of groups for each period as histograms (distribution of place fields across corridor)
color_dict = {'Sham': (149/255, 152/255, 156/255), 'Recovery': (100/255, 152/255, 208/255), 'No Recovery': (238/255, 125/255, 128/255)}
color_dict = {'Sham': 'grey', 'Recovery': 'blue', 'No Recovery': 'red'}

fig, axes = plt.subplots(nrows=3, sharex='all', layout='constrained')
for ax, period in zip(axes, ['pre', 'early', 'late']):
    for z in zones:
        ax.axvspan(z[0], z[1], color='green', alpha=0.2)

    sns.histplot(data=df_melt[df_melt.period == period], x='coord', bins=81, hue='group', stat='percent',
                 common_norm=False, element='step', palette=color_dict, ax=ax)
    ax.set_ylabel(period)
    if period != 'pre':
        ax.get_legend().remove()



