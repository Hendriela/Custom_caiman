#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 04/07/2022 15:40
@author: hheise

Correlation of performance and neural activity parameters with DataJoint.
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats

from schema import common_img, hheise_placecell, hheise_behav, common_mice
from util import helper


# Select mice that should be part of the analysis (only mice that had successful microsphere surgery)
mice = [33, 38, 41, 83, 85, 86, 89, 90, 91, 93, 94, 95]

# Get data from each mouse separately
dfs = []
for mouse in mice:
    # get date of microsphere injection
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch1(
        'surgery_date')

    # Get the dates of 5 imaging sessions before the surgery, and all dates from 2 days after it
    pre_dates = (common_img.Scan & f'mouse_id={mouse}' & f'day < "{surgery_day.date()}"').fetch('day')[-5:]
    post_dates = (common_img.Scan & f'mouse_id={mouse}' & f'day > "{surgery_day.date()+timedelta(days=1)}"').fetch('day')
    dates = [*pre_dates, *post_dates]
    is_pre = [*[True]*len(pre_dates), *[False]*len(post_dates)]

    # Get average performance for each session
    perf = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day in {helper.in_query(dates)}').get_mean()

    # Get mean and median firing rate
    curr_rate = [(common_img.ActivityStatistics.ROI & f'mouse_id={mouse}' & f'day="{day}"').fetch('rate_spikes')
                 for day in dates]
    mean_fr = [np.mean(x) for x in curr_rate]
    median_fr = [np.median(x) for x in curr_rate]

    # Get place cell ratio (Bartos and SI criteria)
    pc_bartos = (hheise_placecell.PlaceCell & f'mouse_id={mouse}' & 'corridor_type=0' &
                 f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')
    pc_si = (hheise_placecell.SpatialInformation & f'mouse_id={mouse}' & 'corridor_type=0' &
             f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')

    # Get within-session stability
    curr_rate = [(hheise_placecell.SpatialInformation.ROI & f'mouse_id={mouse}' & f'day="{day}"').fetch('stability')
                 for day in dates]
    mean_stab = [np.mean(x) for x in curr_rate]
    median_stab = [np.median(x) for x in curr_rate]
    sd_stab = [np.std(x) for x in curr_rate]

    # Build DataFrame
    date_diff = [(date - surgery_day.date()).days for date in dates]
    data = dict(mouse_id=[mouse] * len(dates), date=dates, is_pre=is_pre, from_surg=date_diff, performance=perf, mean_fr=mean_fr,
                median_fr=median_fr, pc_bartos=pc_bartos, pc_si=pc_si, mean_stab=mean_stab, median_stab=median_stab,
                sd_stab=sd_stab)
    dfs.append(pd.DataFrame(data))

data = pd.concat(dfs)

# Plot correlations
# for metric in ['mean_fr', 'median_fr', 'pc_bartos', 'pc_si']:
for metric in ['mean_stab', 'pc_si']:
    sn.lmplot(data=data, x='performance', y=metric)
    rho, p = stats.spearmanr(data['performance'], data[metric])
    plt.text(0.12, 0.8, 'rho: {:.3f}\np: {:.5f}'.format(rho, p), transform=plt.gca().transAxes)

# Metrics that correlate well with performance: pc_si, stability (all 3)
# Export for Prism
data.to_csv(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\perf_neur_corr.csv', sep='\t', index=False,
            columns=['mouse_id', 'performance', 'pc_si', 'mean_stab'])

#%% Add performance grouping to histology data

no_deficit = [91, 94, 95]
recovery = [33, 38, 83, 85, 86, 89, 90]
no_recovery = [41, 63, 69, 93]
mice = [*no_deficit, *recovery, *no_recovery]
grouping = pd.DataFrame(data=dict(mouse_id=mice, group=[*['no_deficit']*len(no_deficit), *['recovery']*len(recovery),
                                                        *['no_recovery']*len(no_recovery)]))
data_grouped = data.merge(grouping)

# Plot correlations
for metric in ['mean_stab', 'pc_si']:
    sn.lmplot(data=data_grouped, x='performance', y=metric, hue='group')
    rho, p = stats.spearmanr(data_grouped['performance'], data_grouped[metric])
    plt.text(0.12, 0.8, 'rho: {:.3f}\np: {:.5f}'.format(rho, p), transform=plt.gca().transAxes)

# Plot averages
sn.boxplot(x='group', y='mean_stab', data=data_grouped, hue='is_pre', hue_order=[True, False])     # Mice in the recovery group seem to have less place cells?

# Export for Prism
export = pd.DataFrame({'no_deficit': data_grouped[data_grouped['group'] == 'no_deficit']['pc_si'],
                       'recovery': data_grouped[data_grouped['group'] == 'recovery']['pc_si'],
                       'no_recovery': data_grouped[data_grouped['group'] == 'no_recovery']['pc_si']})
export = export.apply(lambda x: pd.Series(x.dropna().values))
export.to_csv(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\pc_si_grouped.csv', sep='\t', index=False)

# Export with time info (separate into groups of prestroke, <7d poststroke, >7d poststroke)
data_grouped['time'] = 'none'
data_grouped.loc[data_grouped['is_pre']==True, 'time'] = 'pre'
data_grouped.loc[(data_grouped['from_surg']<8) & (data_grouped['from_surg']>0), 'time'] = '< 7d'
data_grouped.loc[data_grouped['from_surg']>7, 'time'] = '> 7d'

sn.boxplot(x='group', y='pc_si', data=data_grouped, hue='time', hue_order=['pre', '< 7d', '> 7d'])

for timestep in ['pre', '< 7d', '> 7d']:
    curr_timestep = []
    for group in ['no_deficit', 'recovery', 'no_recovery']:
        curr_timestep.append(data_grouped[(data_grouped['group'] == group) & (data_grouped['time'] == timestep)]
                          ['pc_si'].reset_index().drop(columns='index').rename(columns={'pc_si': group}))    # Reset and drop index to make later concat easier

    curr_df = pd.concat(curr_timestep, axis=1).T
    if timestep == '< 7d':
        timestep = '-7d'
    elif timestep == '> 7d':
        timestep = '+7d'
    curr_df.to_csv(f'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Posters\\FENS 2022\\pc_si_grouped_time\\{timestep[-3:]}.csv', sep='\t', index=False)

# Time-dependent PC-SI (not too meaningful)
sn.lineplot(data=data_grouped, x='from_surg', y='pc_si', hue='group')



