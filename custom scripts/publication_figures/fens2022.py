#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 05/07/2022 22:05
@author: hheise

"""

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import os

from schema import common_img, hheise_placecell, hheise_behav, common_mice, hheise_hist
from hheise_scripts import hheise_util
from util import helper

# Group mice by their task performance response to microspheres
no_deficit = [69, 91, 94, 95]
recovery = [33, 38, 83, 85, 86, 89, 90]
no_recovery = [41, 93]
mice = [*no_deficit, *recovery, *no_recovery]

# With Batch 8:
no_deficit = [93, 91, 94, 95, 109, 123, 120]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 38, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]

#%% MANUALLY CHECK LICKING BEHAVIOR FOR CLASSIFICATION
dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\analysis\behavior\manual_screening_for_grouping'

for mouse in mice:
    dates, is_pre = hheise_util.get_microsphere_sessions(mouse, pre_sessions=6)
    (hheise_behav.VRSession & f'mouse_id={mouse}' & f'day in {helper.in_query(dates)}').plot_lick_histogram()
    plt.savefig(os.path.join(dir, f'M{mouse}.png'))
    plt.close()


#%% FIGURE 2: PERFORMANCE-GROUPED HISTOLOGY ANALYSIS

grouping = pd.DataFrame(data=dict(mouse_id=mice,
                              group=[*['no_deficit']*len(no_deficit), *['recovery']*len(recovery),
                                     *['no_recovery']*len(no_recovery)]))

### Collect metrics
metrics = ['spheres', 'spheres_lesion', 'auto', 'auto_spheres']

# Simple sphere/lesion counts
data = grouping
for metric in metrics:
    curr_metric = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & f'mouse_id in {helper.in_query(mice)}' &
                           f'metric_name="{metric}"').fetch('mouse_id', 'metric_name', 'count', 'count_extrap', as_dict=True))
    curr_metric.columns=[curr_metric.columns[0], curr_metric.columns[1],
                         curr_metric['metric_name'][0]+'_'+curr_metric.columns[2],
                         curr_metric['metric_name'][0]+'_'+curr_metric.columns[3]]
    data = data.merge(curr_metric.drop(columns='metric_name'), on='mouse_id')

all_metrics = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & f'mouse_id in {helper.in_query(mice)}' &
                            f'metric_name in {helper.in_query(metrics)}').fetch('mouse_id', 'metric_name', 'count', 'count_extrap', as_dict=True))
total = all_metrics.merge(grouping)

# Plot data
for col in data.columns[2:]:
    plt.figure()
    sn.boxplot(x='group', y=col, data=data)
    sn.stripplot(x='group', y=col, data=data, hue='mouse_id', palette=sn.color_palette("hls", 14))

# in a facet grid
g = sn.catplot(x="group", y="count_extrap", col="metric_name", kind='box', sharey=False, data=total, fliersize=0)
g.map_dataframe(sn.stripplot, x="group", y="count_extrap", dodge=True, color='black')

# Export for Prism
group_arr = np.zeros((2, 7*3)) * np.nan
header = ['']*7*3

for group_idx, curr_group in enumerate([no_deficit, recovery, no_recovery]):
    for idx, mouse in enumerate(curr_group):
        group_arr[0, idx+(group_idx*7)] = total[(total['mouse_id'] == mouse) & (total['metric_name'] == 'spheres')]['count_extrap']
        group_arr[1, idx+(group_idx*7)] = total[(total['mouse_id'] == mouse) & (total['metric_name'] == 'auto')]['count_extrap']
        header[idx+(group_idx*7)] = str(mouse)

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\histo_performance_group.csv', group_arr,
           delimiter='\t', header='\t'.join(header))


#%% FIGURE 4: NEURAL-PERFORMANCE-CORRELATIONS #####

# Select mice that should be part of the analysis (only mice that had successful microsphere surgery)
mice = [33, 38, 41, 69, 83, 85, 86, 89, 90, 91, 93, 94, 95]

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
    # pc_bartos = (hheise_placecell.PlaceCell & f'mouse_id={mouse}' & 'corridor_type=0' &
    #              f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')
    pc_si = (hheise_placecell.SpatialInformation & f'mouse_id={mouse}' & 'corridor_type=0' & f'place_cell_id=0' &
             f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')

    # Get within-session stability
    curr_rate = [(hheise_placecell.SpatialInformation.ROI & f'mouse_id={mouse}' & f'day="{day}"').fetch('stability')
                 for day in dates]
    mean_stab = [np.mean(x) for x in curr_rate]
    median_stab = [np.median(x) for x in curr_rate]
    sd_stab = [np.std(x) for x in curr_rate]

    # Build DataFrame
    data = dict(mouse_id=[mouse] * len(dates), date=dates, is_pre=is_pre, performance=perf, mean_fr=mean_fr,
                median_fr=median_fr, pc_si=pc_si, mean_stab=mean_stab, median_stab=median_stab,
                sd_stab=sd_stab,
                # pc_bartos=pc_bartos
                )
    dfs.append(pd.DataFrame(data))

data = pd.concat(dfs)

data_group = data.merge(grouping)

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

import seaborn as sns
sns.stripplot(data=data_group, x='group', y='pc_si', order=['no_deficit', 'recovery', 'no_recovery'], hue='mouse_id')

