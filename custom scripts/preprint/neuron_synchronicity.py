#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 07/02/2024 16:49
@author: hheise

"""
import pandas as pd
import numpy as np
import seaborn as sns

from schema import hheise_connectivity, hheise_hist, common_mice, hheise_behav, hheise_grouping
from util import helper
from preprint import data_cleaning as dc

mice = [33, 41,  # Batch 3
        63, 69,  # Batch 5
        83, 85, 86, 89, 90, 91, 93, 95,  # Batch 7
        108, 110, 111, 113, 114, 115, 116, 122]  # Batch 8

#%% Load data

spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"' & 'username="hheise"' &
                        f'mouse_id in {helper.in_query(mice)}').proj(spheres='count_extrap').fetch('mouse_id', 'spheres', as_dict=True))
injection = pd.DataFrame((common_mice.Surgery & 'username="hheise"' & f'mouse_id in {helper.in_query(mice)}' &
                          'surgery_type="Microsphere injection"').fetch('mouse_id', 'surgery_date',as_dict=True))
injection.surgery_date = injection.surgery_date.dt.date
vr_performance = pd.DataFrame((hheise_behav.VRPerformance & f'mouse_id in {helper.in_query(mice)}' &
                               'perf_param_id=0').fetch('mouse_id', 'day', 'si_binned_run', as_dict=True))

group = (hheise_grouping.BehaviorGrouping() & 'cluster="fine"' & 'grouping_id=4').get_groups()
group.loc[group.group == "No Deficit", 'group'] = "Sham"

#%%
metrics = ['skewness', 'perc99',  'avg_corr', 'median_corr', 'avg_corr_pc', 'median_corr_pc']
conn_data = pd.DataFrame((hheise_connectivity.NeuronNeuronCorrelation & 'trace_type="dff"' &
                          f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'day', *metrics, as_dict=True))
conn = dc.merge_dfs(df=conn_data, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

def average_corrcoef(arr):
    return np.tanh(np.nanmean(np.arctanh(arr)))

# Compute mouse averages across phases
avg_conn = conn.groupby(['mouse_id', 'phase']).agg({'skewness': np.nanmean, 'perc99': average_corrcoef,
                                                    'avg_corr': average_corrcoef, 'median_corr': average_corrcoef,
                                                    'avg_corr_pc': average_corrcoef, 'median_corr_pc': average_corrcoef})
avg_conn = avg_conn.join(group.set_index('mouse_id'), on='mouse_id').reset_index()

# Plot data
# sns.boxplot(avg_conn, x='phase', y='avg_corr', hue='group', order=['pre', 'early', 'late'], hue_order=['Sham','Recovery','No Recovery'])
avg_conn_melt = avg_conn.melt(id_vars=['mouse_id', 'phase', 'group'], var_name='stat')
g = sns.catplot(data=avg_conn_melt, x='phase', y='value', hue='group', col='stat', kind='box', col_wrap=3, sharey=False,
                hue_order=['Sham', 'Recovery', 'No Recovery'], order=['pre', 'early', 'late'])
g.map(sns.stripplot, 'phase', 'value', 'group', hue_order=['Sham', 'Recovery', 'No Recovery'], order=['pre', 'early', 'late'],
      palette=sns.color_palette()[:3], dodge=True, alpha=1, ec='k', linewidth=1)
for ax in g.axes[2:]:
    ax.axhline(0, color='r', linestyle='--')

#%% Export for prism

avg_conn.pivot(index='phase', columns='mouse_id', values='skewness').loc[['pre', 'early', 'late']].to_clipboard(index=False, header=False)

