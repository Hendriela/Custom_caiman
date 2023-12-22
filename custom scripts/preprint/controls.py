#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 24/10/2023 13:48
@author: hheise

Various control analyses
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

from schema import hheise_placecell, hheise_grouping, common_mice, hheise_behav, hheise_decoder, hheise_pvc, hheise_connectivity

#%% Correlate non-normalized SI performance with neural data (decoder, PVC, neuron-neuron correlation, fraction of PCs,
# within-session stability of cells/place cells)

mice = np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id'))

dfs = []
for mouse in mice:

    # Get prestroke session PKs
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' &
                   f'mouse_id={mouse}').fetch('surgery_date')[0].date()

    session_pks = (hheise_placecell.PlaceCell & f'mouse_id={mouse}' & f'day<="{surgery_day}"' & 'place_cell_id=2' &
                   'corridor_type=0').fetch('KEY')

    # Fetch data
    si_perf = (hheise_behav.VRPerformance & session_pks).fetch('si_binned')
    decoder_acc, decoder_mae, decoder_sens = (hheise_decoder.BayesianDecoderWithinSession & session_pks &
                                              'bayesian_id=1').fetch('accuracy', 'mae_quad', 'sensitivity_rz')
    pc_frac = (hheise_placecell.PlaceCell & session_pks).fetch('place_cell_ratio')
    within_stab_all = [np.nanmean((hheise_placecell.SpatialInformation.ROI & sess).fetch('stability')) for sess in session_pks]
    within_stab_pc = [np.nanmean((hheise_placecell.SpatialInformation.ROI * hheise_placecell.PlaceCell.ROI & sess &
                                  'is_place_cell=1').fetch('stability')) for sess in session_pks]
    mean_corr, mean_corr_pc, median_corr, median_corr_pc = (hheise_connectivity.NeuronNeuronCorrelation & session_pks & 'trace_type="dff"').fetch('avg_corr', 'avg_corr_pc', 'median_corr', 'median_corr_pc')
    mean_corr_spat, mean_corr_pc_spat, median_corr_spat, median_corr_pc_spat = (hheise_connectivity.NeuronNeuronCorrelation & session_pks & 'trace_type="spat_dff"').fetch('avg_corr', 'avg_corr_pc', 'median_corr', 'median_corr_pc')

    curr_df = pd.DataFrame(dict(mouse_id=mouse, day=[x['day'] for x in session_pks], si_perf=si_perf, pc_frac=pc_frac,
                                decoder_acc=decoder_acc, decoder_mae=decoder_mae, decoder_sens=decoder_sens,
                                within_stab_all=within_stab_all, within_stab_pc=within_stab_pc,
                                mean_corr=mean_corr, mean_corr_pc=mean_corr_pc, mean_corr_spat=mean_corr_spat, mean_corr_pc_spat=mean_corr_pc_spat,
                                # median_corr=median_corr, median_corr_pc=median_corr_pc, median_corr_spat=median_corr_spat, median_corr_pc_spat=median_corr_pc_spat
                                ))

    # PVC does not have data for every session, has to be added afterwards and merged by day
    try:
        pvc_df = pd.DataFrame((hheise_pvc.PvcCrossSessionEval & session_pks & 'circular=0' &
                               'locations="all"').fetch('day', 'min_slope', 'max_pvc', as_dict=True))
        curr_df = pd.merge(curr_df, pvc_df, on='day', how='outer')
    except KeyError:
        pass

    dfs.append(curr_df)

df = pd.concat(dfs)
col = df.pop('max_pvc')
df.insert(9, col.name, col)
col = df.pop('min_slope')
df.insert(10, col.name, col)
label_df = pd.concat([df.pop(x) for x in ['mouse_id', 'day', 'si_perf']], axis=1)

fig, axes = plt.subplots(nrows=3, ncols=4, layout='constrained', figsize=(18, 12), sharex='all')

for i, ax in enumerate(axes.flatten()):
    if i < len(df.columns):
        nan_mask = ~df.iloc[:, i].isna()
        corr = stats.pearsonr(label_df['si_perf'][nan_mask], df.iloc[:, i][nan_mask])
        ax.scatter(label_df['si_perf'], df.iloc[:, i])
        if corr.pvalue > 0.05:
            text_color = 'red'
        else:
            text_color = 'green'
        ax.set_title(df.iloc[:, i].name)
        ax.set_box_aspect(1)
        ax.text(0.99, 0.95, f'r={corr.statistic:.3f}\np={corr.pvalue:.3f}', transform=ax.transAxes, c=text_color,
                verticalalignment='top', horizontalalignment='right')

