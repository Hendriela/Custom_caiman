#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 29/08/2023 09:32
@author: hheise

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from schema import hheise_decoder, hheise_grouping, common_mice, hheise_placecell

#%% Check accuracy for all mice in prestroke

mice = np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id'))

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession & f'mouse_id = {mouse}'
                             & f'day <= "{surgery_day}"').fetch(as_dict=True)))
df = pd.concat(dfs)

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSessionLog & f'mouse_id = {mouse}'
                             & f'day <= "{surgery_day}"').fetch(as_dict=True)))
df_log = pd.concat(dfs)

df_merge = pd.merge(left=df, right=df_log, on=['bayesian_id', 'username', 'mouse_id', 'day', 'session_num',
                                               'motion_id', 'caiman_id'], suffixes=(None, '_log'))

# Compare differences between log and non-log model
attr1 = ['accuracy', 'mae', 'mae_quad_std', 'mae_quad', 'in_rz_acc', 'out_rz_acc', 'mae_log']
attr2 = ['accuracy_log', 'mae_log', 'mae_quad_std_log', 'mae_quad_log', 'sensitivity_rz', 'specificity_rz', 'mae_quad_log']
fig, axes = plt.subplots(nrows=2, ncols=4)
for i in range(len(attr1)):
    sns.scatterplot(data=df_merge, x=attr1[i], y=attr2[i], ax=axes.flatten()[i])
    axes.flatten()[i].set_ylim(axes.flatten()[i].get_xlim())
    axes.flatten()[i].plot(np.arange(axes.flatten()[i].get_xlim()[1]), np.arange(axes.flatten()[i].get_ylim()[1]), color='grey', linestyle='--')

# Produce long-format Dataframe
df_long = pd.melt(df, id_vars=)

plt.figure()
sns.barplot(df[df['bayesian_id'] == 1], x='mouse_id', y='accuracy')
sns.stripplot(df[df['bayesian_id'] == 1], x='mouse_id', y='accuracy', color='gray')
sns.barplot(df_merge[df_merge['bayesian_id'] == 0], x='mouse_id', y='mae_log')
sns.stripplot(df_merge[df_merge['bayesian_id'] == 0], x='mouse_id', y='mae_log', color='gray')
sns.barplot(df_merge[df_merge['bayesian_id'] == 0], x='mouse_id', y='mae_quad_log')
sns.stripplot(df_merge[df_merge['bayesian_id'] == 0], x='mouse_id', y='mae_quad_log', color='gray')

plt.figure()
sns.barplot(df[df['bayesian_id'] == 0], x='mouse_id', y='mae_quad')
sns.stripplot(df[df['bayesian_id'] == 0], x='mouse_id', y='mae_quad', color='gray')


#%% Correlate accuracy/errors with number of cells included in the model
n_cells = []
for i, row in df_merge.iterrows():
    key = dict(username=row['username'], mouse_id=row['mouse_id'], day=row['day'], session_num=row['session_num'],
               place_cell_id=2)
    n_cells.append(len((hheise_placecell.PlaceCell.ROI & key & 'is_place_cell=1')))
df_merge['n_cells'] = n_cells

attr = ['accuracy_log', 'mae_log', 'mae_quad_std_log', 'mae_quad_log', 'sensitivity_rz', 'specificity_rz']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i in range(len(attr)):
    sns.regplot(data=df_merge[df_merge['bayesian_id'] == 1], x='n_cells', y=attr[i], ax=axes.flatten()[i])
    ax=axes.flatten()[i].set_box_aspect(1)

