#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 08/12/2023 16:23
@author: hheise

"""

import pandas as pd
import numpy as np

# Load data
data = pd.read_csv(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Victor\ORT_OLT\Hand_label_B3_ORT.csv', delimiter=';')

# Filtering, formatting
data = data[data.state != 'post-stroke-2nd'].rename(columns={'video': 'mouse_id'})


def summarize_ort(table):

    summary = []
    for (state, experiment, mouse_id), df in table.groupby(['state', 'experiment', 'mouse_id']):

        # Rename stroke phase to make it consistent with other ORT plots
        if state == 'post-stroke':
            state_name = 'Stroke'
        else:
            state_name = 'Healthy'

        # Pivot table for computations
        df_pivot = df.pivot(index='minute', columns='object', values='exploration')

        # Average across minutes
        exp_index_o1 = np.nanmean(df_pivot.apply(lambda x: x['o1'] / (x['o1'] + x['o2']), axis=1))
        disc_index_o1 = np.nanmean(df_pivot.apply(lambda x: (x['o1']-x['o2']) / (x['o1'] + x['o2']), axis=1))

        # Sum over entire video
        exp_index_o1_sum = df_pivot['o1'].sum() / df_pivot.sum().sum()
        disc_index_o1_sum = (df_pivot['o1'].sum() - df_pivot['o2'].sum()) / df_pivot.sum().sum()

        summary.append(pd.DataFrame([dict(mouse_id=mouse_id, state=state_name, experiment=experiment, group=df['group'].iloc[0],
                                          exp_index_o1=exp_index_o1, exp_index_o2=1-exp_index_o1,
                                          exp_index_o1_sum=exp_index_o1_sum, exp_index_o2_sum=1-exp_index_o1_sum,
                                          disc_index_o1=disc_index_o1, disc_index_o2=-disc_index_o1,
                                          disc_index_o1_sum=disc_index_o1_sum, disc_index_o2_sum=-disc_index_o1_sum)]))
    return pd.concat(summary)


#%%
ort = summarize_ort(data)

ort[(ort.state == 'Healthy') & (ort.experiment == 'testing')].pivot(columns='mouse_id', index='group',
                                                                        values='disc_index_o2').to_clipboard(index=True, header=True)
