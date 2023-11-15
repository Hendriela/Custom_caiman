#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/11/2023 17:48
@author: hheise

Correlate various neural metrics against sphere loads
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from util import helper
from schema import hheise_placecell, hheise_hist, hheise_pvc, hheise_decoder, common_mice

mice = [33, 41,  # Batch 3
        63, 69,  # Batch 5
        83, 85, 86, 89, 90, 91, 93, 94, 95,  # Batch 7
        108, 110, 111, 112, 113, 114, 115, 116, 122]  # Batch 8
exercise = [83, 85, 89, 90, 91] # Mice that received physical exercise training
single_days = [-4, -3, -2, -1, 0, 3, 6, 9, 12, 15, 18]  # days after 18 have few mice with many


def merge_dfs(df, sphere_df, inj_df):

    df_merge = pd.merge(df, sphere_df, on='mouse_id')
    df_merge = pd.merge(df_merge, inj_df, on='mouse_id')

    df_merge['rel_day'] = df_merge.apply(lambda x: (x['day'] - x['surgery_date']).days, axis=1)

    rel_days_align = []
    for mouse_id, mouse_df in df_merge.groupby('mouse_id'):
        rel_days = mouse_df.rel_day.copy()
        if 3 not in rel_days:
            rel_days[(rel_days == 2) | (rel_days == 4)] = 3
        rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
        rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
        rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
        rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
        rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
        rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
        rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
        if 28 not in rel_days:
            rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27
        rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
        rel_days[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1))+1, 1)
        rel_days.name = 'rel_day_align'
        rel_days_align.append(rel_days)
    rel_days_align = pd.concat(rel_days_align)
    df_merge = df_merge.join(rel_days_align)

    # Do not analyze day 1
    df_merge = df_merge[df_merge.rel_day_align != 1]

    def get_phase(row):
        if row.rel_day_align <= 0:
            return 'pre'
        elif row.rel_day_align <= 7:
            return 'early'
        else:
            return 'late'
    df_merge['phase'] = df_merge.apply(get_phase, axis=1)

    return df_merge


def correlate_metric(df, metric_name, time_name='rel_day_align', plotting=False, ax=None, exclude_mice=None):

    if exclude_mice is not None:
        df_filt = df[~df.mouse_id.isin(exclude_mice)]
    else:
        df_filt = df

    n_mice = df_filt.mouse_id.nunique()

    corr = []
    for day, day_df in df_filt.groupby(time_name):

        if (time_name == 'rel_day_align') and (day in single_days) or (time_name != 'rel_day_align'):
        # if day_df.mouse_id.nunique() == n_mice:

            if time_name == 'phase':    # To compare phases, average phase metrics mouse-wise
                y = day_df.groupby('mouse_id')[metric_name].agg('mean')
                x = day_df.groupby('mouse_id')['spheres'].agg('mean')
            else:
                y = day_df[metric_name]
                x = day_df.spheres

            if metric_name in ['mae_quad', 'mae']:      # Invert metrics that are worse if large
                y = -y

            result = stats.pearsonr(x, y)
            corr.append(pd.DataFrame([dict(day=day, corr=result.statistic, corr_p=result.pvalue, metric=metric_name)]))

    corr = pd.concat(corr, ignore_index=True)

    if plotting:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        sns.lineplot(x=corr.day, y=corr['corr'], ax=ax)

    return corr


#%% Load sphere data
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"'& f'mouse_id in {helper.in_query(mice)}').proj(spheres='count_extrap').fetch('mouse_id', 'spheres', as_dict=True))
injection = pd.DataFrame((common_mice.Surgery & 'username="hheise"' & f'mouse_id in {helper.in_query(mice)}' & 'surgery_type="Microsphere injection"').fetch('mouse_id', 'surgery_date',as_dict=True))
injection.surgery_date = injection.surgery_date.dt.date

#%% Decoder
metrics = ['accuracy', 'mae_quad', 'sensitivity_rz']
decoder = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession() &
                        'bayesian_id=1').fetch('mouse_id', 'day', *metrics, as_dict=True))
decoder = merge_dfs(df=decoder, sphere_df=spheres, inj_df=injection)

# Plot scatter plots
g = sns.FacetGrid(decoder[decoder.rel_day_align.isin(single_days)], col='rel_day_align', col_wrap=4)
g.map_dataframe(sns.scatterplot, x='spheres', y='accuracy').set(xscale='log')

### DAY-WISE ###
decoder_corr = pd.concat([correlate_metric(decoder, met) for met in metrics], ignore_index=True)
decoder_corr.pivot(index='day', columns='metric', values='corr_p').to_clipboard()

# Exclude exercise mice
decoder_corr_ex = pd.concat([correlate_metric(decoder, met, exclude_mice=exercise) for met in metrics], ignore_index=True)
decoder_corr_ex.pivot(index='day', columns='metric', values='corr_p').to_clipboard(index=False, header=False)

### PHASE-WISE ###
decoder_corr = pd.concat([correlate_metric(decoder, met, time_name='phase') for met in metrics], ignore_index=True)
decoder_corr.pivot(index='day', columns='metric', values='corr_p').loc[['pre', 'early', 'late']].to_clipboard()

# Exclude exercise mice
decoder_corr_ex = pd.concat([correlate_metric(decoder, met, time_name='phase', exclude_mice=exercise) for met in metrics], ignore_index=True)
decoder_corr_ex.pivot(index='day', columns='metric', values='corr_p').to_clipboard(index=False, header=False)

