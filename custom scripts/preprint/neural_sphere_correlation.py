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
from schema import hheise_placecell, hheise_behav, hheise_hist, hheise_pvc, hheise_decoder, common_mice, common_img

mice = [33, 41,  # Batch 3
        63, 69,  # Batch 5
        83, 85, 86, 89, 90, 91, 93, 94, 95,  # Batch 7
        108, 110, 111, 112, 113, 114, 115, 116, 122]  # Batch 8
exercise = [83, 85, 89, 90, 91] # Mice that received physical exercise training
single_days = [-4, -3, -2, -1, 0, 3, 6, 9, 12, 15, 18]  # days after 18 have few mice with many


def merge_dfs(df, sphere_df, inj_df, vr_df=None):

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

    if vr_df is not None:
        df_merge = pd.merge(df_merge, vr_df, on=['mouse_id', 'day'])

    return df_merge


def correlate_metric(df, y_metric, x_metric='spheres', time_name='rel_day_align', plotting=False, ax=None,
                     exclude_mice=None, include_mice=None, neg_corr=False):

    if exclude_mice is not None and include_mice is not None:
        raise ValueError('Exclude_mice and include_mice cannot be defined simultaneously.')
    elif exclude_mice is not None:
        df_filt = df[~df.mouse_id.isin(exclude_mice)]
    elif include_mice is not None:
        df_filt = df[df.mouse_id.isin(include_mice)]
    else:
        df_filt = df

    n_mice = df_filt.mouse_id.nunique()

    corr = []
    for day, day_df in df_filt.groupby(time_name):

        if (time_name == 'rel_day_align') and (day in single_days) or (time_name != 'rel_day_align'):
        # if day_df.mouse_id.nunique() == n_mice:

            if time_name == 'phase':    # To compare phases, average phase metrics mouse-wise
                y = day_df.groupby('mouse_id')[y_metric].agg('mean')
                x = day_df.groupby('mouse_id')[x_metric].agg('mean')
            else:
                y = day_df[y_metric]
                x = day_df[x_metric]

            if y_metric in ['mae_quad', 'mae']:      # Invert metrics that are worse if large
                y = -y

            result = stats.pearsonr(x, y)

            corr.append(pd.DataFrame([dict(day=day, corr=result.statistic, corr_p=result.pvalue, y_metric=y_metric,
                                           x_metric=x_metric)]))

    corr = pd.concat(corr, ignore_index=True)

    if neg_corr:
        corr['corr'] = -corr['corr']

    if plotting:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        sns.lineplot(x=corr.day, y=corr['corr'], ax=ax)

    return corr


def iterative_exclusion(df: pd.DataFrame, y_metric, n_exclude, x_metric='spheres', time_name='rel_day_align', n_shuffle=None):

    mice_sorted = df[['mouse_id', 'spheres']].drop_duplicates().sort_values('spheres', ascending=False)

    true_df = []
    rng_df = []

    for n_ex in range(1, n_exclude+1):

        # Perform correlation with the n_ex most sphere-loaded mice removed
        corr_real = correlate_metric(df=df, y_metric=y_metric, x_metric=x_metric, time_name=time_name,
                                     exclude_mice=mice_sorted[:n_ex]['mouse_id'].to_numpy())
        corr_real['mice_excluded'] = [mice_sorted[:n_ex]['mouse_id'].to_numpy()] * len(corr_real)
        corr_real['sphere_limit'] = mice_sorted.iloc[n_ex-1]['spheres']
        corr_real['n_excluded'] = n_ex
        true_df.append(corr_real)

        # Control population with n_ex random mice removed
        if n_shuffle is None:
            n_shuffle = len(mice_sorted)
        for i in range(n_shuffle):
            ex_mice = np.random.choice(mice_sorted['mouse_id'], n_ex, replace=False)
            corr_shuff = correlate_metric(df=df, y_metric=y_metric, x_metric=x_metric, time_name=time_name,
                                          exclude_mice=ex_mice)
            corr_shuff['mice_excluded'] = [ex_mice] * len(corr_real)
            corr_shuff['n_excluded'] = n_ex
            corr_shuff['n_shuffle'] = i
            rng_df.append(corr_shuff)

    true_df = pd.concat(true_df, ignore_index=True)
    rng_df = pd.concat(rng_df, ignore_index=True)

    true_df['label'] = true_df.apply(lambda x: f"{int(x['sphere_limit'])} ({x['n_excluded']})", axis=1)
    rng_df['label'] = true_df.apply(lambda x: f"{int(x['sphere_limit'])} ({x['n_excluded']})", axis=1)

    return true_df, rng_df


#%% Load basic data
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"' &
                        f'mouse_id in {helper.in_query(mice)}').proj(spheres='count_extrap').fetch('mouse_id', 'spheres', as_dict=True))
injection = pd.DataFrame((common_mice.Surgery & 'username="hheise"' & f'mouse_id in {helper.in_query(mice)}' &
                          'surgery_type="Microsphere injection"').fetch('mouse_id', 'surgery_date',as_dict=True))
injection.surgery_date = injection.surgery_date.dt.date
vr_performance = pd.DataFrame((hheise_behav.VRPerformance & f'mouse_id in {helper.in_query(mice)}' &
                               'perf_param_id=0').fetch('mouse_id', 'day', 'si_binned_run', as_dict=True))

#%% Decoder
metrics = ['accuracy', 'mae_quad', 'sensitivity_rz']
decoder = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession() &
                        'bayesian_id=1').fetch('mouse_id', 'day', *metrics, as_dict=True))
decoder = merge_dfs(df=decoder, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

# Plot scatter plots
g = sns.FacetGrid(decoder[decoder.rel_day_align.isin(single_days)], col='rel_day_align', col_wrap=4)
g.map_dataframe(sns.scatterplot, x='spheres', y='accuracy').set(xscale='log')

### DAY-WISE ###
decoder_corr = pd.concat([correlate_metric(df=decoder, y_metric=met) for met in metrics], ignore_index=True)
decoder_corr.pivot(index='day', columns='metric', values='corr_p').to_clipboard()

### CORRELATE AGAINST BEHAVIOR ###
decoder_corr = pd.concat([correlate_metric(df=decoder, y_metric=met, x_metric='si_binned_run', neg_corr=True) for met in metrics], ignore_index=True)
decoder_corr.pivot(index='day', columns='y_metric', values='corr').to_clipboard()

# Exclude high-sphere-load mice
corr_true, corr_shuff = iterative_exclusion(df=decoder, n_exclude=10, y_metric='mae_quad', x_metric='spheres', n_shuffle=10)

corr_true.pivot(index='day', columns='sphere_limit', values='corr').to_clipboard()

fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(18, 10), sharex='all', sharey='row')
sns.lineplot(corr_true, x='day', y='corr', hue='label', ax=ax[0, 0], palette='magma')
sns.lineplot(corr_true, x='day', y='corr_p', hue='label', ax=ax[1, 0], palette='magma')
sns.lineplot(corr_shuff, x='day', y='corr', hue='label', ax=ax[0, 1], palette='magma')
sns.lineplot(corr_shuff, x='day', y='corr_p', hue='label', ax=ax[1, 1], palette='magma')
ax[1, 0].set(yscale='log')
ax[1, 0].axhline(0.05, linestyle=':', color='black')
ax[1, 0].axvline(0.5, linestyle='--', color='red')
ax[1, 1].set(yscale='log')
ax[1, 1].axhline(0.05, linestyle=':', color='black')
ax[1, 1].axvline(0.5, linestyle='--', color='red')

ax[0, 0].axvline(0.5, linestyle='--', color='red')
ax[0, 0].axhline(0, linestyle=':', color='black')
ax[0, 1].axvline(0.5, linestyle='--', color='red')
ax[0, 1].axhline(0, linestyle=':', color='black')




# Exclude exercise mice
decoder_corr_ex = pd.concat([correlate_metric(decoder, met, exclude_mice=exercise) for met in metrics], ignore_index=True)
decoder_corr_ex.pivot(index='day', columns='metric', values='corr_p').to_clipboard(index=False, header=False)

### Average metric within each phase (not too useful) ###
decoder_corr = pd.concat([correlate_metric(decoder, met, time_name='phase') for met in metrics], ignore_index=True)
decoder_corr.pivot(index='day', columns='y_metric', values='corr').to_clipboard()

# Exclude exercise mice
decoder_corr_ex = pd.concat([correlate_metric(decoder, met, time_name='phase', exclude_mice=exercise) for met in metrics], ignore_index=True)
decoder_corr_ex.pivot(index='day', columns='metric', values='corr_p').to_clipboard(index=False, header=False)


#%% VR Performance

performance = merge_dfs(df=vr_performance, sphere_df=spheres, inj_df=injection)

g = sns.FacetGrid(performance[performance.rel_day_align.isin(single_days)], col='rel_day_align', col_wrap=4)
g.map_dataframe(sns.scatterplot, x='spheres', y='si_binned_run').set(xscale='log')

performance_corr = correlate_metric(performance, 'si_binned_run')
performance_corr.pivot(index='day', columns='y_metric', values='corr_p').to_clipboard()

#%% PVC
metrics = ['max_pvc', 'min_slope', 'pvc_rel_dif']
pvc = pd.DataFrame((hheise_pvc.PvcCrossSessionEval() * hheise_pvc.PvcCrossSession & 'locations="all"' &
                    'circular=0').fetch('mouse_id', 'phase', 'day', 'max_pvc', 'min_slope', 'pvc_rel_dif', as_dict=True))
pvc.rename(columns={'phase': 'phase_orig'}, inplace=True)
pvc = merge_dfs(df=pvc, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

pvc_corr = pd.concat([correlate_metric(pvc, met, time_name='phase_orig') for met in metrics], ignore_index=True)
pvc_corr.pivot(index='day', columns='y_metric', values='corr_p').loc[['pre', 'pre_post', 'early', 'late']].to_clipboard()

#%% Place cell ratio
pcr = pd.DataFrame((hheise_placecell.PlaceCell() & 'corridor_type=0' &
                    'place_cell_id=2').fetch('mouse_id', 'day', 'place_cell_ratio', as_dict=True))
pcr = merge_dfs(df=pcr, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

pcr_corr = correlate_metric(pcr, y_metric='place_cell_ratio')
pcr_corr.pivot(index='day', columns='y_metric', values='corr_p').to_clipboard()

#%% Within-session stability
stab = pd.DataFrame((hheise_placecell.SpatialInformation.ROI() & 'corridor_type=0' &
                    'place_cell_id=2').fetch('mouse_id', 'day', 'stability', as_dict=True))
stab = stab.groupby(['mouse_id', 'day']).agg('mean').reset_index()
stab = merge_dfs(df=stab, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

stab_corr = correlate_metric(stab, y_metric='stability')
stab_corr.pivot(index='day', columns='y_metric', values='corr_p').to_clipboard(index=False, header=True)

#%% Firing rate stability
fr = pd.DataFrame((common_img.ActivityStatistics.ROI * common_img.Segmentation.ROI &
                   'accepted=1').fetch('mouse_id', 'day', 'rate_spikes', as_dict=True))
fr = fr.groupby(['mouse_id', 'day']).agg('mean').reset_index()
fr = merge_dfs(df=fr, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

fr_corr = correlate_metric(fr, y_metric='rate_spikes')
fr_corr.pivot(index='day', columns='y_metric', values='corr_p').to_clipboard(index=False, header=True)
