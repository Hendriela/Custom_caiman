#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 29/08/2023 09:32
@author: hheise

"""
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scistats

from schema import hheise_decoder, hheise_grouping, common_mice, hheise_placecell, hheise_behav

useful_metrics = ['accuracy', 'mae', 'mae_quad', 'sensitivity_rz', 'specificity_rz']
#%% Load data

mice = np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id'))

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession * hheise_decoder.BayesianParameter & f'mouse_id = {mouse}'
                             & f'day <= "{surgery_day}"' & 'bayesian_id > 0' & 'bayesian_id < 10').fetch(as_dict=True)))
df = pd.concat(dfs)

#%% Plot accuracy for all mice in prestroke
plt.figure()
sns.barplot(df[df['bayesian_id'] == 1], x='mouse_id', y='accuracy')
sns.stripplot(df[df['bayesian_id'] == 1], x='mouse_id', y='accuracy', color='gray')


#%% Compare differences between neuron subsets
subsets = df.neuron_subset.unique()
subset_comps = list(itertools.combinations(df.neuron_subset.unique(), 2))
attr = ['accuracy', 'mae', 'mae_quad_std', 'mae_quad', 'sensitivity_rz', 'specificity_rz', 'mcc', 'abs_error']

# Plot scatter plots
for sub1, sub2 in subset_comps:
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.suptitle(f"{sub1} vs {sub2}")
    for i in range(len(attr)):
        ax = axes.flatten()[i]
        sns.scatterplot(x=np.array(df[df['neuron_subset'] == sub1][attr[i]]),
                        y=np.array(df[df['neuron_subset'] == sub2][attr[i]]),
                        ax=ax)
        ax.set_title(attr[i])

        lower = np.min([ax.get_ylim()[0], ax.get_xlim()[0]])
        upper = np.max([ax.get_ylim()[1], ax.get_xlim()[1]])
        ax.set_xlim((lower, upper))
        ax.set_ylim((lower, upper))
        ax.set_xlabel(sub1)
        ax.set_ylabel(sub2)
        diag = ax.axline((upper, upper), slope=1, c='grey', ls='--')
        # ax.fill_between(diag._x, diag._y, transform=ax.get_yaxis_transform())
        ax.set_aspect('equal')


# Plot absolute values
df_long = pd.melt(df, id_vars='neuron_subset', value_vars=[*attr], var_name='metric')
df_long['value_norm'] = df_long.groupby('metric').transform(lambda x: (x - x.mean()) / x.std())
plt.figure()
ax = sns.boxplot(data=df_long, x='metric', y='value_norm', hue='neuron_subset', showfliers=False)

# Compute and plot difference
diff_df = []
for sub1, sub2 in subset_comps:
    for i in range(len(attr)-1):
        diff = (np.array(df[df['neuron_subset'] == sub2][attr[i]]) - np.array(df[df['neuron_subset'] == sub1][attr[i]])) /\
               np.array(df[df['neuron_subset'] == sub1][attr[i]]) * 100
        if 'mae' in attr[i]:
            diff = -diff
        diff_df.append(pd.DataFrame(dict(diff=diff, comp=f'{sub1} vs {sub2}', attr=attr[i])))
diff_df = pd.concat(diff_df)

plt.figure()
# ax = sns.barplot(data=diff_df, x='comp', y='diff', hue='attr')
ax = sns.boxplot(data=diff_df, x='comp', y='diff', hue='attr', showfliers=False)
ax.axhline(0, c='grey', ls='--')

# ax = sns.stripplot(data=diff_df, x='comp', y='diff', hue='attr', ax=ax)
ax.set_ylabel('Performance improvement [%]')

#%% Test normal distribution for error metrics

out = (hheise_decoder.BayesianDecoderWithinSession & 'bayesian_id =1').fetch(format='frame')
numeric_only = out.select_dtypes(include=np.number)
test = numeric_only.apply(scistats.normaltest)

#%% Test if prestroke metrics are above chance level for mice (use stability sorting)

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession & f'mouse_id = {mouse}' & f'day <= "{surgery_day}"' & 'bayesian_id = 1').fetch(as_dict=True)))
df = pd.concat(dfs)

# Fetch chance levels
chance = hheise_decoder.BayesianDecoderErrorChanceLevels().fetch(format='frame').drop(columns=['calculation_procedure']).squeeze()
chance.name = 'chance_levels'
chance = chance[~chance.index.str.contains('std')]

# Plot barplots
df_long = pd.melt(df.reset_index(), value_vars=list(chance.index), id_vars=['mouse_id', 'index'], var_name='metric')
# g = sns.FacetGrid(df_long, col="metric", col_wrap=6, sharey=False, hue='mouse_id')
# g.map(sns.barplot, "mouse_id", 'value', order=df_long.mouse_id.unique())
g = sns.FacetGrid(df_long, col="metric", col_wrap=6, sharey=False)
g.map(sns.boxplot, "mouse_id", 'value')
for col, ax in zip(g.col_names, g.axes):
    ax.axhline(chance[col], c='grey', ls='--')

# Compute p-values for one-sample t-test -> is per-mouse prestroke avg above chance level?
# First tried wilcoxon, as samples might not be normally distributed, but Wilcoxon cant handle low sample sizes
t_tests = []
raw_data = []
for metric in chance.index:
    curr_metric = df_long[df_long['metric'] == metric]
    raw_data.append(curr_metric.pivot(index='index', columns='mouse_id', values='value'))
    for mouse_id, mouse_data in curr_metric.groupby('mouse_id'):
        res = scistats.ttest_1samp(mouse_data['value'], chance[metric])
        if res.pvalue <= 0.05:
            direction = 'greater' if (mouse_data['value'] - chance[metric]).mean() > 0 else 'less'
        else:
            direction = 'none'
        t_tests.append(pd.DataFrame([{'metric': metric, 'mouse_id': mouse_id, 'p': res.pvalue, 'direction': direction}]))
t_tests = pd.concat(t_tests)

#%% Correlate metrics against each other
numeric_only_filt = numeric_only.iloc[:, (~numeric_only.columns.str.contains('shift')) & (~numeric_only.columns.str.contains('std'))]
numeric_only_filt['accuracy'] = 1 - numeric_only_filt['accuracy']
numeric_only_filt['accuracy_quad'] = 1 - numeric_only_filt['accuracy_quad']
numeric_only_filt['acc_max'] = 1 - numeric_only_filt['acc_max']
numeric_only_filt['acc_max_quad'] = 1 - numeric_only_filt['acc_max_quad']
corr = numeric_only_filt.corr()
plt.figure()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), vmin=-1, vmax=1, annot=True,
            xticklabels=corr.columns, yticklabels=corr.index, cmap='vlag')

corr = numeric_only_filt[useful_metrics].corr()
plt.figure()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), vmin=-1, vmax=1, annot=True,
            xticklabels=corr.columns, yticklabels=corr.index, cmap='vlag')


#%% Correlate VR performance with metrics

### Session-wise
data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession * hheise_behav.VRPerformance
                     & 'bayesian_id=1').fetch('mouse_id', 'day', 'si_binned_run', *useful_metrics, as_dict=True))
pks = (hheise_decoder.BayesianDecoderWithinSession * hheise_behav.VRPerformance & 'bayesian_id=1').fetch('KEY')
data['binned_lick_ratio'] = [(hheise_behav.VRPerformance & pk).get_mean()[0] for pk in pks]

is_pre = []
for mouse_id, mouse_data in data.groupby('mouse_id'):
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse_id}').fetch(
        'surgery_date')[0].date()
    is_pre.append(mouse_data['day'] <= surgery_day)
data['is_pre'] = pd.concat(is_pre)

behav_metric = 'si_binned_run'

# Scatter plots
data_long = data.melt(id_vars=['mouse_id', behav_metric], value_vars=useful_metrics)
g = sns.FacetGrid(data_long, col="variable", col_wrap=3, sharey=False, hue='mouse_id')
g.map(sns.scatterplot, behav_metric, 'value')

export = data.pivot(columns='mouse_id', values='accuracy')

# Correlate metrics with SI performance
perf_corr = []
for mouse_id, mouse_data in data.groupby('mouse_id'):
    # Pearson
    # for corr_method in ['pearson', 'spearman']:
    corr_method = 'spearman'
    for behav_metric in ['si_binned_run', 'binned_lick_ratio']:
        corr_all = mouse_data[[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_pre = mouse_data[mouse_data['is_pre']][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_post = mouse_data[~mouse_data['is_pre']][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_all), period='both', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_pre), period='pre', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_post), period='post', method=corr_method, metric=behav_metric)]))
perf_corr = pd.concat(perf_corr)
perf_corr_long = perf_corr.melt(value_vars=useful_metrics, id_vars=['mouse_id', 'period', 'metric'])

dat = perf_corr_long[(perf_corr_long['variable'] == 'accuracy')]
plt.bar(x=dat['mouse_id'].astype('str'), height=dat['value'])
sns.barplot(dat, x='period', y='value', hue='mouse_id')

for period in ['both', 'pre', 'post']:
    g = sns.FacetGrid(perf_corr_long[perf_corr_long['period'] == period], col="variable", col_wrap=3, sharey=False,
                      ylim=(-1, 1))
    g.map(sns.barplot, "mouse_id", 'value')
    g.fig.suptitle(period)


g = sns.FacetGrid(perf_corr_long, col="variable", col_wrap=3, sharey=False, ylim=(-1, 1))
for ax in g.axes:
    ax.axhline(0, c='grey', ls='--')
g.map(sns.boxplot, "period", 'value', 'metric')
g.axes.flatten()[-1].legend()

plt.figure()
sns.boxplot(perf_corr_long[perf_corr_long['variable'] == 'mae_quad'], x='period', y='value', hue='method')

### Trial-wise
data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession.Trial * hheise_behav.VRPerformance
                     & 'bayesian_id=1').fetch('mouse_id', 'day', 'trial_id', 'si_count', 'binned_lick_ratio',
                                              *useful_metrics, as_dict=True))
data['blr'] = data.apply(lambda x: x['binned_lick_ratio'][x['trial_id']], axis=1)
data['si'] = data.apply(lambda x: x['si_count'][x['trial_id']], axis=1)

# Scatter plots
behav_metric = 'blr'
data_long = data.melt(id_vars=['mouse_id', behav_metric], value_vars=useful_metrics)
g = sns.FacetGrid(data_long, col="variable", col_wrap=3, sharey=False, hue='mouse_id')
g.map(sns.scatterplot, behav_metric, 'value')
