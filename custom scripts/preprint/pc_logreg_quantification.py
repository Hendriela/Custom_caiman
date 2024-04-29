#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/04/2024 14:57
@author: hheise

"""

import pandas as pd
import seaborn as sns
from scipy import stats

from schema import hheise_transitions

#%% Quantify which solver could regress most sessions without errors

df = pd.DataFrame((hheise_transitions.PlaceCellRegressionUnivariate() * hheise_transitions.PlaceCellRegressionParams()).fetch('variable', 'solver', 'cross', 'mouse_id', as_dict=True))
df = df.loc[df.cross == 0].drop(columns='cross')

# Count solver per variable
counter = df.groupby(['variable', 'cross'])['solver'].value_counts()
counter.name = 'counter'
counter = counter.reset_index()

# Plot data per variable
g = sns.catplot(counter, kind="bar", x="solver", y="counter", col="variable", row="cross", margin_titles=True)
g.set(ylim=(450, 550))

g = sns.catplot(counter, kind="bar", x="solver", y="counter", col="variable", hue="cross", col_wrap=4, margin_titles=True)
g.set(ylim=(450, 550))

#%% Quantify which sampling method performs best
df = pd.DataFrame((hheise_transitions.PlaceCellRegressionUnivariate() * hheise_transitions.PlaceCellRegressionParams() & 'solver="bfgs"').fetch(as_dict=True))

### Count successful samplings per variable
counter = df.groupby(['variable', 'cross'])['sampling'].value_counts()
counter.name = 'counter'
counter = counter.reset_index()

# Divide counts of samplings because we have four models per sampling method
counter.loc[counter.sampling != 'none', 'counter'] = counter.loc[counter.sampling != 'none', 'counter'] / 4

# Plot data per variable
g = sns.catplot(counter, kind="bar", x="sampling", y="counter", col="variable", hue="cross", col_wrap=4, margin_titles=True).set(ylim=(50, 65))
ax = sns.barplot(counter, x="sampling", y="counter", hue="cross", errorbar='sd').set(ylim=(50, 65))   # Average across variables (they are all similar)

### Which sampling yields most NaNs
counter_nan = df.loc[df.isna().sum(axis=1).astype(bool)].groupby(['variable', 'cross'])['sampling'].value_counts()
counter_nan.name = 'counter'
counter_nan = counter_nan.reset_index()

# Divide counts of samplings because we have four models per sampling method
counter_nan.loc[counter_nan.sampling != 'none', 'counter'] = counter_nan.loc[counter_nan.sampling != 'none', 'counter'] / 4

# Normalize counts by total number of successful models
counter_nan['freq'] = counter_nan['counter'] / counter['counter'] * 100

# Plot data per variable
sns.catplot(counter_nan, kind="bar", x="sampling", y="freq", col="variable", hue="cross", col_wrap=4, margin_titles=True).set(ylim=(0, 100))
sns.barplot(counter_nan, x="sampling", y="freq", hue="cross", errorbar='sd').set(ylim=(0, 100))   # Average across variables (they are all similar)
sns.boxplot(counter_nan, x="sampling", y="freq", hue="cross").set(ylim=(0, 100))   # Average across variables (they are all similar)


### Which sampling yields best performance

# Set undefined precision to 0 for this calculation
df['precision'].fillna(0, inplace=True)

metrics = ['accuracy', 'precision', 'sensitivity', 'log_likelihood', 'pseudo_r2']
# To make variables and metrics comparable, z-score metrics for each variable
zscores = df[df.phase == 'pre'][['variable', *metrics]].groupby('variable', group_keys=True).apply(stats.zscore).reset_index().set_index('level_1')
zscore_df = df.loc[:, ~df.columns.isin(['variable', *metrics])].join(zscores)
zscore_df_long = zscore_df.melt(value_vars=metrics, value_name='z-score',
                                id_vars=zscore_df.columns[~zscore_df.columns.isin(['variable', *metrics])])

# Plot metrics per variable
sns.catplot(zscore_df_long, kind="bar", x="sampling", y="z-score", col="variable", hue="cross", col_wrap=5, margin_titles=True)

#%% Quantify which weights perform best

# Compute for both sampling methods separately
sampling_mask = (df.sampling == 'under') & (df.phase == 'pre')

df['precision'].fillna(0, inplace=True)

metrics = ['accuracy', 'precision', 'sensitivity', 'log_likelihood', 'pseudo_r2']
# To make variables and metrics comparable, z-score metrics for each variable
zscores = df[sampling_mask][['variable', *metrics]].groupby('variable', group_keys=True).apply(stats.zscore).reset_index().set_index('level_1')
zscore_df = df.loc[sampling_mask, ~df.columns.isin(['variable', *metrics])].join(zscores)
zscore_df_long = zscore_df.melt(value_vars=metrics, value_name='z-score',
                                id_vars=zscore_df.columns[~zscore_df.columns.isin(['variable', *metrics])])

# Plot metrics per variable
sns.catplot(zscore_df_long, kind="bar", x="freq_weight", y="z-score", col="variable", hue="cross", col_wrap=5, margin_titles=True)

# Plot non-normalized prediction metrics
sns.violinplot(df.loc[sampling_mask], x="freq_weight", y="precision", hue="cross") #, errorbar='sd')

sns.catplot(df.loc[sampling_mask], kind="bar", x="freq_weight", y="sensitivity", col="variable", hue="cross", col_wrap=5, margin_titles=True)



