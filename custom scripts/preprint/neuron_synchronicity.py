#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 07/02/2024 16:49
@author: hheise

"""
import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pg
from glob import glob
import os
import matplotlib.pyplot as plt
import scipy.stats as sts

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
metrics = ['skewness', 'perc99', 'perc95', 'perc80', 'avg_corr', 'median_corr', 'avg_corr_pc', 'median_corr_pc']
conn_data = pd.DataFrame((hheise_connectivity.NeuronNeuronCorrelation & 'trace_type="spat_dff"' &
                          f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'day', *metrics, as_dict=True))
conn = dc.merge_dfs(df=conn_data, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

def average_corrcoef(arr):
    return np.tanh(np.nanmean(np.arctanh(arr)))

# Compute mouse averages across phases
avg_conn = conn.groupby(['mouse_id', 'phase']).agg({'skewness': np.nanmean, 'perc99': average_corrcoef, 'perc80': average_corrcoef,
                                                    'perc95': average_corrcoef,
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

avg_conn.pivot(index='phase', columns='mouse_id', values='perc95').loc[['pre', 'early', 'late']].to_clipboard(index=True, header=True)


#%% Kernel Density Estimates of correlation matrices

kde_data = pd.DataFrame((hheise_connectivity.NeuronNeuronCorrelationKDE & 'trace_type="dff"' &
                         f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'day', 'y_kde', as_dict=True))
kde_data = dc.merge_dfs(df=kde_data, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)


# Average KDE curves and CMF curves per mouse and phase
kde_df = []
for (mouse, phase), data in kde_data.groupby(['mouse_id', 'phase']):
    avg_kde = np.mean(np.stack(data['y_kde']), axis=0)
    kde_df.append(pd.DataFrame(dict(mouse_id=mouse, phase=phase, group=group[group.mouse_id == mouse]['group'].iloc[0],
                                    avg_kde=avg_kde, x=np.linspace(-1, 1, num=200))))
kde_df = pd.concat(kde_df)


# Plot data
g = sns.FacetGrid(data=kde_df, row='phase', hue='group', row_order=['pre', 'early', 'late'])
g.map(sns.lineplot, 'x', 'avg_kde', errorbar='ci')
g.add_legend()

# Export for prism
kde_df[kde_df.phase == 'late'].pivot(index='x', columns='mouse_id', values='avg_kde').to_clipboard(header=False, index=False)

# Cumulative distribution function
conn_data = pd.DataFrame((hheise_connectivity.NeuronNeuronCorrelation & 'trace_type="dff"' &
                          f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'day', 'corr_matrix', as_dict=True))
conn = dc.merge_dfs(df=conn_data, sphere_df=spheres, inj_df=injection, vr_df=vr_performance)

cdf_df = []
for (mouse, phase), data in conn.groupby(['mouse_id', 'phase']):
    cdf_df.append(pd.DataFrame(dict(mouse_id=mouse, phase=phase, group=group[group.mouse_id == mouse]['group'].iloc[0],
                                    corrcoefs=np.concatenate([mat.flatten()[~np.isnan(mat.flatten())] for mat in data['corr_matrix']]))))
cdf_df = pd.concat(cdf_df)

g = sns.FacetGrid(data=cdf_df, row='phase', hue='group', row_order=['pre', 'early', 'late'])
g.map(sns.ecdfplot, 'corrcoefs')
g.add_legend()

# Compute difference in distribution (2-sample Kolmogorov-Smirnov) per phase between groups
for phase, data in cdf_df.groupby('phase'):
    print(f'Phase {phase}:')
    sham_rec = sts.ks_2samp(data1=data.loc[data.group == 'Sham', 'corrcoefs'].to_numpy(), data2=data.loc[data.group == 'Recovery', 'corrcoefs'].to_numpy())
    print(f'\tSham vs Recovery:\n\t\tD={sham_rec.statistic:.4f} - p={sham_rec.pvalue:.4f} - R at largest difference: {sham_rec.statistic_location:.4f}')
    sham_norec = sts.ks_2samp(data1=data.loc[data.group == 'Sham', 'corrcoefs'].to_numpy(), data2=data.loc[data.group == 'No Recovery', 'corrcoefs'].to_numpy())
    print(f'\tSham vs No Recovery:\n\t\tD={sham_norec.statistic:.4f} - p={sham_norec.pvalue:.4f} - R at largest difference: {sham_norec.statistic_location:.4f}')
    rec_norec = sts.ks_2samp(data1=data.loc[data.group == 'Recovery', 'corrcoefs'].to_numpy(), data2=data.loc[data.group == 'No Recovery', 'corrcoefs'].to_numpy())
    print(f'\tRecovery vs No Recovery:\n\t\tD={rec_norec.statistic:.4f} - p={rec_norec.pvalue:.4f} - R at largest difference: {rec_norec.statistic_location:.4f}')

# Plot as violin plots
data_filt=cdf_df[cdf_df.group.isin(['Sham', 'No Recovery'])]
plt.figure()
sns.violinplot(data=data_filt, x='phase', y='corrcoefs', hue='group', split=True, order=['pre', 'early', 'late'],
               hue_order=['Sham', 'No Recovery'], log_scale=True)


# Standard deviation of correlation matrices
sd_df = []
for (mouse, phase), data in conn.groupby(['mouse_id', 'phase']):
    sd_df.append(pd.DataFrame([dict(mouse_id=mouse, phase=phase, group=group[group.mouse_id == mouse]['group'].iloc[0],
                                    sd=np.mean([np.nanstd(mat) for mat in data['corr_matrix']]))]))
sd_df = pd.concat(sd_df, ignore_index=True)

sd_df.pivot(index='phase', columns='mouse_id', values='sd').loc[['pre', 'early', 'late']].to_clipboard(header=True, index=True)


#%% Test 20th percentile functional pair dataset for sphericity

# Import and filter data
filepath = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell-pair-fractions-in-correlation-distribution-quantile-0.8.csv'
data = pd.read_csv(filepath)
data = data.loc[:, data.iloc[0] == 'above count divided by total count by cell type quantile 0.8']
nc_nc = data.loc[:, data.iloc[1] == 'non-coding-non-coding'].iloc[2:].set_axis(['pre', 'early', 'late'], axis=1,inplace=True,copy=False)
nc_pc = data.loc[:, data.iloc[1] == 'non-coding-place-cell'].iloc[2:].set_axis(['pre', 'early', 'late'], axis=1,inplace=True,copy=False)
pc_pc = data.loc[:, data.iloc[1] == 'place-cell-place-cell'].iloc[2:].set_axis(['pre', 'early', 'late'], axis=1,inplace=True,copy=False)



#%% Import and handle Filippos data for celltype-specific pairs in top X percentile of highly correlated cells

def load_data(root = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo\cell_pair_correlations'):

    csv_files = glob(os.path.join(root, '*.csv'))

    data_dict = {'dff': {}, 'sa': {}}

    for file in csv_files:
        splits = file.split('.')[-2].split('-')
        percentile = splits[0]
        dtype = splits[-1]

        data = pd.read_csv(file, index_col=0)

        # Remove day 1
        data = data.loc[data.day != 1]

        # Average data per mouse and period
        avg_cols = data.columns[2:5]
        avg_df = data.groupby(['mouse', 'period'], as_index=False).agg({avg_cols[0]: np.nanmean, avg_cols[1]: np.nanmean,
                                                        avg_cols[2]: np.nanmean})
        avg_df = pd.merge(avg_df, data[['mouse', 'mouse_group_fine']].drop_duplicates(), on='mouse', how='left')

        # Remove mice that have a NaN value somewhere
        # nan_mice = avg_df.loc[avg_df.isna().any(axis=1), 'mouse'].unique()
        # avg_df_filt = avg_df[~avg_df.mouse.isin(nan_mice)]

        data_dict[dtype][percentile] = avg_df

    # Make two quick plots
    cols = ['NC-NC', 'NC-PC', 'PC-PC']
    for dtype, dicts in data_dict.items():
        fig, ax = plt.subplots(nrows=len(dicts), ncols=3, sharex='all', sharey='none', layout='constrained')
        fig.suptitle(dtype)
        for row, (perc, df) in enumerate(dicts.items()):
            if len(perc) == 1:
                perc += '0'

            for col, col_name in enumerate(avg_cols):
                ax[row, col].axhline((100 - int(perc)) / 100, color='grey', linestyle='--')
                sns.boxplot(data=df, x='period', y=col_name, order=['pre', 'early', 'late'], hue='mouse_group_fine',
                            hue_order=['sham', 'recovery', 'no recovery'], ax=ax[row, col])
                if row == 0:
                    ax[row, col].set_title(cols[col])
                if col == 0:
                    ax[row, col].set_ylabel(f'{perc}th percentile')
                else:
                    ax[row, col].set_ylabel('')

                if row != 0 or col != 0:
                    ax[row, col].get_legend().remove()


        data_dict['dff']['99'].pivot(index='period', columns='mouse', values=avg_cols[0]).loc[['pre', 'early', 'late']].to_clipboard(index=False, header=False)


