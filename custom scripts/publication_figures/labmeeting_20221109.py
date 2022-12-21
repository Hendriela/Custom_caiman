#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 07/11/2022 17:23
@author: hheise

Code for the labmeeting on 9.11.22 and the Synapsis 2022 forum
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import os

from hheise_scripts import hheise_util
from util import helper
from schema import hheise_behav, common_img

#%% Performance Batch 8

no_deficit = [109, 123, 120]
no_deficit_flicker = [111, 114, 116]
recovery = [113]
deficit_no_flicker = [121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]
mice = [*no_deficit, *recovery, *no_deficit_flicker, *deficit_no_flicker, *deficit_flicker, *sham_injection]

# Normalized performance data (normalized to 3 days pre-stroke)
norm_performance = (hheise_behav.VRPerformance & f'mouse_id in {helper.in_query(mice)}').get_normalized_performance(baseline_days=3, plotting=False)
# Pivot data for export to Prism
prism_performance = norm_performance.pivot(index='day', columns='mouse_id', values='performance')
prism_performance.to_csv(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\Progress Reports\09.11.2022\batch8_performance.csv',
                         sep='\t')

# Plot dFF examples of flicker sessions
dff = (common_img.Segmentation & 'mouse_id=110' & 'day="2022-08-21"').get_traces()
max_idx = np.unravel_index(np.argmax(dff, axis=None), dff.shape)[0]

idx = np.array([64698, 71863, 79059, 86225, 93360, 100528]) // 30   # start idx of flicker periods for M112
shift = 0
y=dff[max_idx, shift:]
x=np.arange(len(y))/30
plt.figure()
plt.plot(x, y)
plt.xlabel('time [s]', fontsize=14)
plt.ylabel('dF/F', fontsize=14)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
[plt.axvspan(i-shift//30, i+60-shift//30, color='red', alpha=0.2) for i in idx]

#%% Synapsis Forum Poster

def compare_corr_matrix(trace1, trace2, name1='trace1', name2='trace2', alternative='two-sided',
                        export_csv=False, dirpath = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch8\analysis\flicker'):
    """

    Args:
        trace1: 2D np.ndarray, with shape (n_neurons, n_frames), lower triangle in correlation matrix
        trace2: 2D np.ndarray, with shape (n_neurons, n_frames), upper triangle in correlation matrix
        alternative: Alternative hypothesis. 'Less': trace1 < trace1, 'greater': trace1 > trace2

    Returns:

    """

    def upper(df):
        '''Returns the upper triangle of a correlation matrix.
        You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
        Args:
          df: pandas or numpy correlation matrix
        Returns:
          list of values from upper triangle
        '''
        try:
            assert (type(df) == np.ndarray)
        except:
            if type(df) == pd.DataFrame:
                df = df.values
            else:
                raise TypeError('Must be np.ndarray or pd.DataFrame')
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]

    ## Compute correlation (remove diagonal)
    trace1_corr = np.corrcoef(trace1)
    trace2_corr = np.corrcoef(trace2)
    diag_idx = np.diag_indices(trace1_corr.shape[0])
    trace1_corr[diag_idx] = np.nan
    trace2_corr[diag_idx] = np.nan

    trace1_tril = np.tril(trace1_corr, k=-1)
    trace1_tril[trace1_tril == 0] = np.nan
    trace2_tril = np.tril(trace2_corr, k=-1)
    trace2_tril[trace2_tril == 0] = np.nan

    ## Plot shared correlation matrix (trace1 in lower, trace2 in upper triangle)
    comb_mat = trace1_tril.copy()
    upper_tril = np.triu_indices(comb_mat.shape[0], k=1)
    comb_mat[upper_tril] = trace2_corr[upper_tril]
    fig_ = plt.figure(figsize=(16, 12))
    ax_ = sns.heatmap(comb_mat)
    ax_.set_xticks([])
    ax_.set_yticks([])

    # Formatting for large size (16, 12)
    ax_.set_ylabel('# neurons', fontsize=28)
    ax_.set_xlabel('# neurons', fontsize=28)

    # colorbar
    ax_.collections[0].colorbar.ax.tick_params(labelsize=30)
    ax_.collections[0].colorbar.set_label(label="Pearsons' R", size=35)

    ax_.text(0.5, 1.06, 'Flicker on', ha='center', va='top', transform=ax_.transAxes, fontsize=38)
    ax_.text(-0.12, 0.5, 'Flicker off', ha='left', va='center', rotation=90, transform=ax_.transAxes, fontsize=38)

    ## Run permutation significance test
    # (from https://towardsdatascience.com/how-to-measure-similarity-between-two-correlation-matrices-ce2ea13d8231)
    mat1 = pd.DataFrame(trace1.T).corr()
    mat2 = pd.DataFrame(trace2.T).corr()

    rhos_ = []
    n_iter_ = 5000
    true_rho_, spear_p_ = stats.spearmanr(upper(mat1), upper(mat2))
    # matrix permutation, shuffle the groups
    m_ids_ = list(mat1.columns)
    m2_v_ = upper(mat2)
    for iter_ in range(n_iter_):
        np.random.shuffle(m_ids_)  # shuffle list
        r_, _ = stats.spearmanr(upper(mat1.loc[m_ids_, m_ids_]), m2_v_)
        rhos_.append(r_)
    perm_p_ = ((np.sum(np.abs(true_rho_) <= np.abs(rhos_))) + 1) / (n_iter_ + 1)  # two-tailed test

    # Plot bootstrapping results
    f_, ax_1 = plt.subplots()
    plt.hist(rhos_, bins=20)
    ax_1.axvline(true_rho_, color='r', linestyle='--')
    ax_1.set(title=f"Equal samples Permuted p: {perm_p_:.3f}", ylabel="counts", xlabel="rho")
    plt.show()

    # Compare correlation coefficient distribution
    trace1_corr_flat = trace1_tril.flatten()[~np.isnan(trace1_tril.flatten())]
    trace2_corr_flat = trace2_tril.flatten()[~np.isnan(trace2_tril.flatten())]
    wilc_stat, wilc_p = stats.ttest_rel(a=np.arctanh(trace1_corr_flat), b=np.arctanh(trace2_corr_flat))
    print('Wilcoxon Test:\n\tW-stat: {} ({:.2f}%)\n\t'
          'Wilcoxon p: {:.2e}'.format(int(wilc_stat), wilc_stat/(len(trace1_corr_flat)**2)*100, wilc_p))

    # Export data for violinplot in Prism
    # Exclude data points around 0 to make plots readable
    export_trace1_corr = np.zeros(trace1_corr_flat.shape) * np.nan
    trace1_mask = (trace1_corr_flat > 0.05) | (trace1_corr_flat < -0.05)
    export_trace1_corr[trace1_mask] = trace1_corr_flat[trace1_mask]

    export_trace2_corr = np.zeros(trace2_corr_flat.shape) * np.nan
    trace2_mask = (trace2_corr_flat > 0.05) | (trace2_corr_flat < -0.05)
    export_trace2_corr[trace2_mask] = trace2_corr_flat[trace2_mask]

    if export_csv:
        np.savetxt(os.path.join(dirpath, f'{name1}_neuronpairs_corr.csv'), export_trace1_corr.T, fmt='%.8f')
        np.savetxt(os.path.join(dirpath, f'{name2}_neuronpairs_corr.csv'), export_trace2_corr.T, fmt='%.8f')

    df1 = pd.DataFrame(dict(array=name1, r=export_trace1_corr, metric='neuron_pairs'))
    df2 = pd.DataFrame(dict(array=name2, r=export_trace2_corr, metric='neuron_pairs'))
    df_neur_pairs = pd.concat([df1, df2])
    plt.figure()
    sns.violinplot(data=df_neur_pairs, x='array', y='r').set(title='Neuron-neuron pairs')

    # Pairwise change
    pair_change_tril = trace2_tril-trace1_tril
    pair_change = pair_change_tril.flatten()[~np.isnan(pair_change_tril.flatten())]

    pair_abs_thresh = np.std(pair_change)
    pair_abs_thresh = 0.2
    increased_r = np.sum(pair_change >= pair_abs_thresh)/len(pair_change)*100
    decreased_r = np.sum(pair_change <= -pair_abs_thresh)/len(pair_change)*100
    same_r = np.sum(np.logical_and(pair_change > -pair_abs_thresh, pair_change < pair_abs_thresh))/len(pair_change)*100
    print('Neuron-neuron pair correlation absolute change:\n\tAverage change: {:.4f} +/- {:.4f}\n\t'
          'Increased R (+{:.3f}): {} ({:.2f}%)\n\tDecreased R (-{:.3f}): {} ({:.2f}%)'
          '\n\tSame R: {} ({:.2f}%)'.format(np.mean(pair_change), np.std(pair_change),
                                            pair_abs_thresh, np.sum(pair_change >= pair_abs_thresh), increased_r,
                                            pair_abs_thresh, np.sum(pair_change <= -pair_abs_thresh), decreased_r,
                                            np.sum(np.logical_and(pair_change > -pair_abs_thresh, pair_change < pair_abs_thresh)), same_r))

    # Compare average z-transformed correlation per neuron
    trace1_neur = np.nanmean(np.arctanh(trace1_corr), axis=0)
    trace2_neur = np.nanmean(np.arctanh(trace2_corr), axis=0)

    df1 = pd.DataFrame(dict(array=name1, r=trace1_neur, metric='neuron_avg'))
    df2 = pd.DataFrame(dict(array=name2, r=trace2_neur, metric='neuron_avg'))
    df_neur_avg = pd.concat([df1, df2])
    plt.figure()
    sns.violinplot(data=df_neur_avg, x='array', y='r').set(title='Neuron avg')

    if export_csv:
        np.savetxt(os.path.join(dirpath, f'{name1}_NeurAvg_corr.csv'), trace1_neur.T, fmt='%.8f')
        np.savetxt(os.path.join(dirpath, f'{name2}_NeurAvg_corr.csv'), trace2_neur.T, fmt='%.8f')

    neur_fold = trace2_neur/trace1_neur
    neur_fold_thresh = np.std(neur_fold)
    # neur_fold_thresh = 5
    increased_r = np.sum(neur_fold >= neur_fold_thresh)/len(neur_fold) * 100
    decreased_r = np.sum(neur_fold <= 1/neur_fold_thresh)/len(neur_fold) * 100
    same_r = np.sum(np.logical_and(neur_fold > 1/neur_fold_thresh, neur_fold < neur_fold_thresh))/len(neur_fold) * 100
    print('Neuron mean correlation fold-change:\n\tAverage change: {:.4f} +/- {:.4f}\n\t'
          'Increased R (>{:.2f}): {} ({:.2f}%)\n\tDecreased R (<{:.2f}): {} ({:.2f}%)'
          '\n\tSame R: {} ({:.2f}%)'.format(np.mean(neur_fold), np.std(neur_fold),
                                            neur_fold_thresh, np.sum(neur_fold >= neur_fold_thresh), increased_r,
                                            1/neur_fold_thresh, np.sum(neur_fold <= 1/neur_fold_thresh), decreased_r,
                                            np.sum(np.logical_and(neur_fold > 1/neur_fold_thresh, neur_fold < neur_fold_thresh)), same_r))

    neur_change = trace2_neur-trace1_neur
    # change_thresh = np.std(neur_change)
    change_thresh = 0.005
    increased_r = np.sum(neur_change >= change_thresh)/len(neur_change) * 100
    decreased_r = np.sum(neur_change <= -change_thresh)/len(neur_change) * 100
    same_r = np.sum(np.logical_and(neur_change > -change_thresh, neur_change < change_thresh))/len(neur_change) * 100
    print('Neuron mean correlation change:\n\tAverage change: {:.4f} +/- {:.4f}\n\t'
          'Increased R (+{:.3f}): {} ({:.2f}%)\n\tDecreased R (-{:.3f}): {} ({:.2f}%)'
          '\n\tSame R: {} ({:.2f}%)'.format(np.mean(neur_change), np.std(neur_change),
                                            change_thresh, np.sum(neur_change >= change_thresh), increased_r,
                                            change_thresh, np.sum(neur_change <= -change_thresh), decreased_r,
                                            np.sum(np.logical_and(neur_change > -change_thresh, neur_change < change_thresh)),
                                            same_r))

    if export_csv:
        np.savetxt(os.path.join(dirpath, f'Pairwise_neuron_change.csv'), neur_change.T, fmt='%.12f')

    return comb_mat, pd.concat([df_neur_pairs, df_neur_avg])

# Plot deconvolved
key = {'mouse_id':110, 'day':"2022-08-21"}
decon = (common_img.Segmentation & key).get_traces('decon')
# Get idx of flicker periods
flick = 492     # Flicker starts in file_00025 at frame 492
flick1 = flick + (common_img.RawImagingFile & key & 'part<22').fetch('nr_frames').sum()
flick = 1343     # file_00027
flick2 = flick + (common_img.RawImagingFile & key & 'part<24').fetch('nr_frames').sum()
flick = 960     # file_00029
flick3 = flick + (common_img.RawImagingFile & key & 'part<26').fetch('nr_frames').sum()
flick = 2143     # file_00031
flick4 = flick + (common_img.RawImagingFile & key & 'part<28').fetch('nr_frames').sum()
flick = 432     # file_00035
flick5 = flick + (common_img.RawImagingFile & key & 'part<32').fetch('nr_frames').sum()

# Plot traces with On-flicker marked
idx_sec = np.array([flick1, flick2, flick3, flick4, flick5]) // 30
shift = 28000
y=decon[max_idx, shift:]
x=np.arange(len(y))/30
plt.figure()
plt.plot(x, y)
plt.xlabel('time [s]', fontsize=14)
plt.ylabel('dF/F', fontsize=14)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
[plt.axvspan(i-shift/30, i+60-shift/30, color='red', alpha=0.2) for i in idx_sec]

# Activity during flicker vs between flicker
snr = (common_img.Segmentation.ROI & key).fetch('snr')
snr_mask = snr > 3  # Only use neurons with an SNR > 3 (high-noise neurons might be contaminated

shift = (common_img.RawImagingFile & key & 'part<22').fetch('nr_frames').sum()
flick_decon = decon[snr_mask, shift:]
idx = np.array([flick1, flick2, flick3, flick4, flick5]) - shift
flicker_mask = np.zeros(flick_decon.shape[1], dtype=bool)
for i in idx:
    flicker_mask[i:i+1800] = True


# Get last minute of flicker off to have same number of time points (wrap around ends to get same number of samples
def get_part_off(trace_arr, snr_mask_, frame_shift, flick_start_idx, pre_ON=1, buffer=5):
    """

    Args:
        trace_arr: Deconvolved traces of the session
        snr_mask_: bool mask which neurons have high enough SNR to be considered
        frame_shift: Idx of first flicker trial
        flick_start_idx:
        pre_ON:
        buffer:

    Returns:

    """
    curr_flick_decon = trace_arr[snr_mask_, frame_shift:]
    frame_buff = buffer*30
    new_flicker_mask = np.zeros(curr_flick_decon.shape[1], dtype=bool)
    for i in flick_start_idx[1:]:
        start_frame = i - frame_buff - pre_ON*1800
        new_flicker_mask[start_frame:start_frame+1800] = True  # Accept frames 65 seconds to 5 seconds before flicker onset
    regular_off = curr_flick_decon[:, new_flicker_mask]

    # Add last point manually, take from last trial with VR
    global_first_idx = flick_start_idx[0] + frame_shift
    first_off = global_first_idx - frame_buff - pre_ON*1800
    additional_off = trace_arr[snr_mask_, first_off:first_off+1800]
    complete_off = np.hstack([additional_off, regular_off])
    return complete_off


# For each neuron, get firing rate during flicker vs between flicker
fr_on = flick_decon[:, flicker_mask]
fr_off = flick_decon[:, ~flicker_mask]

fr_off_1st_min = get_part_off(trace_arr=decon, snr_mask_=snr_mask, frame_shift=shift, flick_start_idx=idx, pre_ON=3, buffer=0)
fr_off_2nd_min = get_part_off(trace_arr=decon, snr_mask_=snr_mask, frame_shift=shift, flick_start_idx=idx, pre_ON=2, buffer=0)
fr_off_3rd_min = get_part_off(trace_arr=decon, snr_mask_=snr_mask, frame_shift=shift, flick_start_idx=idx, pre_ON=1, buffer=5)

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\Synapsis Forum 2022\in_flicker_fr.csv', fr_on.T, fmt='%.8f')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\Synapsis Forum 2022\bet_flicker_fr.csv', fr_off.T, fmt='%.8f')

# mean firing rate
fr_on_mean = np.sum(fr_on, axis=1) / (fr_on.shape[1]/30)
fr_off_mean = np.sum(fr_off_1st_min, axis=1) / (fr_off_1st_min.shape[1]/30)
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch8\analysis\flicker\1st_min_off\off_on_mean_fr_1min.csv', np.vstack((fr_off_mean, fr_on_mean)).T, fmt='%.8f')

# Check how many neurons increase their firing rate (>10% increase)
rel_change = fr_on_mean / fr_off_mean
rel_fr_upper_thresh = np.mean(rel_change) + np.std(rel_change)
rel_fr_lower_thresh = np.mean(rel_change) - np.std(rel_change)
rel_fr_upper_thresh = 1.1
rel_fr_lower_thresh = 0.9
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch8\analysis\flicker\1st_min_off\off_on_fr_fold_change_1min.csv', rel_change.T, fmt='%.8f')

increased_act = np.sum(rel_change >= rel_fr_upper_thresh) / len(rel_change) * 100
decreased_act = np.sum(rel_change <= rel_fr_lower_thresh) / len(rel_change) * 100
same_act = np.sum(np.logical_and(rel_change < rel_fr_upper_thresh, rel_change > rel_fr_lower_thresh)) / len(rel_change) * 100
print('Neuron mean correlation fold-change:\n\tAverage change: {:.4f} +/- {:.4f}\n\t'
      'Increased R (>{:.2f}): {} ({:.2f}%)\n\tDecreased R (<{:.2f}): {} ({:.2f}%)'
      '\n\tSame R: {} ({:.2f}%)'.format(np.mean(rel_change), np.std(rel_change),
                                        rel_fr_upper_thresh, np.sum(rel_change >= rel_fr_upper_thresh), increased_act,
                                        rel_fr_lower_thresh, np.sum(rel_change <= rel_fr_lower_thresh), decreased_act,
                                        np.sum(np.logical_and(rel_change < rel_fr_upper_thresh,
                                                              rel_change > rel_fr_lower_thresh)), same_act))


## Compute correlation between all neurons

# Whole OFF duration (unequal sample sizes)
whole_off_mat, whole_off_stat = compare_corr_matrix(trace1=fr_off, trace2=fr_on, name1='OFF', name2='ON', export_csv=False)

# Only third OFF minute
pre1min_off_mat, pre1min_off_stat = compare_corr_matrix(trace1=fr_off_3rd_min, trace2=fr_on, name1='OFF', name2='ON', export_csv=False)

# Only second OFF minute
pre2min_off_mat, pre2min_off_stat = compare_corr_matrix(trace1=fr_off_2nd_min, trace2=fr_on, name1='OFF', name2='ON', export_csv=False)

# Only first OFF minute
pre3min_off_mat, pre3min_off_stat = compare_corr_matrix(trace1=fr_off_1st_min, trace2=fr_on, name1='OFF', name2='ON', export_csv=False)

# Compare different OFF sets
vs3_1min_off_mat, vs3_1min_off_stat = compare_corr_matrix(trace1=fr_off_1st_min, trace2=fr_off_3rd_min, name1='pre3', name2='pre1', export_csv=False)
vs2_1min_off_mat, vs2_1min_off_stat = compare_corr_matrix(trace1=fr_off_2nd_min, trace2=fr_off_3rd_min, name1='pre2', name2='pre1', export_csv=False)
vs3_2min_off_mat, vs3_2min_off_stat = compare_corr_matrix(trace1=fr_off_1st_min, trace2=fr_off_2nd_min, name1='pre3', name2='pre2', export_csv=False)

# plt.tight_layout()
plt.savefig(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch8\analysis\flicker\corr_matrix_1st_min_off.png')




# Test for periodicity
from scipy import signal
crosscorr = signal.correlate(in_flick[598], in_flick[21])







import matplotlib.pyplot as plt
import numpy as np

Fs = 1000
f1 = 40
f2 = 30
sample = 1000
x = np.arange(sample)
y1 = np.sin(2 * np.pi * f1 * x / Fs)
plt.figure()
plt.plot(x, y1)
y2 = np.sin(2 * np.pi * f2 * x / Fs)
plt.plot(x, y2)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()