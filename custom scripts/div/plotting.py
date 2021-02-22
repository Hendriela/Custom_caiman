import numpy as np
from glob import glob
import re
import os
import standard_pipeline.performance_check as performance
import matplotlib.pyplot as plt

"""
File to reproduce plots from posters, presentations etc.
"""

#%% Licks over position bins (alternative to learning curve)

path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200827'
bin_size = 2


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]]) + 10

mouse = path.split(sep=os.path.sep)[-2]

file_list = glob(path + '\\*\\merged_behavior*.txt')
if len(file_list) == 0:
    file_list = glob(path + '\\merged_behavior*.txt')
file_list.sort(key=natural_keys)

data = np.zeros((len(file_list), int(120 / bin_size)))
for idx, file in enumerate(file_list):
    data[idx] = performance.get_binned_licking(np.loadtxt(file), bin_size=bin_size, normalized=False)

data[data > 0] = 1

plt.figure(figsize=(8,4))
plt.hist(np.linspace(0, 400, 60), bins=60, weights=(np.sum(data, axis=0)/len(data))*100,
         facecolor='black', edgecolor='black')

for zone in zone_borders:
    plt.axvspan(zone[0]*(10/3), zone[1]*(10/3), color='red', alpha=0.3)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0,105)
plt.xlabel('VR position', fontsize=22)
plt.ylabel('Licked in bin [%]', fontsize=22)
plt.tight_layout()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%% normalize data
import standard_pipeline.place_cell_pipeline as pipe
import statsmodels.api as sm

# Load example session
pcf = pipe.load_pcf(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200627')

# Get dF/F from mobile frames for example neuron
neuron = 369
all_mask = np.concatenate(pcf.params["resting_mask"]) # merge all trials
all_act = np.concatenate(pcf.session_spikes[neuron]) # merge all trials
trace = all_act[all_mask]

# plotting
plt.figure()
ax = plt.subplot(2,3,1)
sm.qqplot(trace, ax=ax, line="s")
ax.set_title("Peters spike prediction")
ax = plt.subplot(2,3,4)
sm.qqplot(trace, ax=ax, line="45")

ax = plt.subplot(2,3,2)
sm.qqplot(np.log(trace), ax=ax, line="s")
ax.set_title("log(x)")
ax = plt.subplot(2,3,5)
sm.qqplot(np.log(trace), ax=ax, line="45")

ax = plt.subplot(2,3,3)
sm.qqplot(np.sqrt(trace), ax=ax, line="s")
ax.set_title("sqrt(x)")
ax = plt.subplot(2,3,6)
sm.qqplot(np.sqrt(trace), ax=ax, line="45")

#%% Correlate simple data with performance
import pandas as pd
import seaborn as sns
import scipy.stats as st

simple_data = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\simple_data.pickle')
simple_data_filt = simple_data[simple_data.mouse.isin(['M33', 'M38', 'M39', 'M41'])]
simple_data_filt = simple_data_filt[simple_data_filt.session.between(20200818, 20200911)]
fields = ['mean_spikerate', 'median_spikerate', 'ratio', 'pvc_slope', 'min_pvc', 'sec_peak_ratio']
ylabels = ['Mean spike rate [Hz]', 'Median spike rate [Hz]', 'Place cell ratio', 'PVC slope', 'minimum PVC', '2nd peak ratio']



plt.figure()
sm.qqplot(simple_data_filt.licking_binned, line="s")
plt.figure()
plt.hist(simple_data_filt.licking_binned, bins="auto")

field="ratio"

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
sns.set_context("talk")
plt.figure()
ax1 = plt.subplot(1,2,2)
sns.scatterplot(x="licking_binned", y="mean_spikerate", data=simple_data_filt, hue="mouse", ax=ax1)
sns.regplot(x="licking_binned", y="mean_spikerate", data=simple_data_filt, scatter=False, color="grey", ax=ax1)
ax1.set(xlabel='Performance', ylabel='Mean firing rate [Hz]')
corr, p = st.spearmanr(a=simple_data_filt.licking_binned, b=simple_data_filt["mean_spikerate"])
string = "$\\rho$ = {:.2f}\np < 0.05".format(corr)
ax1.text(0.05, 0.95, string, transform=ax1.transAxes, fontsize=17,
        verticalalignment='top', bbox=props)

ax2 = plt.subplot(1,2,1)
sns.scatterplot(x="licking_binned", y="ratio", data=simple_data_filt, hue="mouse", ax=ax2, legend=False)
sns.regplot(x="licking_binned", y="ratio", data=simple_data_filt, scatter=False, color="grey", ax=ax2)
ax2.set(xlabel='Performance', ylabel="Place cells [%]")
corr, p = st.spearmanr(a=simple_data_filt.licking_binned, b=simple_data_filt["ratio"])
string = "$\\rho$ = {:.2f}\np < 0.001".format(corr)
ax2.text(0.05, 0.95, string, transform=ax2.transAxes, fontsize=17,
        verticalalignment='top', bbox=props)
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# place a text box in upper left in axes coords

plt.tight_layout()


#%% place cell plots
import standard_pipeline.place_cell_pipeline as pipe
pcf = pipe.load_pcf(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200905')
pcf.plot_all_place_cells(sort="max")
