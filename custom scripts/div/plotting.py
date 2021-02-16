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

