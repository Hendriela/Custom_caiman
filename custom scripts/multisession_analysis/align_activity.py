import standard_pipeline.place_cell_pipeline as pipe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

session = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200627'
window_size = (2, 4)

# Load PCF
pcf = pipe.load_pcf(session)

# Set RZ borders
if pcf.params['novel']:
    zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
else:
    zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])

# Find frame count of valve openings
df_hit_list = []
df_miss_list = []
for trial_idx, trial in enumerate(pcf.behavior):
    # Recover frames where the mouse was not moving and get frame indices
    frame_idx = np.where(np.nan_to_num(trial[:, 3], nan=1) == 1)[0]
    valve_idx = np.where(trial[:, 6] == 1)[0]
    # Check if any reward zone was without reward
    for idx, zone in enumerate(zone_borders):
        zone_data = trial[np.logical_and(zone[0] < trial[:, 1], trial[:, 1] < zone[1]), :]
        if sum(zone_data[:, 6]) == 0:
            df_miss_list.append(pd.DataFrame({'trial': [trial_idx], 'zone':[idx]}))

    # For every valve opening, get the index of the next frame and save it for later alignment
    for reward in valve_idx:
        # Check if the current valve opening happened in a reward zone
        rzs = [True if (zone[0] < trial[reward, 1] < zone[1]) else False for zone in zone_borders]
        in_rz = any(rzs)
        if in_rz:
            # Check the time delay of entering the reward zone and valve opening
            time_diff = trial[reward, 0] - trial[np.where(trial[:, 1] > zone_borders[np.where(rzs)[0][0], 0])[0][0], 0]
            # Get the valve opening distance from the reward zone start
            pos_diff = trial[reward, 1] - zone_borders[np.where(rzs)[0][0], 0]

        # Get frame count differences (to find the next frame index)
        diff = frame_idx - reward
        df_hit_list.append(pd.DataFrame({'trial': [trial_idx], 'frame': [np.where(diff > 0, diff, np.inf).argmin()],
                                         'rz': [in_rz], 'pos': pos_diff, 'delay': time_diff}))
# structure of df_hit:
# trial index; frame at which the valve opened; whether the valve opened in a RZ or not; distance (in VR coords) between
# RZ start and valve opening; time delay (in s) of entering the reward zone and valve opening
if len(df_hit_list) > 0:
    df_hit = pd.concat(df_hit_list, ignore_index=True)
if len(df_miss_list) > 0:
    df_miss = pd.concat(df_miss_list, ignore_index=True)

# For every neuron, align traces to all reward frames
align = np.zeros((len(pcf.cnmf.estimates.F_dff), (window_size[0]+window_size[1])*pcf.cnmf.params.data['fr'], len(df_hit)))
for cell_idx in range(len(pcf.session)):
    for rew_idx, rew in df_hit.iterrows():
        first_frame = rew['frame']-window_size[0]*pcf.cnmf.params.data['fr']
        last_frame = rew['frame']+window_size[1]*pcf.cnmf.params.data['fr']
        align[cell_idx, :, rew_idx] = pcf.session[cell_idx][rew['trial']][first_frame:last_frame]

fig = plt.figure()
for i in range(align.shape[2]):
    plt.plot(np.linspace(-window_size[0], window_size[1], align.shape[1]),
             align[466,:,i])

plt.figure()
x = np.linspace(-window_size[0], window_size[1], align.shape[1])
for i in range(align.shape[0]):
    align_trace = np.mean(align[i,:], axis=1)
    if any(align_trace > 0.20):
        plt.plot(x, align_trace, alpha=0.5, label=f"cell {i}")
plt.legend()

#%% COM
x = np.linspace(0,400,80)
y = pcf.bin_avg_activity[706]

def fun(x, A, k, x0):
    return A * np.cos(k*(x - x0))

A = max(y)
k = 80
x0 = 0

from scipy.optimize import curve_fit
import math
param, cov = curve_fit(fun, x, y, [A, k, x0])
plt.figure()
plt.plot(x, y, color="red", linewidth=1,linestyle="dashed")
plt.plot(x, fun(x, *param), color="blue", linewidth=1)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def polar(x, y):
    """returns r, theta(degrees)
    """
    r = (x ** 2 + y ** 2) ** .5
    theta = math.degrees(math.atan2(y,x))
    return r, theta

rh, ph = cart2pol(x, y)

r, theta = polar(x,y)
plt.figure()
plt.polar(theta, r)

#%% Speed


# Get average speed for every frame in each trial
speed = []
for trial in range(len(pcf.behavior)):
    # First, we have to get the window size around each frame trigger to get an average speed during this frame
    # You get it by getting the half-median distance between frame trigger signals
    win = np.median(np.diff(np.where(np.nan_to_num(pcf.behavior[trial][:,3], nan=1))[0]))/2
    # Get indices of frames
    fr_idx = np.where(pcf.behavior[trial][:,3]==1)[0]
    # go through every frame, and get the mean speed in each frame window
    curr_speed = np.zeros(len(fr_idx))*np.nan
    for n_frame, i in enumerate(fr_idx):
        j = np.arange(i-win, i+win, dtype=int)
        if max(j) > len(pcf.behavior[trial]):
            j = j[j<len(pcf.behavior[trial])]
        curr_speed[n_frame] = np.mean(pcf.behavior[trial][j, 5])
    speed.append(curr_speed)

# CONCATENATE TRIALS, TREAT THEM AS ONE
speed = np.concatenate(speed) # merge all trials

# get activity of each cell in the right format
cell_act = np.zeros((len(pcf.session), len(speed)))*np.nan
all_mask = np.concatenate(pcf.params["resting_mask"]) # merge all trials
for cell in range(len(pcf.session)):
    all_act = np.concatenate(pcf.session[cell]) # merge all trials
    cell_act[cell] = all_act[all_mask]


#%% own simple correlation

# compute correlation between speed and activity of all neurons
import scipy.stats as st
spear = np.zeros((len(cell_act), 2))*np.nan
for cell in range(len(cell_act)):
    corr, p = st.spearmanr(a=cell_act[cell], b=speed)
    spear[cell, 0] = corr
    spear[cell, 1] = p*len(cell_act)

# COMPUTE CORRELATION FOR SINGLE TRIALS AND AVERAGE
# for trial in range(len(pcf.behavior)):


spear = pd.DataFrame(spear, columns=("corr", "p"))
# get cells with a significant correlation
spear_s = spear.loc[spear["p"]<0.05]
spear_sort = spear_s.sort_values(by=["corr"], ascending=True)

# plot the 9 cells with the highest rho
plt.figure()
for i in range(1,10):
    ax = plt.subplot(3,3,i)
    curr_cell = spear_sort.iloc[-i]
    ax.plot(speed, cell_act[curr_cell.name], ".")
    ax.axhline(y=0, c="black")
    ax.set_title("Cell {:d}, rho={:.2f}, p={:.1e}".format(curr_cell.name, curr_cell["corr"], curr_cell["p"]))
plt.suptitle("Highest corr")

# Example cell: 257
cell_id=783

speed_smooth = scipy.ndimage.gaussian_filter1d(speed, sigma=3)
act_smooth = scipy.ndimage.gaussian_filter1d(cell_act[cell_id], sigma=3)

fig, ax = plt.subplots(2, 1, sharex="all")
ax[0].plot(speed_smooth)
ax[1].plot(act_smooth)

# Avg activity per speed bin
n_bins = 15
bin_borders = np.linspace(np.min(speed_smooth),np.max(speed_smooth), n_bins)
speed_bins = np.digitize(speed_smooth, bin_borders)
act_bins = []
bin_names = []
for i in range(1, n_bins):
    act_bins.append(act_smooth[speed_bins==i])
    bin_names.append([i]*len(act_smooth[speed_bins==i]))

data = pd.DataFrame(dict(signal=np.concatenate(act_bins), bin=np.concatenate(bin_names)))
plt.figure()
sns.stripplot(x="bin", y="signal", data=data)
sns.boxplot(x="bin", y="signal", data=data, fliersize=0)

x_bar = bin_borders
y_bar = [np.mean(x) for x in act_bins]
plt.figure()
plt.bar(x_bar, y_bar)

# Export data for prism
# example traces
start = 5870
end = 6480
n_samples = end-start
x = np.linspace(0, n_samples/30, n_samples)
y_speed = speed_smooth[start:end]
y_act = cell_act[cell_id, start:end]

np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_x.txt", x.T, delimiter='\t', fmt='%.5f')
np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_y_speed.txt", y_speed.T, delimiter='\t', fmt='%.5f')
np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_y_act.txt", y_act.T, delimiter='\t', fmt='%.5f')

# binned data
np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_bin_x.txt", x_bar.T, delimiter='\t', fmt='%.5f')
np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_bin_y.txt", np.array(y_bar).T, delimiter='\t', fmt='%.5f')

# Get 95% confidence interval
std = np.array([np.std(bin) for bin in act_bins])
np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\speed_cell_bin_std.txt", std.T, delimiter='\t', fmt='%.5f')


#%% Speed cell method from Iwase et al. 2020

import scipy.ndimage as nd
import scipy.stats as st
import time

def shuffle(data, sp, n, min_dur, max_dur, fps):
    min_sample, max_sample = np.array([min_dur, max_dur], dtype=int)*fps
    corr_shuf = np.zeros((data.shape[0], n))*np.nan
    for cell in range(data.shape[0]):
        for i in range(n):
            shift = np.random.randint(min_sample, max_sample)
            data_shuf = np.roll(data[cell], shift)
            corr_shuf[cell, i], _ = st.spearmanr(a=data_shuf, b=sp)

    perc1, perc99 = np.percentile(corr_shuf, 1), np.percentile(corr_shuf, 99)
    return perc1, perc99

window = 0.25        # window size in seconds of Gaussian filter
shift = 60           # multiple of fps that should be allowed for temporal bias of speed cells

fps = pcf.cnmf.params.data["fr"]
# apply Gaussian filter (0.5s window)
sd = window*fps
speed_f = nd.gaussian_filter1d(speed, sd)

# filter out periods with running speeds < 2 cm/s
mask = speed_f >= 2
speed_f = speed_f[mask]
act_f = cell_act[:,mask]

# Produce shuffled correlation distribution of ALL cells
p1, p99 = shuffle(cell_act, speed_f, n=100, min_dur=30, max_dur=cell_act.shape[1] / fps - 30, fps=fps)

# Get Spearman's rank correlation for each cell and determine if its a speed cell
spear = np.zeros((len(cell_act), 3))*np.nan
for cell in range(len(cell_act)):
    # Calculate speed score (correlation) of current cell
    corr, p = st.spearmanr(a=cell_act[cell], b=speed_f)
    spear[cell, 0] = corr
    spear[cell, 1] = p*len(cell_act)
    if corr < p1:
        spear[cell, 2] = -1     # negative speed cell (marked as -1)
    elif corr > p99:
        spear[cell, 2] = 1      # positive speed cell (marked as +1)
    else:
        spear[cell, 2] = 0      # no speed cell (marked as 0)

# Get temporal bias of positive speed cells
pos_cells = spear[:,2] == 1
shifts = np.

idx = 0
plt.figure(); plt.plot(speed_f, cell_act[np.where(pos_cells)[0][idx]], ".")
plt.figure(); plt.plot(cell_act[np.where(pos_cells)[0][idx]])

