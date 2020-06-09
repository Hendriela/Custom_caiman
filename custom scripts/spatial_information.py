from standard_pipeline import place_cell_pipeline as pipe
import numpy as np
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, shapiro
from statannot import add_stat_annotation

def bin_activity_to_vr(neuron_traces, n_bins, bin_frame_count):
    """
    Takes bin_frame_count and bins the dF/F traces of all trials of one neuron to achieve a uniform trial length
    for place cell analysis. Procedure for every trial: Algorithm goes through every bin and extracts the
    corresponding frames according to bin_frame_count.
    :param neuron_traces: list of arrays that contain the dF/F traces of a neuron. From self.session[n_neuron]
    :param n_bins: int, number of bins the trace should be split into
    :param bin_frame_count: np.array containing the number of frames in each bin
    :return: bin_activity (list of trials), bin_avg_activity (1D array) for this neuron
    """

    bin_act = np.zeros((n_trials, n_bins))
    for trial in range(n_trials):
        curr_trace = neuron_traces[trial]
        curr_bins = bin_frame_count[:, trial]
        curr_act_bin = np.zeros(n_bins)
        for bin_no in range(n_bins):
            # extract the trace of the current bin from the trial trace
            if len(curr_trace) > curr_bins[bin_no]:
                trace_to_avg, curr_trace = curr_trace[:curr_bins[bin_no]], curr_trace[curr_bins[bin_no]:]
            elif len(curr_trace) == curr_bins[bin_no]:
                trace_to_avg = curr_trace
            else:
                raise Exception('Something went wrong during binning...')
            if trace_to_avg.size > 0:
                curr_act_bin[bin_no] = np.nanmean(trace_to_avg)
            else:
                curr_act_bin[bin_no] = 0
        bin_act[trial] = curr_act_bin

    # Get average activity across trials of this neuron for every bin
    bin_avg_act = np.nanmean(bin_act, axis=0)

    return bin_act, bin_avg_act


# working standard functions


#%% manual spatial information with Peters data
peter = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved_PR.npy')
decon = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved.npy')
dff = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\dff.npy')
transients = peter
#%% split trace into trials
def spatial_info_per_trial():
    root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2'
    n_neuron = transients.shape[0]
    session = list(np.zeros(n_neuron))
    frame_list = [1369, 1659, 1207, 1191, 1077, 1148, 1118, 920]
    n_trials = len(frame_list)

    for neuron in range(n_neuron):
        curr_neuron = list(np.zeros(n_trials))  # initialize neuron-list
        session_trace = transients[neuron]  # temp-save DF/F session trace of this neuron

        for trial in range(n_trials):
            # extract trace of the current trial from the whole session
            if len(session_trace) > frame_list[trial]:
                trial_trace, session_trace = session_trace[:frame_list[trial]], session_trace[frame_list[trial]:]
                curr_neuron[trial] = trial_trace  # save trial trace in this neuron's list
            elif len(session_trace) == frame_list[trial]:
                curr_neuron[trial] = session_trace
            else:
                print('Error in PlaceCellFinder.split_traces()')
        session[neuron] = curr_neuron  # save data from this neuron to the big session list

    # import behavior and align traces
    behavior = []
    is_faulty = False
    count = 0
    n_bins = 40
    bin_width = 10

    for step in os.walk(root):
        folder_list = step[1]
        break
    trial_list = [os.path.join(root, folder) for folder in folder_list]

    for trial in trial_list:
        path = glob(trial + '//merged_behavior*.txt')
        if len(path) == 1:
            behavior.append(np.loadtxt(path[0], delimiter='\t'))
            count_list = int(frame_list[count])
            count_imp = int(np.sum(behavior[-1][:, 3]))
            if count_imp != count_list:
                print(f'Contradicting frame counts in trial {trial} (no. {count}):\n'
                      f'\tExpected {count_list} frames, imported {count_imp} frames...')
                is_faulty = True
            count += 1
        else:
            print(f'Couldnt find behavior file at {trial}')
    if is_faulty:
        raise Exception('Frame count mismatch detected, stopping analysis.')

    bin_frame_count = np.zeros((n_bins, n_trials), 'int')
    for trial in range(len(behavior)):  # go through vr data of every trial and prepare it for analysis

        # bin data in distance chunks
        bin_borders = np.linspace(-10, 110, n_bins)
        idx = np.digitize(behavior[trial][:, 1], bin_borders)  # get indices of bins

        # check how many frames are in each bin
        for i in range(n_bins):
            bin_frame_count[i, trial] = np.sum(behavior[trial][np.where(idx == i + 1), 3])

    # double check if number of frames are correct
    for i in range(len(frame_list)):
        frame_list_count = frame_list[i]
        if frame_list_count != np.sum(bin_frame_count[:, i]):
            print(
                f'Frame count not matching in trial {i + 1}: Frame list says {frame_list_count}, import says {np.sum(bin_frame_count[:, i])}')

    # bin the activity for every neuron to the VR position, construct bin_activity and bin_avg_activity
    bin_activity = []
    bin_avg_activity = np.zeros((n_neuron, n_bins))
    for neuron in range(n_neuron):
        neuron_bin_activity, neuron_bin_avg_activity = bin_activity_to_vr(session[neuron], n_bins, bin_frame_count)
        bin_activity.append(neuron_bin_activity)
        bin_avg_activity[neuron, :] = neuron_bin_avg_activity


    #%% spatial information
    # calculate average fraction of time spent in this field via the new bin_frame_count
    fractions = np.zeros((bin_frame_count.shape[0], bin_frame_count.shape[1]))
    for trial in range(bin_frame_count.shape[1]):
        # divide every bin_frame_count in the trial by the total amount of frames in that bin
        fractions[:, trial] = bin_frame_count[:, trial] / np.sum(bin_frame_count[:, trial])
    avg_fractions = np.mean(fractions, axis=1)

    # get a value for spatial information for each cell
    spatial_info = np.zeros(n_neuron)
    for cell in range(n_neuron):
        curr_trace = bin_avg_activity[cell]  # this is the binned activity of the current cell
        dff_tot = np.mean(curr_trace)  # this is the total dF/F averaged across all bins
        bin_si = np.zeros(bin_frame_count.shape[0])  # initialize array that holds SI value for each bin
        for i in range(bin_frame_count.shape[0]):
            # apply the SI formula to every bin
            bin_si[i] = curr_trace[i] * np.log(curr_trace[i] / dff_tot) * avg_fractions[i]
        spatial_info[cell] = np.sum(bin_si)

#%% Spatial information not split by trials (Peters data)

root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2'
pcf = pipe.load_pcf(root)
#data_all = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved_PR.npy').T
data_all = pcf.cnmf.estimates.F_dff.T
position_all = []
for trial in range(len(pcf.behavior)):
    position_all.append(pcf.behavior[trial][np.where(pcf.behavior[trial][:, 3] == 1), 1])
position_all = np.hstack(position_all).T

#%% remove samples where the mouse was stationary (less than 30 movement per frame)
behavior_masks = []
for trial in pcf.behavior:
    behavior_masks.append(np.ones(int(np.sum(trial[:, 3])), dtype=bool))    # bool list for every frame of that trial
    frame_idx = np.where(trial[:, 3] == 1)[0]                               # find sample_idx of all frames
    for i in range(len(frame_idx)):
        if i != 0:
            if np.sum(trial[frame_idx[i-1]:frame_idx[i], 4]) > -30:    # make the index of the current frame False if
                behavior_masks[-1][i] = False                          # the mouse didnt move much during the frame
        else:
            if trial[0, 4] > -30:
                behavior_masks[-1][i] = False

trial_lengths = [int(np.sum(trial)) for trial in behavior_masks]

behavior_mask = np.hstack(behavior_masks)
data = data_all[behavior_mask]
position = position_all[behavior_mask]

# remove data points where decoding was nan (beginning and end)
nan_mask = np.isnan(data[:, 0])
data = data[~nan_mask]
position = position[~nan_mask]

# bin data into n_bins, get mean event rate per bin
n_bins = 60
bin_borders = np.linspace(-10, 110, n_bins)
idx = np.digitize(position, bin_borders)  # get indices of bins

# get fraction of bin occupancy
unique_elements, counts_elements = np.unique(idx, return_counts=True)
bin_freq = np.array([x/np.sum(counts_elements) for x in counts_elements])

# get mean spikes/s for each bin
bin_mean_act = np.zeros((n_bins, data.shape[1]))
for bin_nr in range(n_bins):
    curr_bin_idx = np.where(idx == bin_nr+1)[0]
    bin_act = data[curr_bin_idx]
    bin_mean_act[bin_nr, :] = np.sum(bin_act, axis=0)/(bin_act.shape[0]/30)
    #bin_mean_act[bin_nr, :] = np.mean(bin_act, axis=0)
total_firing_rate = np.sum(data, axis=0)/(data.shape[0]/30)
#total_firing_rate = np.mean(data, axis=0)

# calculate spatial information content
spatial_info = np.zeros(len(total_firing_rate))
for cell in range(len(total_firing_rate)):
    curr_trace = bin_mean_act[:, cell]
    tot_act = total_firing_rate[cell]  # this is the total dF/F averaged across all bins
    bin_si = np.zeros(n_bins)  # initialize array that holds SI value for each bin
    for i in range(n_bins):
        # apply the SI formula to every bin
        if curr_trace[i] <= 0 or tot_act <= 0:
            bin_si[i] = 0
        else:
            bin_si[i] = curr_trace[i] * np.log2(curr_trace[i] / tot_act) * bin_freq[i]
    spatial_info[cell] = np.sum(bin_si)
#%% plot spatial info
pc_label = []
pc_idx = [x[0] for x in pcf.place_cells]
pc_rej = [x[0] for x in pcf.place_cells_reject]
for cell_idx in range(len(spatial_info)):
    if cell_idx in pc_idx:
        pc_label.append('yes')
    # elif cell_idx in pc_rej:
    #     pc_label.append('rejected')
    else:
        pc_label.append('no')

# add sample size to labels
no_count = pc_label.count('no')
acc_count = pc_label.count('yes')
rej_count = pc_label.count('rejected')
pc_label = [f'no (n={no_count})' if x == 'no' else x for x in pc_label]
pc_label = [f'yes (n={acc_count})' if x == 'yes' else x for x in pc_label]
pc_label = [f'rejected (n={rej_count})' if x == 'rejected' else x for x in pc_label]

df = pd.DataFrame(data={'SI': spatial_info, 'Place cell': pc_label, 'dummy': np.zeros(len(spatial_info))})

file_dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\spatial information'
mouse = pcf.params['mouse']
session = pcf.params['session']
file_name = f'spatial_info_{mouse}_{session}_no_rest.png'
fig = plt.figure()
plt.title(f'Spatial info {mouse}, {session}')
ax = sns.barplot(x='Place cell', y='SI', data=df)
results = add_stat_annotation(ax, data=df, x='Place cell', y='SI', text_format='full',
                              box_pairs=[(f'yes (n={acc_count})', f'no (n={no_count})')], test='Mann-Whitney', verbose=2)
sns.stripplot(x='Place cell', y='SI', data=df, linewidth=1)

#%% check diff between place cells and non-place cells
pc_mask = np.zeros(len(spatial_info), dtype=bool)
pc_mask[pc_idx] = True
pc_si = spatial_info[pc_mask]
non_pc_si = spatial_info[~pc_mask]
# normality test
w, p_pc = shapiro(pc_si)
w, p_nonpc = shapiro()
t_2, p_2 = ttest_ind(non_pc_si, pc_si, equal_var=False)


#%%

