#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import sys
import pickle
import place_cell_pipeline as pipe

#Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

#Import decoder functions
from Neural_Decoding.decoders import NaiveBayesDecoder
from Neural_Decoding.preprocessing_funcs import bin_spikes
from Neural_Decoding.preprocessing_funcs import bin_output
#%% Load data
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2'
data_raw = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved_PR.npy').T
pcf = pipe.load_pcf(root)

#%% Bin data
behavior = np.concatenate(pcf.behavior)

# get universal time stamps for behavior files by adding the last time stamp of the previous trial
behavior = behavior[np.where(behavior[:, 3] == 1)[0]]

# remove values without spiking data (beginning and end of session)
nan_mask = np.where(~np.isnan(data_raw[:, 0]))[0]
behavior = behavior[nan_mask]
data_raw = data_raw[nan_mask]

# find index of new trial by sudden position difference of -120 (track length) (+1 to get idx of next trial)
pos_diff = np.where(np.diff(behavior[:, 1]) < -110)[0] + 1
pos_diff = np.append(pos_diff, len(behavior)-1)  # add index of last trial (doesnt show up during np.diff)

for i in range(len(pos_diff)):
    # remember global indices of this trial
    min_idx = pos_diff[i-1]
    max_idx = pos_diff[i]
    prev_time = behavior[min_idx-1, 0]  # get last time stamp of previous trial
    # add previous time stamp plus 33 ms (frame rate) to the time stamp of the current trial
    behavior[min_idx:max_idx, 0] = behavior[min_idx:max_idx, 0] + prev_time + 0.033
behavior[-1, 0] = behavior[-2, 0] + 0.033   # add the last time stamp manually (skipped by previous loop)

# bin data
dt = .2     # size of time bins in seconds
t_start = behavior[0, 0]
t_end = behavior[-1, 0]
downsample_factor = 1

edges = np.arange(t_start, t_end+dt, dt)  # Get edges of time bins
num_bins = edges.shape[0] - 1  # Number of bins
num_neurons = data_raw.shape[1]  # Number of neurons
neural_data = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
pos = np.empty(num_bins)
vel = np.empty(num_bins)
# Count number of spikes in each bin for each neuron, and put in array
for i in range(num_neurons):
    curr_neur = data_raw[:, i]
    for low_edge in range(num_bins):
        high_edge = low_edge+1
        idx = np.where((edges[low_edge] <= behavior[:, 0]) & (behavior[:, 0] < edges[high_edge]))
        neural_data[low_edge, i] = np.sum(curr_neur[idx])
        pos[low_edge] = np.mean(behavior[idx, 1])
        vel[low_edge] = np.mean(behavior[idx, 4])

#%% Preprocessing

bins_before = 4   # How many bins of neural data prior to the output are used for decoding
bins_current = 1  # Whether to use concurrent time bin of neural data
bins_after = 5    # How many bins of neural data after the output are used for decoding

# Remove neurons with too low spike rate (< 0.02 Hz)
spike_rate = np.nansum(neural_data, axis=0) / (neural_data.shape[0]/(1/dt))  # Spikes/sec of each neuron
rmv_nrn = np.where(spike_rate < 0.02)                                        # Find neurons with spike rate of < 0.02 Hz
X = np.delete(neural_data, rmv_nrn, 1)                                       # Remove those neurons and set as input

# set binned position as decoding output (add a second empty axis if its only one axis)
y = np.vstack((pos+10, np.random.randint(low=10, high=120, size=len(pos)))).T
#y = pos[..., np]

# Number of bins to sum spikes over
N = bins_before + bins_current + bins_after

# Set what part of data should be part of the training/testing/validation sets
training_range = [0, 0.6]  # 75% of data for training
valid_range = [0.6, 0.8]  # 15% of data for validation
testing_range = [0.8, 1]   # 15% of data for testing

# Split data (for Naive Bayes)
# Note that each range has a buffer of "bins_before" bins at the beginning, and "bins_after" bins at the end
# This makes it so that the different sets don't include overlapping neural data
training_set = np.arange(np.int(np.round(training_range[0]*num_bins))+bins_before,
                         np.int(np.round(training_range[1]*num_bins))-bins_after)
testing_set = np.arange(np.int(np.round(testing_range[0]*num_bins))+bins_before,
                        np.int(np.round(testing_range[1]*num_bins))-bins_after)
valid_set = np.arange(np.int(np.round(valid_range[0]*num_bins))+bins_before,
                      np.int(np.round(valid_range[1]*num_bins))-bins_after)

# Get training data
X_train = X[training_set, :]
y_train = y[training_set, :]

# Get testing data
X_test = X[testing_set, :]
y_test = y[testing_set, :]

# Get validation data
X_valid = X[valid_set, :]
y_valid = y[valid_set, :]

# COMBINE DATA ACROSS SPECIFIED BINS
# Get total number of spikes across "bins_before, "bins_current" and "bins_after"
# Initialize matrices for neural data in Naive bayes format
num_nrns = X_train.shape[1]
X_b_train = np.empty([X_train.shape[0] - N + 1, num_nrns])
X_b_valid = np.empty([X_valid.shape[0] - N + 1, num_nrns])
X_b_test = np.empty([X_test.shape[0] - N + 1, num_nrns])

# Below assumes that bins_current = 1 (otherwise alignment will be off by 1 between the spikes and outputs)
# For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
# Do this for the training/validation/testing sets
for i in range(num_nrns):
    X_b_train[:, i] = N * np.convolve(X_train[:, i], np.ones((N,)) / N,
                                      mode='valid')  # Convolving w/ ones is a sum across those N bins
    X_b_valid[:, i] = N * np.convolve(X_valid[:, i], np.ones((N,)) / N, mode='valid')
    X_b_test[:, i] = N * np.convolve(X_test[:, i], np.ones((N,)) / N, mode='valid')

# Make integer format (round probabilities
X_b_train = np.round(X_b_train).astype(int)
X_b_valid = np.round(X_b_valid).astype(int)
X_b_test = np.round(X_b_test).astype(int)

# Make y's aligned w/ X's
# e.g. remove the first y if we are using 1 bin before, and remove the last y if we are using 1 bin after
if bins_before > 0 and bins_after > 0:
    y_train = y_train[bins_before:-bins_after, :]
    y_valid = y_valid[bins_before:-bins_after, :]
    y_test = y_test[bins_before:-bins_after, :]

if bins_before > 0 and bins_after == 0:
    y_train = y_train[bins_before:, :]
    y_valid = y_valid[bins_before:, :]
    y_test = y_test[bins_before:, :]

#%% Run Decoder

# Declare model

# The parameter "encoding_model" can either be linear or quadratic, although additional encoding models could be added.
# The parameter "res" is the number of bins used (resolution) for decoding predictions
# So if res=100, we create 100 bins going from the minimum to maximum of the output variable (position)
# The prediction the decoder makes will be a value on that grid

model_nb = NaiveBayesDecoder(encoding_model='linear', res=120)

#Fit model
model_nb.fit(X_b_train, y_train)










#%%

