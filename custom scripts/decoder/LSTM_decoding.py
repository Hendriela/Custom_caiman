#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from standard_pipeline import place_cell_pipeline as pipe

#Import function to get the covariate matrix that includes spike history from previous bins
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

#Import metrics
from Neural_Decoding.metrics import get_R2

#Import decoder functions
from Neural_Decoding.decoders import WienerFilterDecoder
from Neural_Decoding.decoders import DenseNNDecoder
from Neural_Decoding.decoders import LSTMDecoder

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

# Fix position averaging across trial borders
pos_diff = np.where(np.diff(pos) < -10)[0]                   # Get indices of separate trials
pos_fix = pos_diff[np.where(np.diff(pos_diff) == 1)[0] + 1]  # Find indices where pos was averaged across trial borders
pos[pos_fix] = 110                                           # Set these positions to "110"

#%% Preprocessing

# Remove neurons with too low spike rate (< 0.02 Hz)
spike_rate = np.nansum(neural_data, axis=0) / (neural_data.shape[0]/(1/dt))  # Spikes/sec of each neuron
rmv_nrn = np.where(spike_rate < 0.02)                                        # Find neurons with spike rate of < 0.02 Hz
neural_data = np.delete(neural_data, rmv_nrn, 1)                             # Remove those neurons and set as input

# User input variables
bins_before = 4   # How many bins of neural data prior to the output are used for decoding
bins_current = 1  # Whether to use concurrent time bin of neural data
bins_after = 5    # How many bins of neural data after the output are used for decoding

# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
# Function to get the covariate matrix that includes spike history from previous bins
X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

# Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
#Put in "flat" format, so each "neuron / time" is a single feature
X_flat = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

#Set decoding output
y = pos[..., np.newaxis]

# Find indices of separate trials
y_diff = np.where(np.diff(y[:, 0]) < -10)[0] + 1
y_diff = np.append(y_diff, len(y)-1)
trial_idx = []
low_idx = 0
for i in range(len(y_diff)):
    high_idx = y_diff[i]
    trial_idx.append(np.arange(start=low_idx, stop=high_idx))
    low_idx = high_idx
trial_idx[-1] = np.append(trial_idx[-1], trial_idx[-1][-1]+1)

use_fractions = True

if use_fractions:
    # Split into training/testing/validation sets
    # Set what part of data should be part of the training/testing/validation sets
    training_range = [0, 0.6]  # 75% of data for training
    valid_range = [0.6, 0.8]  # 15% of data for validation
    testing_range = [0.8, 1]   # 15% of data for testing

    num_examples = X.shape[0]

    # Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    # This makes it so that the different sets don't include overlapping neural data
    training_set = np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,
                             np.int(np.round(training_range[1]*num_examples))-bins_after)
    testing_set = np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,
                            np.int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set = np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,
                          np.int(np.round(valid_range[1]*num_examples))-bins_after)
else:
    # select longest trial as valid and 2nd longest as test (not first and last trial as they don't have all positions)
    trial_idx_sort = trial_idx[1:-1].copy()
    trial_idx_sort.sort(key=len)
    valid_set = np.concatenate((trial_idx_sort[-1], trial_idx_sort[-2]))
    testing_set = trial_idx_sort[-3]
    diff = np.isin(np.concatenate(trial_idx), np.concatenate((valid_set, testing_set)))
    training_set = np.concatenate(trial_idx)[~diff]

# Get training data
X_train = X[training_set, :, :]
X_flat_train = X_flat[training_set, :]
y_train = y[training_set, :]

# Get testing data
X_test = X[testing_set, :, :]
X_flat_test = X_flat[testing_set, :]
y_test = y[testing_set, :]

# Get validation data
X_valid = X[valid_set, :, :]
X_flat_valid = X_flat[valid_set, :]
y_valid = y[valid_set, :]

# Process covariates
# Z-score "X" inputs.
X_train_mean = np.nanmean(X_train, axis=0)
X_train_std = np.nanstd(X_train, axis=0)
X_train = (X_train-X_train_mean)/X_train_std
X_test = (X_test-X_train_mean)/X_train_std
X_valid = (X_valid-X_train_mean)/X_train_std

#Z-score "X_flat" inputs.
X_flat_train_mean = np.nanmean(X_flat_train, axis=0)
X_flat_train_std = np.nanstd(X_flat_train, axis=0)
X_flat_train = (X_flat_train-X_flat_train_mean)/X_flat_train_std
X_flat_test = (X_flat_test-X_flat_train_mean)/X_flat_train_std
X_flat_valid = (X_flat_valid-X_flat_train_mean)/X_flat_train_std

#Zero-center outputs
y_train_mean = np.mean(y_train, axis=0)
y_train = y_train-y_train_mean
y_test = y_test-y_train_mean
y_valid = y_valid-y_train_mean

#%% Hyperparameter optimization (Hyperopt)


def dnn_evaluate2(params):
    #Put parameters in proper format
    num_units = int(params['num_units'])
    frac_dropout = float(params['frac_dropout'])
    n_epochs = int(params['n_epochs'])
    model_lstm = LSTMDecoder(units=num_units, dropout=frac_dropout, num_epochs=n_epochs) # Define model
    model_lstm.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)   # Get validation set predictions
    return -np.mean(get_R2(y_valid,y_valid_predicted_dnn))  # Return -R2 value of validation set

#%% Wiener Filter (Linear Regression)

#Declare model
model_wf=WienerFilterDecoder()

X_flat_traine = np.nan_to_num(X_flat_train)
X_flat_valide = np.nan_to_num(X_flat_valid)

#Fit model
model_wf.fit(X_flat_traine, y_train)

#Get predictions
y_valid_predicted_wf = model_wf.predict(X_flat_valide)

#Get metric of fit
R2s_wf=get_R2(y_valid, y_valid_predicted_wf)
print('R2s:', R2s_wf)

fig_x = plt.figure()
plt.plot(y_valid[:, 0]+y_train_mean[0], 'b')
plt.plot(y_valid_predicted_wf[:, 0]+y_train_mean[0], 'r')

#%% Run LSTM Decoder
#Declare model
model_lstm = LSTMDecoder(units=100, dropout=.25, num_epochs=10)

#Fit model
model_lstm.fit(X_train, y_train)

#Get predictions
y_valid_predicted_lstm = model_lstm.predict(X_valid)

#Get metric of fit
R2s_lstm=get_R2(y_valid,y_valid_predicted_lstm)
print('R2s:', R2s_lstm)

fig_x = plt.figure()
plt.plot(y_valid[:, 0]+ y_train_mean[0], 'b')
plt.plot(y_valid_predicted_lstm[:, 0]+y_train_mean[0], 'r')

#%% Run Dense Feedforward Neural Network
#Declare model
model_dnn=DenseNNDecoder(units=100, dropout=0.25, num_epochs=10)

#Fit model
model_dnn.fit(X_flat_train, y_train)

#Get predictions
y_valid_predicted_dnn = model_dnn.predict(X_flat_valid)

#Get metric of fit
R2s_dnn = get_R2(y_valid, y_valid_predicted_dnn)
print('R2s:', R2s_dnn)

fig_x = plt.figure()
plt.plot(y_valid[:, 0]+y_train_mean[0], 'b')
plt.plot(y_valid_predicted_dnn[:, 0]+y_train_mean[0], 'r')

#%%

