#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys

#Import function to get the covariate matrix that includes spike history from previous bins
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
from Neural_Decoding.preprocessing_funcs import bin_spikes
from Neural_Decoding.preprocessing_funcs import bin_output

#Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

#Import decoder functions
from Neural_Decoding.decoders import WienerCascadeDecoder
from Neural_Decoding.decoders import WienerFilterDecoder
from Neural_Decoding.decoders import DenseNNDecoder
from Neural_Decoding.decoders import SimpleRNNDecoder
from Neural_Decoding.decoders import GRUDecoder
from Neural_Decoding.decoders import LSTMDecoder
from Neural_Decoding.decoders import XGBoostDecoder
from Neural_Decoding.decoders import SVRDecoder

#%%
folder = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\neural_decoding_example'

data=io.loadmat(folder+'\hc_data_raw.mat')
spike_times=data['spike_times'] #Load spike times of all neurons
pos=data['pos'] #Load x and y positions
pos_times=data['pos_times'][0] #Load times at which positions were recorded

spike_times=np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]=np.squeeze(spike_times[i])

dt=.2 #Size of time bins (in seconds)
t_start=pos_times[0] #Time to start extracting data - here the first time position was recorded
t_end=5608 #pos_times[-1] #Time to finish extracting data - when looking through the dataset, the final position was recorded around t=5609, but the final spikes were recorded around t=5608
downsample_factor=1 #Downsampling of output (to make binning go faster). 1 means no downsampling.

edges=np.arange(t_start,t_end,dt)

#Bin neural data using "bin_spikes" function
neural_data=bin_spikes(spike_times,dt,t_start,t_end)

#Bin output (position) data using "bin_output" function
pos_binned=bin_output(pos,pos_times,dt,t_start,t_end,downsample_factor)

with open(folder+'\example_data_hc.pickle', 'rb') as f:
     neural_data_pre, pos_binned = pickle.load(f, encoding='latin1')

model = 'Wiener'
#%% preprocessing

bins_before = 4   # How many bins of neural data prior to the output are used for decoding
bins_current = 1  # Whether to use concurrent time bin of neural data
bins_after = 5    # How many bins of neural data after the output are used for decoding

#Remove neurons with too few spikes in HC dataset
nd_sum = np.nansum(neural_data, axis=0)             # Total number of spikes of each neuron
rmv_nrn = np.where(nd_sum < 100)                    # Find neurons who have less than 100 spikes total
neural_data = np.delete(neural_data, rmv_nrn, 1)    # Remove those neurons

# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
# Function to get the covariate matrix that includes spike history from previous bins
X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

# Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
# Put in "flat" format, so each "neuron / time" is a single feature
X_flat = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

#Set decoding output
y = pos_binned

#Remove time bins with no output (y value)
rmv_time = np.where(np.isnan(y[:, 0]) | np.isnan(y[:, 1]))      # Find time bins with no output
X = np.delete(X, rmv_time, 0)                                   # Remove those time bins from X
X_flat = np.delete(X_flat, rmv_time, 0)                         # Remove those time bins from X_flat
y = np.delete(y, rmv_time, 0)                                   # Remove those time bins from y

# Set what part of data should be part of the training/testing/validation sets
# Note that there was a long period of no movement after about 80% of recording, so I did not use this data.
training_range = [0, 0.5]
valid_range = [0.5, 0.65]
testing_range = [0.65, 0.8]

# SPLIT DATA
num_examples = X.shape[0]

# Note that each range has a buffer of "bins_before" bins at the beginning, and "bins_after" bins at the end
# This makes it so that the different sets don't include overlapping neural data
training_set = np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,
                         np.int(np.round(training_range[1]*num_examples))-bins_after)
testing_set = np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,
                        np.int(np.round(testing_range[1]*num_examples))-bins_after)
valid_set = np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,
                      np.int(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train = X[training_set, :, :]
X_flat_train = X_flat[training_set, :]
y_train = y[training_set, :]

#Get testing data
X_test = X[testing_set, :, :]
X_flat_test = X_flat[testing_set, :]
y_test = y[testing_set, :]

#Get validation data
X_valid = X[valid_set, :, :]
X_flat_valid = X_flat[valid_set, :]
y_valid = y[valid_set, :]

# Process covariates: normalize (z-score) the inputs and zero-center the outputs.
# Parameters for z-scoring are determined only on the training set but applied to all sets.
#Z-score "X" inputs.
X_train_mean=np.nanmean(X_train,axis=0)
X_train_std=np.nanstd(X_train,axis=0)
X_train=(X_train-X_train_mean)/X_train_std
X_test=(X_test-X_train_mean)/X_train_std
X_valid=(X_valid-X_train_mean)/X_train_std

#Z-score "X_flat" inputs.
X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
X_flat_train_std=np.nanstd(X_flat_train,axis=0)
X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

#Zero-center outputs
y_train_mean=np.mean(y_train,axis=0)
y_train=y_train-y_train_mean
y_test=y_test-y_train_mean
y_valid=y_valid-y_train_mean

#%% Run Decoders: Wiener Filter (Linear Regression)

#Declare model
model_wf=WienerFilterDecoder()

#Fit model
model_wf.fit(X_flat_train,y_train)

#Get predictions
y_valid_predicted_wf=model_wf.predict(X_flat_valid)

#Get metric of fit
R2s_wf=get_R2(y_valid, y_valid_predicted_wf)
print('R2s:', R2s_wf)

# Plotting
fig_x_wf = plt.figure()
plt.plot(y_valid[:, 0]+y_train_mean[0], 'b')
plt.plot(y_valid_predicted_wf[:, 0]+y_train_mean[0], 'r')


#%%

