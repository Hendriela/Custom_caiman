import os
import glob as glob
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('seaborn-darkgrid')

import numpy as np
import scipy.io as sio
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import resample
from scipy.interpolate import interp1d

import os
from os.path import normpath, basename
import glob

import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Input, LSTM,BatchNormalization, LocallyConnected1D, Activation, concatenate
from keras import backend as K


# from helper_scripts.utils import define_model, noiselevels_test_dataset, preprocess_test_dataset
# from helper_scripts.utils import preprocess_groundtruth_artificial_noise, calibrated_ground_truth_artificial_noise
# from helper_scripts.utils_discrete_spikes import divide_and_conquer, fill_up_APs, systematic_exploration, prune_APs # random_motion
from utils import define_model, noiselevels_test_dataset, preprocess_test_dataset
from utils import preprocess_groundtruth_artificial_noise, calibrated_ground_truth_artificial_noise
from utils_discrete_spikes import divide_and_conquer, fill_up_APs, systematic_exploration, prune_APs # random_motion

#%%

def get_noise_levels()