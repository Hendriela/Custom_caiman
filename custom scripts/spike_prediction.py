import numpy as np
from scipy.ndimage.filters import gaussian_filter

import os
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import backend as K


def get_noise_levels(traces, framerate):
    """
    Calculates noise level for each neuron of an array of dF/F calcium traces (one recording session).
    :param traces: np.array with shape (#frames, #neurons) containing dF/F calcium trace
    :param framerate: int, frame rate of recording
    :return: np.array with shape (#neurons, ) containing one noise level value of this session for each neuron
    """

    nb_neurons = traces.shape[1]

    noise_levels = np.nan * np.zeros((nb_neurons,))
    for neuron in range(nb_neurons):
        noise_levels[neuron] = np.nanmedian(np.abs(np.diff(traces[:, neuron]))) / np.sqrt(framerate)

    return noise_levels


def preprocess_test_dataset(dF_traces, before_frac, windowsize, after_frac):
    """
    Creates a large matrix X that contains for each timepoint of each calcium trace a vector of
    length 'windowsize' around the timepoint.
    :param dF_traces: np.array with shape (#timepoints, #neurons) containing dF/F traces of one or more sessions
    :return: np.array with shape (#neurons, #timepoints, #windowsize)
    """
    before = int(before_frac * windowsize)
    after = int(after_frac * windowsize)

    nb_neurons = dF_traces.shape[1]
    nb_timepoints = dF_traces.shape[0]

    X = np.nan * np.zeros((nb_neurons, nb_timepoints, windowsize,))

    for neuron in range(nb_neurons):
        for timepoint in range(nb_timepoints - windowsize):
            X[neuron, timepoint + before, :] = dF_traces[timepoint:(timepoint + before + after), neuron]

    return X


def predict_spikes(data, params=None, thresh=True, plot_noise_levels=False, verbose=False):
    """
    Perform Peters spike prediction on a single session of dF/F traces.
    :param data: np.array, shape (#neurons, #timestamps) of dF/F traces, e.g. directly from cnmf.estimates.F_dff
    :param params: dict, containing all parameters that the spike prediction algorithm needs. If None, take defaults.
    :param thresh: bool flag whether prediction should be thresholded (avoids spike overestimation of noise)
    :param plot_noise_levels: bool flag whether noise levels of the data should be plotted as a histogram
    :param verbose: bool flag whether to print out progress messages during analysis
    :return: np.array with shape (#neurons, #timestamps) containing summable spike predictions for every neuron
    """

    default_params = {'smoothing': 0.2,     # std of Gaussian smoothing in time (sec) [default 0.2] #todo way to get smoothing from model?
                      'ensemble_size': 5,   # use ensemble learning [5]
                      'frame_rate': 30,     # frame/sampling rate of the recording in Hz [default 30]
                      'pretrained_model_folder': (r'C:\Users\hheise\PycharmProjects\Caiman\Calibrated-inference-of-spiking-master'
                                                  r'\pretrained_models\all_good_datasets_30Hz_ensemblesize5'),
                      # Determines how the time window used as input is positioned around the actual time point
                      'windowsize': 64,     # default 64 time points
                      'before_frac': 0.5,   # default 0.5 and 0.5 (window symmetrically over time point)
                      'after_frac': 0.5}

    # If params dict is not provided, use default parameters
    if params is None:
        params = default_params
    # If a dict is provided, check that all parameters are provided, and use defaults in case any are missing
    else:
        for key in default_params.keys():
            if key not in params.keys():
                params[key] = default_params[key]

    # Transform data structure because Peter's algorithm works with an array of shape (#timestamps, #neurons)
    data = data.T

    ## Compute noise level of all neurons of the dataset
    noise_levels_all = get_noise_levels(data, params['frame_rate'])
    percent99 = np.percentile(noise_levels_all, 99)

    if plot_noise_levels:
        # calculate additional noise percentiles required for histogram plotting
        percent999 = np.percentile(noise_levels_all, 99.9)
        percent1 = np.percentile(noise_levels_all, 1)
        # plot histogram of noise level distribution
        plt.figure(1121)
        plt.hist(noise_levels_all, stacked=True, bins=300)
        plt.plot([percent99, percent99], [0, 1])
        plt.plot([percent1, percent1], [0, 1])
        plt.ylim([0, 1])
        plt.xlim([0, percent999])

    if np.ceil(percent99) > 1:
        noise_levels_model = np.arange(2, np.ceil(percent99) + 1)
    else:
        # if the noise level is <1, manually choose models with noise of 2 and 3 to be sure to capture everything
        noise_levels_model = np.arange(2, 3 + 1)
    nb_noise_levels = len(noise_levels_model)

    ## Load pretrained model
    set_of_models = [[None] * params['ensemble_size'] for _ in range(nb_noise_levels)]

    # Load every pre-trained ensemble model for every noise level present in the dataset
    for noise_level_index, noise_level in enumerate(noise_levels_model):
        for ensemble in range(params['ensemble_size']):
            if verbose:
                print('Loading model ' + str(ensemble + 1) + ' with noise level ' + str(noise_level))
            model_directory = os.path.join(params['pretrained_model_folder'],
                                           'Model_noise_' + str(int(noise_level)) + '_' + str(ensemble) + '.h5')
            set_of_models[noise_level_index][ensemble] = load_model(model_directory)

    ## Process test data and predict spikes
    XX = preprocess_test_dataset(data, params['before_frac'], params['windowsize'], params['after_frac'])

    Y_predict = np.zeros((XX.shape[0], XX.shape[1]))

    for model_noise_index, model_noise in enumerate(noise_levels_model):

        if verbose:
            print('Predictions for noise level ' + str(int(model_noise)))

        # Find indices of neurons with a given noise level ('model_noise')
        if model_noise == noise_levels_model[-1]:  # Highest noise bin (or even higher)
            neurons_ixs = np.where(noise_levels_all >= noise_levels_model[-1] - 1)[0]
        elif model_noise == noise_levels_model[0]:
            neurons_ixs = np.where(noise_levels_all < model_noise)[0]
        else:
            neurons_ixs = np.where((noise_levels_all < model_noise) & (noise_levels_all >= model_noise - 1))[0]

        calcium_this_noise = XX[neurons_ixs, :, :]  # / 100  (division by 100 if dF/F input was in %)
        calcium_this_noise = np.reshape(calcium_this_noise, (calcium_this_noise.shape[0] * calcium_this_noise.shape[1],
                                                             calcium_this_noise.shape[2]))

        for ensemble in range(params['ensemble_size']):
            prediction = set_of_models[model_noise_index][ensemble].predict(np.expand_dims(calcium_this_noise, axis=2),
                                                                            batch_size=4096)
            prediction = np.reshape(prediction, (len(neurons_ixs), XX.shape[1]))
            Y_predict[neurons_ixs, :] += prediction / params['ensemble_size']

    # NaN for first and last datapoints, for which no predictions can be made
    Y_predict[:, 0:int(params['before_frac'] * params['windowsize'])] = np.nan
    Y_predict[:, -int(params['after_frac'] * params['windowsize']):] = np.nan
    Y_predict[Y_predict == 0] = np.nan

    # Enforce non-negative spike prediction values
    Y_predict[Y_predict < 0] = 0

    # My edit: substitute NaNs with 0, otherwise Caiman cant save the array in its hdf5 (cant handle nans)
    # Y_predict = np.nan_to_num(Y_predict)

    if thresh:
        # Put threshold on prediction to avoid overestimation of noise
        gauss = np.zeros((100, 1))                          # Initiate an array
        gauss[50] = 1                                       # Let the array sum to one
        # Apply gaussian filter to create theoretical kernel for a single spike
        gauss = gaussian_filter(gauss, sigma=params['frame_rate'] * params['smoothing'])
        params['thresh'] = np.max(gauss)/np.exp(1)          # Threshold is the peak of one spike div. by exponential factor
        Y_predict[Y_predict < params['thresh']] = 0

    # Clear Keras models from memory (otherwise, they accumulate and slow down things)
    K.clear_session()

    return Y_predict
