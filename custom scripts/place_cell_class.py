#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Hendrik Heiser
# created on 11 October 2019
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import glob
from math import ceil, floor
import os
import re
from behavior_import import progress
from performance_check import is_session_novel
from ScanImageTiffReader import ScanImageTiffReader
import gui_without_movie as gui
from caiman.utils import visualization
import pandas as pd
from statannot import add_stat_annotation
from multisession_registration import draw_single_contour
from scipy.ndimage.filters import gaussian_filter1d

#todo: create a class "data" that includes all data trace arrays (session, session_trans, bin_act, bin_avg_act) and
# is initialized twice, once for dF/F data and once for spike probabilities

class PlaceCellFinder:
    """
    Class that holds all the data, parameters and results to perform place cell analysis.
    The analysis steps and major parameters are mainly adapted from Dombeck2007/2010, Hainmüller2018 and Koay2019.
    """
    def __init__(self, cnmf, param_dict=None):
        """
        Constructor of the PCF class

        PlaceCellFinder objects are organized in two major parts:
        - Actual data like the cnmf object, dF/F traces or binned data is stored in individual attributes.
        - Initial analysis parameters or parameters that are calculated during the analysis pipeline are stored in the
          params dictionary.

        PCF objects have to be initialized with the raw calcium data (cnmf object) and can already be initialized with
        parameters in the params dictionary. Other parameters can be set later, and all other attributes are results
        from analysis steps and will be filled during the pipeline.

        :param cnmf: CNMF object that holds the raw calcium data
        :param param_dict: dictionary that holds all parameters. All keys that are not initialized get default value.
        """
        self.cnmf = cnmf                # CNMF object that you obtain from the CaImAn pipeline
        self.session = None             # List of neurons containing dF/F traces ordered by neurons and trials
        self.session_spikes = None      # List of neurons containing spike probabilities ordered by neurons and trials
        self.session_trans = None       # Same as session, just with significant transients only (rest is 0)
        self.behavior = None            # Array containing behavioral data and frame time stamps (for alignment)
        self.bin_activity = None        # Same structure as session, but binned activity normalized to VR position
        self.bin_avg_activity = None    # List of neurons and their binned activity averaged across trials
        self.bin_spike_rate = None      # Same structure as session, but binned spike rate normalized to VR position
        self.bin_avg_spike_rate = None  # List of neurons and their binned spike rate averaged across trials
        self.place_cells = []           # List of accepted place cells. Stored as tuples with (neuron_id,
                                        # place_field_bins, p-value)
        self.place_cells_reject = []    # List of place cells that passed all criteria, but with p > 0.05.

        # noinspection PyDictCreation
        self.params = {'root': None,            # main directory of that session
                       'trial_list': None,  # list of trial folders in this session
                    # The following parameters can be provided, but reset to default values if not
                       'trans_length': 0.5,     # minimum length in seconds of a significant transient
                       'trans_thresh': 4,       # factor of sigma above which a transient is significant; int or
                                                # tuple of int (for different start and end of transients)
                       'bin_length': 5,         # length in cm VR distance of each bin in which to group the dF/F traces
                       'bin_window_avg': 3,     # sliding window of bins (left and right) for trace smoothing
                       'bin_base': 0.25,        # fraction of lowest bins that are averaged for baseline calculation
                       'place_thresh': 0.25,    # threshold of being considered for place fields, calculated
                                                # from difference between max and baseline dF/F
                       'min_pf_size': 15,       # minimum size in cm for a place field (should be 15-20 cm)
                       'fluo_infield': 7,       # factor above which the mean DF/F in the place field should lie compared to outside the field
                       'trans_time': 0.2,       # fraction of the (unbinned!) signal while the mouse is located in
                                                # the place field that should consist of significant transients
                       'split_size': 50,        # size in frames of bootstrapping segments
                       'track_length': 400,     # length in cm of the VR corridor track
                    # The following parameters are calculated during analysis and do not have to be set by the user
                       'frame_list': None,      # list of number of frames in every trial in this session
                       'n_neuron': None,        # number of neurons that were detected in this session
                       'n_trial': None,         # number of trials in this session
                       'sigma': None,           # array[n_neuron x n_trials], noise level (from FWHM) of every trial
                       'bin_frame_count': None, # array[n_bins x n_trials], number of frames averaged in each bin
                       'place_results': None,   # array[n_neuron x criteria] that stores place cell finder results with
                                                # order pre_screen - bin_size - dF/F - transients - p<0.05
                       'mouse': None,
                       'session': None}

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        # implement user-given parameters
        for key in param_dict.keys():
            if key in self.params.keys():
                self.params[key] = param_dict[key]
            elif key == 'min_pf_size_cm':
                self.params['min_pf_size'] = param_dict[key]
            else:
                raise Exception(f'Parameter {key} was not recognized!')

        if self.params['root'] is None:
            raise Exception(f'Essential parameter root has not been provided upon initialization.')
        if self.params['trial_list'] is None:
            for step in os.walk(self.params['root']):
                folder_list = step[1]
                break
            self.params['trial_list'] = [os.path.join(self.params['root'], folder) for folder in folder_list]
        self.params['trial_list'].sort(key=natural_keys)

        # calculate track_length dependent binning parameters
        if self.params['track_length'] % self.params['bin_length'] == 0:
            self.params['n_bins'] = int(self.params['track_length'] / self.params['bin_length'])
        else:
            raise Exception('Bin_length has to be a divisor of track_length!')
        self.params['min_bin_size'] = int(ceil(self.params['min_pf_size'] / self.params['bin_length']))

        # find directories, files and frame counts
        self.params['frame_list'] = []
        for trial in self.params['trial_list']:
            if len(glob.glob(trial+'//*.mmap')) == 1:
                self.params['frame_list'].append(int(glob.glob(trial+'//*.mmap')[0].split('_')[-2]))
            elif len(glob.glob(trial+'//*.tif')) == 1:
                with ScanImageTiffReader(glob.glob(trial+'//*.tif')[0]) as tif:
                    frame_count = tif.shape()[0]
                self.params['frame_list'].append(frame_count)
            else:
                print(f'No movie files found at {trial}!')

        # find mouse number, session and network
        try:
            self.params['mouse'] = self.params['root'].split(os.sep)[-3]
            if len(self.params['mouse']) > 3:
                self.params['mouse'] = self.params['mouse'][-3:]
            self.params['session'] = self.params['root'].split(os.sep)[-2]
            self.params['network'] = self.params['root'].split(os.sep)[-1]
        except IndexError:
            self.params['mouse'] = self.params['root'].split('/')[-3]
            self.params['session'] = self.params['root'].split('/')[-2]
            self.params['network'] = self.params['root'].split('/')[-1]

        # get regions of reward zones
        if is_session_novel(self.params['root']):
            zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
            self.params['novel'] = True
        else:
            zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])
            self.params['novel'] = False

        # Transform coordinates from VR-coordinates to bin indices
        zone_borders = zone_borders + 10                                # Change scaling from -10-110 to 0-120 VR coords
        zone_borders = zone_borders / (120 / self.params['n_bins'])     # Apply scale from VR coordinates to bins
        zone_length = int(np.round(zone_borders[0, 1] - zone_borders[0, 0]))    # Get length of reward zones
        zone_borders[:, 0] = np.array(zone_borders[:, 0], dtype=int)            # Round down first zone bin
        zone_borders[:, 1] = zone_borders[:, 0] + zone_length                   # Add RZ length to first bin idx
        self.params['zone_borders'] = np.array(zone_borders, dtype=int)         # Transform to int and save param

    def change_param(self, new_params):
        """
        :param new_params: Parameter(s) that should be changed in dict form (key: new_value)
        """
        changed_params = []
        if type(new_params) != dict:
            print('To change a parameter, give the key and the new value in dict form!')
            return
        for key in new_params.keys():
            if self.params[key] != new_params[key]:
                self.params[key] = new_params[key]
                changed_params.append(key)

        # check if any further parameters have to be re-calculated
        if any(key in changed_params for key in ['track_length', 'bin_length', 'min_pf_size']):
            if self.params['track_length'] % self.params['bin_length'] == 0:
                self.params['n_bins'] = int(self.params['track_length'] / self.params['bin_length'])
            else:
                raise Exception('Bin_length has to be a divisor of track_length!')
            self.params['min_bin_size'] = int(ceil(self.params['min_pf_size'] / self.params['bin_length']))
            if self.session is not None:
                self.import_behavior_and_align_traces()

    def save(self, file_name='pcf_results', overwrite=False):
        """
        Saves PCF object as a pickled file to the root directory.
        :param file_name: str; name of the saved file, defaults to 'pcf_results'
        :param overwrite: bool, overwrites files automatically if there is one
        :return:
        """

        if '.' not in file_name:
            save_path = os.path.join(self.params['root'], file_name + '.pickle')
        else:
            save_path = os.path.join(self.params['root'], file_name)

        if os.path.isfile(save_path) and not overwrite:
            answer = None
            while answer not in ("y", "n", 'yes', 'no'):
                answer = input(f"File [...]{save_path[-40:]} already exists!\nOverwrite? [y/n] ")
                if answer == "yes" or answer == 'y':
                    print('Saving...')
                    with open(save_path, 'wb') as file:
                        self.cnmf.dview = None
                        pickle.dump(self, file)
                    print(f'PCF results successfully saved at {save_path}')
                    return save_path
                elif answer == "no" or answer == 'n':
                    print('Saving cancelled.')
                    return None
                else:
                    print("Please enter yes or no.")
        else:
            print('Saving...')
            with open(save_path, 'wb') as file:
                self.cnmf.dview = None
                pickle.dump(self, file)
            print(f'PCF results successfully saved at {save_path}')

    def split_traces_into_trials(self):
        """
        First function to call in the "normal" pipeline.
        Takes raw, across-trial DF/F traces from the CNMF object and splits it into separate trials using the frame
        counts provided in frame_list.
        It returns a "session" list of all neurons. Each neuron itself is a list of 1D arrays that hold the DF/F trace
        for each trial in that session. "Session" can thus be indexed as session[number_neurons][number_trials].

        :return: Updated PCF object with the session list
        """
        if self.params['frame_list'] is not None:
            frame_list = self.params['frame_list']
            n_trials = len(frame_list)
        else:
            raise Exception('You have to provide frame_list before continuing the analysis!')

        data = self.cnmf.estimates.F_dff
        spikes = self.cnmf.estimates.spikes
        n_neuron = data.shape[0]
        session = list(np.zeros(n_neuron))
        session_spikes = list(np.zeros(n_neuron))

        for neuron in range(n_neuron):
            curr_neuron = list(np.zeros(n_trials))  # initialize neuron-list
            curr_neuron_spike = list(np.zeros(n_trials))  # initialize neuron-list
            session_trace = data[neuron]            # temp-save session trace of this neuron
            session_spike = spikes[neuron]  # temp-save session trace of this neuron

            for trial in range(n_trials):
                # extract trace of the current trial from the whole session
                if len(session_trace) > frame_list[trial]:
                    trial_trace, session_trace = session_trace[:frame_list[trial]], session_trace[frame_list[trial]:]
                    curr_neuron[trial] = trial_trace  # save trial trace in this neuron's list
                    trial_spike, session_spike = session_spike[:frame_list[trial]], session_spike[frame_list[trial]:]
                    curr_neuron_spike[trial] = trial_spike  # save trial trace in this neuron's list
                elif len(session_trace) == frame_list[trial]:
                    curr_neuron[trial] = session_trace
                    curr_neuron_spike[trial] = session_spike
                else:
                    print('Error in PlaceCellFinder.split_traces()')
            session[neuron] = curr_neuron  # save data from this neuron to the big session list
            session_spikes[neuron] = curr_neuron_spike  # save data from this neuron to the big session list

        self.session = session
        self.session_spikes = session_spikes
        self.params['n_neuron'] = n_neuron
        self.params['n_trial'] = n_trials

        print('\nSuccessfully separated traces into trials and sorted them by neurons.'
              '\nResults are stored in pcf.session and pcf.session_spikes.\n')

    def create_transient_only_traces(self):
        """
        Takes the traces ordered by neuron and trial and modifies them into transient-only traces.
        Significant transients are detected using the full-width-half-maximum measurement of standard deviation (see
        Koay et al., 2019). The traces itself are left untouched, but all values outside of transients are set to 0.
        This is mainly useful for the place field criterion 3 (20% of time inside place field has to be transients).
        The structure of the resulting list is the same as PCF.session (see split_traces_into_trials()).
        Additionally, the noise level sigma is saved for every neuron for each trial in params[sigma] as an array with
        the shape [n_neurons X n_trials] and can be indexed as such to retrieve noise levels for every trial.
        :return: Updated PCF object with the significant-transient-only session_trans list
        """
        session_trans = copy.deepcopy(self.session)
        self.params['sigma'] = np.zeros((self.params['n_neuron'], self.params['n_trial']))
        self.params['sigma'].fill(np.nan)
        for neuron in range(self.params['n_neuron']):
            curr_neuron = self.session[neuron]
            for i in range(len(curr_neuron)):
                trial = curr_neuron[i]
                # get noise level of the data via FWHM
                try:
                    sigma = self.get_noise_fwhm(trial)
                except ValueError:
                    raise ValueError(f'No data for neuron {neuron}, trial {i}.')
                self.params['sigma'][neuron][i] = sigma     # save noise level of current trial in the params dict
                # get time points where the signal is more than 4x sigma (Koay et al., 2019)
                if sigma == 0:
                    idx = np.array([])
                else:
                    thresh = self.params['trans_thresh']
                    if type(thresh) == int:    # use one threshold for borders of transient (Koay)
                        idx = np.where(trial >= thresh * sigma)[0]
                    elif type(thresh) == tuple:  # use different thresholds for on and offset of transients
                        idx = np.where(trial >= thresh[0] * sigma)[0]
                    else:
                        raise Exception(f'Parameter "trans_thresh" has to be int or tuple, but was {type(thresh)}!')

                if idx.size > 0:
                    # split indices into consecutive blocks
                    blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

                    # if Dombecks criterion of 2-0.5 sigma should be used, the transient has to be extended
                    if type(thresh) == tuple:
                        for j in range(len(blocks)-1):
                            # test if the transient comes back down to 0.5sigma at all until the end
                            if np.all(trial[blocks[j][0]:] >= thresh[1] * sigma):
                                # if this transient goes until the end, the last stop is the last index of the trial
                                new_stop = len(trial)-1
                            else:
                                # otherwise find the index where it crosses the 2nd threshold
                                new_stop = np.where(trial[blocks[j][0]:] <= thresh[1] * sigma)[0][0]
                            blocks[j] = np.arange(blocks[j][0], blocks[j][0]+new_stop)

                    # find blocks of >500 ms length (use frame rate in cnmf object) and merge them in one array
                    duration = int(self.params['trans_length'] / (1 / self.cnmf.params.data['fr']))
                    try:
                        transient_idx = np.concatenate([x for x in blocks if x.size >= duration])
                    except ValueError:
                        transient_idx = []
                else:
                    transient_idx = []
                # create a transient-only trace of the raw calcium trace
                trans_only = trial.copy()
                select = np.in1d(range(trans_only.shape[0]), transient_idx)  # create mask for trans-only indices
                trans_only[~select] = 0     # set everything outside of this mask to 0
                # add the transient only trace to the list
                session_trans[neuron][i] = trans_only

        # add the final data structure to the PCF object
        self.session_trans = session_trans
        print('\nSuccessfully created transient-only traces.\nThey are stored in pcf.session_trans.')

    def get_noise_fwhm(self, data):
        """
        Returns noise level as standard deviation (sigma) from a dataset using full-width-half-maximum
        (Koay et al. 2019). This method is less sensitive to long tails in the distribution (useful?).
        :param data: dataset in a 1D array that you need the noise level from
        :return: noise: float, sigma of given dataset
        """
        if np.all(data == 0): # catch trials without data
            sigma = 0
        else:
            plt.figure()
            x_data, y_data = sns.distplot(data).get_lines()[0].get_data()
            y_max = y_data.argmax()  # get idx of half maximum
            # get points above/below y_max that is closest to max_y/2 by subtracting it from the data and
            # looking for the minimum absolute value
            nearest_above = (np.abs(y_data[y_max:] - max(y_data) / 2)).argmin()
            nearest_below = (np.abs(y_data[:y_max] - max(y_data) / 2)).argmin()
            # get FWHM by subtracting the x-values of the two points
            fwhm = x_data[nearest_above + y_max] - x_data[nearest_below]
            # return noise level as FWHM/2.3548
            sigma = fwhm/2.3548
            plt.close()
        return sigma

    def import_behavior_and_align_traces(self, encoder_unit='raw'):
        """
        Imports behavioral data (merged_behavior.txt) and aligns the calcium traces to the VR position.
        Behavioral data is saved in the 'behavior' list that includes one array per trial with the following structure:
            universal time stamp -- VR position -- lick sensor -- 2p trigger -- encoder (speed)
        Frames per bin are saved in bin_frame_count, an array of shape [n_bins x n_trials] showing the number of frames
        that have to be averaged for each bin in every trial (stored in params).
        :param encoder_unit
        :return: Updated PCF object with behavior and binned data
        """
        self.params['n_bins'] = int(self.params['track_length'] / self.params['bin_length'])
        behavior = []
        is_faulty = False
        count = 0
        if self.params['trial_list'] is not None:
            for trial in self.params['trial_list']:
                path = glob.glob(trial+'//merged_behavior*.txt')
                if len(path) >= 1:
                    # if there are more than 1 behavior file, load the latest
                    mod_times = [os.path.getmtime(file) for file in path]
                    behavior.append(np.loadtxt(path[np.argmax(mod_times)], delimiter='\t'))
                    count_list = int(self.params['frame_list'][count])
                    count_imp = int(np.nansum(behavior[-1][:, 3]))
                    if count_imp != count_list:
                        print(f'Contradicting frame counts in trial {trial} (no. {count}):\n'
                              f'\tExpected {count_list} frames, imported {count_imp} frames...')
                        is_faulty = True
                    count += 1
                else:
                    print(f'Couldnt find behavior file at {trial}')
            if is_faulty:
                raise Exception('Frame count mismatch detected, stopping analysis.')
        else:
            raise Exception('You have to provide trial_list before aligning data to VR position!')

        self.behavior = behavior.copy()

        # Get frame counts for each bin for complete dataset (moving and resting frames)
        bin_frame_count_all = np.zeros((self.params['n_bins'], self.params['n_trial']), 'int')
        for trial in range(len(behavior)):  # go through vr data of every trial and prepare it for analysis

            # bin data in distance chunks
            bin_borders = np.linspace(-10, 110, self.params['n_bins'])
            idx = np.digitize(behavior[trial][:, 1], bin_borders)  # get indices of bins

            # check how many frames are in each bin
            for i in range(self.params['n_bins']):
                bin_frame_count_all[i, trial] = np.nansum(behavior[trial][np.where(idx == i + 1), 3])

        # double check if number of frames are correct
        for i in range(len(self.params['frame_list'])):
            frame_list_count = self.params['frame_list'][i]
            if frame_list_count != np.sum(bin_frame_count_all[:, i]):
                raise ValueError(f'Frame count not matching in trial {i + 1}: Frame list says {frame_list_count}, '
                                 f'import says {np.sum(bin_frame_count_all[:, i])}')

            # check that every bin has at least one frame in it
        if np.any(bin_frame_count_all == 0):
            all_zero_idx = np.where(bin_frame_count_all == 0)
            # if not, take a frame of the next bin (or the previous bin in case its the last bin
            for i in range(len(all_zero_idx[0])):
                zero_idx = (all_zero_idx[0][i], all_zero_idx[1][i])
                if zero_idx[0] == 79 and bin_frame_count_all[78, zero_idx[1]] > 1:
                    bin_frame_count_all[78, zero_idx[1]] -= 1
                    bin_frame_count_all[79, zero_idx[1]] += 1
                elif zero_idx[0] < 79 and bin_frame_count_all[zero_idx[0]+1, zero_idx[1]] > 1:
                    bin_frame_count_all[zero_idx[0]+1, zero_idx[1]] -= 1
                    bin_frame_count_all[zero_idx[0], zero_idx[1]] += 1
                else:
                    raise ValueError('No frame in these bins (#bin, #trial): {}'.format(*zip(zero_idx[0], zero_idx[1])))

        ##########################################################################################################
        ##################### Get frame counts for each bin for only moving frames ###############################
        # How to remove resting frames
        # - for every trial, make a mask with length n_frames (one entry per frame) that is True for frames where
        #   the mouse ran and False where the mouse was stationary
        # - temporarily save the new frame list with new frame counts (check binned frames with this list)
        # - update self.session

        # create a bool mask for every trial that tells if a frame should be included or not
        behavior_masks = []
        for trial in range(len(behavior)):
            # bool list for every frame of that trial (used to later filter out dff samples)
            behavior_masks.append(np.ones(int(np.nansum(behavior[trial][:, 3])), dtype=bool))
            frame_idx = np.where(behavior[trial][:, 3] == 1)[0]  # find sample_idx of all frames
            for i in range(len(frame_idx)):
                if i != 0:
                    # TODO: implement smoothing speed (2 s window) before removing (after Harvey 2009)
                    if encoder_unit == 'speed':
                        if np.mean(behavior[trial][frame_idx[i - 1]:frame_idx[i], 5]) <= 2.5:
                            # set index of mask to False (excluded in later analysis)
                            behavior_masks[-1][i] = False
                            # set the bad frame in the behavior array to 0 to skip it during bin_frame_counting
                            behavior[trial][frame_idx[i], 3] = np.nan
                    elif encoder_unit == 'raw':
                        if abs(np.sum(behavior[trial][frame_idx[i - 1]:frame_idx[i], 4])) < 30:
                            # set index of mask to False (excluded in later analysis)
                            behavior_masks[-1][i] = False
                            # set the bad frame in the behavior array to 0 to skip it during bin_frame_counting
                            behavior[trial][frame_idx[i], 3] = np.nan
                    else:
                        pass
                        #return print('Encoder unit not recognized, behavior could not be aligned.')
                else:
                    if behavior[trial][0, 4] > -30:
                        behavior_masks[-1][i] = False
                        behavior[trial][frame_idx[i], 3] = np.nan
        self.params['resting_mask'] = behavior_masks

        # get new bin_frame_count
        bin_frame_count = np.zeros((self.params['n_bins'], self.params['n_trial']), 'int')
        for trial in range(len(behavior)):  # go through vr data of every trial and prepare it for analysis

            # bin data in distance chunks
            bin_borders = np.linspace(-10, 110, self.params['n_bins'])
            idx = np.digitize(behavior[trial][:, 1], bin_borders)  # get indices of bins

            # check how many frames are in each bin
            for i in range(self.params['n_bins']):
                bin_frame_count[i, trial] = np.nansum(behavior[trial][np.where(idx == i + 1), 3])

        # check that every bin has at least one frame in it
        if np.any(bin_frame_count == 0):
            all_zero_idx = np.where(bin_frame_count == 0)
            # if not, take a frame of the next bin (or the previous bin in case its the last bin
            for i in range(len(all_zero_idx[0])):
                zero_idx = (all_zero_idx[0][i], all_zero_idx[1][i])
                if zero_idx[0] == 79 and bin_frame_count[78, zero_idx[1]] > 1:
                    bin_frame_count[78, zero_idx[1]] -= 1
                    bin_frame_count[79, zero_idx[1]] += 1
                elif zero_idx[0] < 79 and bin_frame_count[zero_idx[0]+1, zero_idx[1]] > 1:
                    bin_frame_count[zero_idx[0]+1, zero_idx[1]] -= 1
                    bin_frame_count[zero_idx[0], zero_idx[1]] += 1
                else:
                    raise ValueError('No frame in these bins (#bin, #trial): {}'.format(*zip(zero_idx[0], zero_idx[1])))

        ############################################################################################################

        # if everything worked fine, we can save the frame count parameters
        self.params['bin_frame_count'] = bin_frame_count
        self.params['bin_frame_count_all'] = bin_frame_count_all

    def bin_activity_to_vr(self, remove_resting=None):
        """
        Bins activity trace of every neuron to the VR position. This brings the neuronal activity during all trials
        to a uniform length.
        Binned calcium data is saved in two formats:
            - bin_activity, a list of neurons, each element being an array with shape [n_bins x n_trials] that stores
              the binned dF/F for every trial.
            - bin_avg_activity, an array of shape [n_neuron x n_bins] that contain the dF/F for each bin of every
              neuron, averaged across trials. This is what will mainly be used for place cell analysis.
        :param remove_resting: bool flag whether resting frames should be removed before binning. This choice will
                               affect downstream analysis steps and will thus be saved in params['resting_removed'].
        """
        self.params['n_bins'] = int(self.params['track_length'] / self.params['bin_length'])
        if remove_resting is None:
            if 'resting_removed' in self.params.keys():
                remove_resting = self.params['resting_removed']
            else:
                remove_resting = True
                print('Remove resting not provided. Default to True.')

        # bin the activity for every neuron to the VR position, construct bin_activity and bin_avg_activity
        self.bin_activity = []
        self.bin_spike_rate = []
        self.bin_avg_activity = np.zeros((self.params['n_neuron'], self.params['n_bins']))
        self.bin_avg_spike_rate = np.zeros((self.params['n_neuron'], self.params['n_bins']))
        for neuron in range(self.params['n_neuron']):
            if remove_resting:
                # if resting frames should be removed, mask data before binning (trial-wise)
                self.params['resting_removed'] = remove_resting
                n_bin_act, n_bin_avg_act, n_bin_spikes, n_bin_avg_spikes = \
                    self.bin_neuron_activity_to_vr(self.session[neuron], self.session_spikes[neuron],
                                                   bf_count=self.params['bin_frame_count'])
            else:
                self.params['resting_removed'] = remove_resting
                n_bin_act, n_bin_avg_act, n_bin_spikes, n_bin_avg_spikes = \
                    self.bin_neuron_activity_to_vr(self.session[neuron], self.session_spikes[neuron],
                                                   bf_count=self.params['bin_frame_count_all'])
            self.bin_activity.append(n_bin_act)
            self.bin_spike_rate.append(n_bin_spikes)
            self.bin_avg_activity[neuron, :] = n_bin_avg_act
            self.bin_avg_spike_rate[neuron, :] = n_bin_avg_spikes
        print('\nSuccessfully aligned calcium data to the VR position bins.'
              '\nResults are stored in pcf.bin_activity and pcf.bin_avg_activity,\nbinned spike rates in '
              'pcf.bin_spike_rate and pcf.bin_avg_spike_rate.')

    def bin_neuron_activity_to_vr(self, neuron_traces, spikes, n_bins=None, bf_count=None):
        """
        Takes bin_frame_count and bins the dF/F traces of all trials of one neuron to achieve a uniform trial length
        for place cell analysis. Procedure for every trial: Algorithm goes through every bin and extracts the
        corresponding frames according to bin_frame_count.
        :param neuron_traces: list of arrays that contain the dF/F traces of a neuron. From self.session[n_neuron]
        :param spikes: list of arrays that contain spike probabilities of a neuron. From self.session_spike[n_neuron]
        :param n_bins: int, number of bins the trace should be split into
        :param bf_count: np.array containing the number of frames in each bin
        :return: bin_activity (list of trials), bin_avg_activity (1D array) for this neuron
        """
        self.params['n_bins'] = int(self.params['track_length'] / self.params['bin_length'])
        if n_bins is None:
            n_bins = self.params['n_bins']
        if bf_count is None:
            if self.params['resting_removed']:
                bin_frame_count = self.params['bin_frame_count']
            else:
                bin_frame_count = self.params['bin_frame_count_all']
        else:
            bin_frame_count = bf_count

        bin_activity = np.zeros((self.params['n_trial'], n_bins))
        bin_spike_rate = np.zeros((self.params['n_trial'], n_bins))
        for trial in range(self.params['n_trial']):
            if self.params['resting_removed']:
                curr_trace = neuron_traces[trial][self.params['resting_mask'][trial]]
                curr_spikes = np.nan_to_num(spikes[trial][self.params['resting_mask'][trial]])
            else:
                curr_trace = neuron_traces[trial]
                curr_spikes = spikes[trial]
            curr_bins = bin_frame_count[:, trial]
            curr_act_bin = np.zeros(n_bins)
            curr_spike_bin = np.zeros(n_bins)
            for bin_no in range(n_bins):
                # extract the trace of the current bin from the trial trace
                if len(curr_trace) > curr_bins[bin_no]:
                    trace_to_avg, curr_trace = curr_trace[:curr_bins[bin_no]], curr_trace[curr_bins[bin_no]:]
                    spike_to_avg, curr_spikes = curr_spikes[:curr_bins[bin_no]], curr_spikes[curr_bins[bin_no]:]
                elif len(curr_trace) == curr_bins[bin_no]:
                    trace_to_avg = curr_trace
                    spike_to_avg = curr_spikes
                else:
                    raise Exception('Something went wrong during binning...')
                if trace_to_avg.size > 0:
                    curr_act_bin[bin_no] = np.mean(trace_to_avg)
                    # sum instead of mean (Peters spike probability is cumulative)
                    curr_spike_bin[bin_no] = np.nansum(spike_to_avg)
                else:
                    curr_act_bin[bin_no] = 0
                    curr_spike_bin[bin_no] = 0
            bin_activity[trial] = curr_act_bin
            # Smooth average spike rate and transform values into mean firing rates by dividing by the time in s
            # occupied by the bin (from number of samples * sampling rate)
            smooth_spike_bin = gaussian_filter1d(curr_spike_bin, 1)
            for i in range(n_bins):
                smooth_spike_bin[i] = smooth_spike_bin[i]/(bin_frame_count[i, trial]*(1/self.cnmf.params.data['fr']))
            bin_spike_rate[trial] = smooth_spike_bin

        # Get average activity across trials of this neuron for every bin
        bin_avg_activity = np.mean(bin_activity, axis=0)
        bin_avg_spike_rate = np.nanmean(bin_spike_rate, axis=0)

        return bin_activity, bin_avg_activity, bin_spike_rate, bin_avg_spike_rate

    def find_place_cells(self, show_prog_bar=False):
        """
        Wrapper function that checks every neuron for place fields by calling find_place_field_neuron().
        Also initializes params['place_results'], where results from place cell checks are stored for each neuron.
        Order of 'place_results': pre_screening -- is_large_enough -- is_strong_enough -- has_transients -- p<0.05
        :return: updated PCF object with place cell results
        """
        self.params['place_results'] = np.zeros((self.params['n_neuron'], 5), dtype='bool')
        self.place_cells = []
        for i in range(self.bin_avg_activity.shape[0]):
            self.find_place_field_neuron(self.bin_avg_activity[i, :], i)
            if show_prog_bar:
                progress(i+1, self.bin_avg_activity.shape[0], status='Processing neurons...', percent=False)
        print(f'Done! {len(self.place_cells)} place cells found in total!')

    def find_place_field_neuron(self, data, neuron_id):
        """
        Performs place field analysis (smoothing, pre-screening and criteria application) on a single neuron data set.

        :param data: 1D array, binned and across-trial-averaged dF/F data of one neuron, e.g. from bin_avg_activity[i,:]
        :param neuron_id: ID of the neuron that the data belongs to (its index in session list)
        :return: updated PCF object with filled-in place_cells and place_cells_reject
        """
        # smoothing binned data by averaging over adjacent bins
        smooth_trace = self.smooth_trace(data)

        # pre-screening for potential place fields
        pot_place_blocks = self.pre_screen_place_fields(smooth_trace)

        # if the trace has above-threshold values, continue with place cell criteria
        if len(pot_place_blocks) > 0:
            self.params['place_results'][neuron_id, 0] = True
            place_fields_passed = self.apply_pf_criteria(smooth_trace, pot_place_blocks, neuron_id)
            # if this neuron has one or more place fields that passed all three criteria, validate via bootstrapping
            if len(place_fields_passed) > 0:
                p_value = self.bootstrapping(neuron_id)
                # if the p_value is lower than 0.05, accept the current cell as a place cell
                if p_value < 0.05:
                    self.place_cells.append((neuron_id, place_fields_passed, p_value))
                    self.params['place_results'][neuron_id, 4] = True
                    print(f'Neuron {neuron_id} identified as place cell with p={p_value}!')
                # if the p_value is higher than 0.05, save it in a separate list
                else:
                    self.place_cells_reject.append((neuron_id, place_fields_passed, p_value))
                    print(f'Neuron {neuron_id} identified as place cell, but p={p_value}.')

    def pre_screen_place_fields(self, trace):
        """
        Performs pre-screening of potential place fields in a trace. A potential place field is any bin/point that
        has a higher dF/F value than 'place_thresh'% (default 25%) of the difference between the baseline and maximum
        dF/F of this trace. The baseline dF/F is set as the mean of the 'bin_base' % (default 25%) least active bins.

        :param trace: 1D array of the trace where place fields are to be found, e.g. smoothed binned trial-averaged data
        :return: list of arrays containing separate potential place fields (empty list if there are no place fields)
        """
        f_max = max(trace)  # get maximum DF/F value
        # get baseline dF/F value from the average of the 'bin_base' % least active bins (default 25% of n_bins)
        f_base = np.mean(np.sort(trace)[:int(trace.size * self.params['bin_base'])])
        # get threshold value above which a point is considered part of the potential place field (default 25%)
        f_thresh = ((f_max - f_base) * self.params['place_thresh']) + f_base
        # get indices where the smoothed trace is above threshold
        pot_place_idx = np.where(trace >= f_thresh)[0]

        if pot_place_idx.size != 0:
            # split indices into consecutive blocks to get separate place fields
            pot_place_blocks = np.split(pot_place_idx, np.where(np.diff(pot_place_idx) != 1)[0] + 1)
        else:
            # return an empty list in case there were no potential place cells found in this trace
            pot_place_blocks = []

        return pot_place_blocks

    def smooth_trace(self, trace):
        """
        Smoothes a trace (usually binned, but can also be used unbinned) by averaging each point across adjacent values.
        Sliding window size is determined by params['bin_window_avg'] (default 3).

        :param trace: 1D array containing the data points.
        :return: array of the same size as input trace, but smoothed
        """
        smooth_trace = trace.copy()
        for i in range(len(trace)):
            # get the frame windows around the current time point i
            if i < self.params['bin_window_avg']:
                curr_left_bin = trace[:i]
            else:
                curr_left_bin = trace[i - self.params['bin_window_avg']:i]
            if i + self.params['bin_window_avg'] > len(trace):
                curr_right_bin = trace[i:]
            else:
                curr_right_bin = trace[i:i + self.params['bin_window_avg']]
            curr_bin = np.concatenate((curr_left_bin, curr_right_bin))

            smooth_trace[i] = np.mean(curr_bin)

        return smooth_trace

    def apply_pf_criteria(self, trace, place_blocks, neuron_id, save_results=True):
        """
        Applies the criteria of place fields to potential place fields of a trace. A place field is accepted when...
            1) it stretches at least 'min_bin_size' bins (default 10)
            2) its mean dF/F is larger than outside the field by a factor of 'fluo_infield'
            3) during 'trans_time'% of the time the mouse is located in the field, the signal consists of significant transients
        Place fields that pass these criteria have to have a p-value < 0.05 to be fully accepted. This is checked in
        the bootstrap() function.

        :param place_blocks: list of array(s) that hold bin indices of potential place fields, one array per field (from pot_place_blocks)
        :param trace: 1D array containing the trace in which the potential place fields are located
        :param neuron_id: Index of current neuron in the session_trans list. Needed for criterion 3.
        :param save_results: boolean flag whether place_cell_check results should be saved in params (no for bootstrap)
        :return: list of place field arrays that passed all three criteria (empty if none passed).
        """
        place_field_passed = []
        for pot_place in place_blocks:
            bin_size = self.is_large_enough(pot_place)
            intensity = self.is_strong_enough(trace, pot_place, place_blocks)
            transients = self.has_enough_transients(neuron_id, pot_place)
            if bin_size and intensity and transients:
                place_field_passed.append(pot_place)

            if save_results:
                if bin_size:
                    self.params['place_results'][neuron_id, 1] = True
                if intensity:
                    self.params['place_results'][neuron_id, 2] = True
                if transients:
                    self.params['place_results'][neuron_id, 3] = True

        return place_field_passed

    def is_large_enough(self, place_field):
        """
        Checks if the potential place field is large enough according to 'min_bin_size' (criterion 1).

        :param place_field: 1D array of indices of data points that form the potential place field
        :return: boolean value whether the criterion is passed or not
        """
        return place_field.size >= self.params['min_bin_size']

    def is_strong_enough(self, trace, place_field, all_fields):
        """
        Checks if the place field has a mean dF/F that is 'fluo_infield'x higher than outside the field (criterion 2).

        :param trace: 1D array of the trace data
        :param place_field: 1D array of indices of data points that form the potential place field
        :param all_fields: 1D array of indices of all place fields in this trace
        :return: boolean value whether the criterion is passed or not
        """
        pot_place_idx = np.in1d(range(trace.shape[0]), place_field)  # get an idx mask for the potential place field
        all_place_idx = np.in1d(range(trace.shape[0]), np.concatenate(all_fields))   # get an idx mask for all place fields
        return np.mean(trace[pot_place_idx]) >= self.params['fluo_infield'] * np.mean(trace[~all_place_idx])

    def has_enough_transients(self, neuron_id, place_field):
        """
        Checks if of the time during which the mouse is located in the potential field, at least 'trans_time'%
        consist of significant transients (criterion 3).

        :param neuron_id: 1D array of index of the current neuron in the session list
        :param place_field: 1D array of indices of data points that form the potential place field
        :return: boolean value whether the criterion is passed or not
        """
        place_frames_trace = []  # stores the trace of all trials when the mouse was in a place field as one data row
        for trial in range(self.params['n_trial']):
            if self.params['resting_removed']:
                # get the start and end frame for the current place field from the bin_frame_count array that stores how
                # many frames were pooled for each bin
                curr_place_frames = (np.sum(self.params['bin_frame_count'][:place_field[0], trial]),
                                     np.sum(self.params['bin_frame_count'][:place_field[-1] + 1, trial]))
                # use masked session_trans data to remove resting frames
                sess_trans_masked = self.session_trans[neuron_id][trial][self.params['resting_mask'][trial]]
                # attach the transient-only trace in the place field during this trial to the array
                place_frames_trace.append(sess_trans_masked[curr_place_frames[0]:curr_place_frames[1] + 1])
            else:
                curr_place_frames = (np.sum(self.params['bin_frame_count_all'][:place_field[0], trial]),
                                     np.sum(self.params['bin_frame_count_all'][:place_field[-1] + 1, trial]))
                # attach the transient-only trace in the place field during this trial to the array
                place_frames_trace.append(
                    self.session_trans[neuron_id][trial][curr_place_frames[0]:curr_place_frames[1] + 1])

        # create one big 1D array that includes all frames where the mouse was located in the place field
        # as this is the transient-only trace, we make it boolean, with False = no transient and True = transient
        place_frames_trace = np.hstack(place_frames_trace).astype('bool')
        # check if at least 'trans_time' percent of the frames are part of a significant transient
        return np.sum(place_frames_trace) >= self.params['trans_time'] * place_frames_trace.shape[0]

    def bootstrapping(self, neuron_id):
        """
        Performs bootstrapping on a unbinned dF/F trace and returns p-value for a place cell in this trace.
        The trace is divided in parts with 'split_size' length which are randomly shuffled 1000 times. Then, place cell
        detection is performed on each shuffled trace. The p-value is defined as the ratio of place cells detected in
        the shuffled traces versus number of shuffles (1000; see Dombeck et al., 2010). If this neuron's trace gets a
        p-value of p < 0.05 (place fields detected in less than 50 shuffles), the place field is accepted.

        :param neuron_id: 1D array of index of the checked neuron in the session list
        :return: p-value of place fields in this neuron
        """
        p_counter = 0
        for i in range(1000):
            # create shuffled neuron data by shuffling every trial
            shuffle = []
            for trial in self.session[neuron_id]:
                # divide the trial trace into splits of 'split_size' size and manually append the remainder
                div_length = trial.shape[0] - trial.shape[0] % self.params['split_size']
                split_trace = np.split(trial[:div_length], trial[:div_length].shape[0]/self.params['split_size'])
                split_trace.append(trial[div_length:])

                curr_shuffle = np.concatenate(random.sample(split_trace, len(split_trace)))
                shuffle.append(curr_shuffle)

            # bin trials to VR position
            if self.params['resting_removed']:
                bf_count = self.params['bin_frame_count']
            else:
                bf_count = self.params['bin_frame_count_all']
            # parse shuffle twice as a spike rate and catch unwanted output in two dummy variables
            bin_act, bin_avg_act, dummy1, dummy2 = self.bin_neuron_activity_to_vr(shuffle, shuffle, bf_count=bf_count)

            # perform place cell analysis on binned and trial-averaged activity of shuffled neuron trace
            smooth_trace = self.smooth_trace(bin_avg_act)
            pot_place_blocks = self.pre_screen_place_fields(smooth_trace)

            # if the trace has passed pre-screening, continue with place cell criteria
            if len(pot_place_blocks) > 0:
                place_fields_passed = self.apply_pf_criteria(smooth_trace, pot_place_blocks,
                                                             neuron_id, save_results=False)
                # if the shuffled trace contained a place cell that passed all 3 criteria, count it
                if len(place_fields_passed) > 0:
                    p_counter += 1

        return p_counter/1000   # return p-value of this neuron (number of place fields after 1000 shuffles)

    def reject_place_cells(self, rej):
        """
        Moves place cells from the place_cells to the place_cells_reject list.
        :param rej: list, contains global indices of place cells to be rejected.
        :return:
        """

        # extract tuples of rejected place cells
        bad_cells = [x for x in self.place_cells if x[0] in rej]

        # check if any cell in rej couldnt be found in place_cells
        bad_pc_idx = [x[0] for x in bad_cells]
        false_pc = [x for x in rej if x not in bad_pc_idx]
        if len(false_pc) > 0:
            return print(f'The following cells are not in place_cells: {false_pc}!')

        # append bad cells to the place_cells_reject list
        self.place_cells_reject = self.place_cells_reject + bad_cells

        # extract tuples of place cells that are not rejected
        self.place_cells = [x for x in self.place_cells if x[0] not in rej]

    def accept_place_cells(self, acc):
        """
        Moves place cells from the place_cells_reject to the place_cells list.
        :param acc: list, contains global indices of place cells to be accepted.
        :return:
        """

        # extract tuples of rejected place cells
        good_cells = [x for x in self.place_cells_reject if x[0] in acc]

        # check if any cell in rej couldnt be found in place_cells
        good_pc_idx = [x[0] for x in good_cells]
        false_pc = [x for x in acc if x not in good_pc_idx]
        if len(false_pc) > 0:
            return print(f'The following cells are not in place_cells_reject: {false_pc}!')

        # append bad cells to the place_cells_reject list
        self.place_cells = self.place_cells + good_cells

        # extract tuples of place cells that are not rejected
        self.place_cells_reject = [x for x in self.place_cells_reject if x[0] not in acc]


#%% Spatial information

    def get_spatial_information(self, bin_width=None, n_bins=None, trace='F_dff', remove_stationary=True, save=False):
        """
        Calculates amount of spatial information (SI) contained in the extracted neurons. Spatial information is calcu-
        lated with the formula from Skaggs et al., 1993, and applied to the activity trace averaged across all trials
        of that session. This results in one SI value for every cell. Data is re-binned to 10-cm wide bins (Bartos2018).
        The formula that is applied to all bins is: dff_bin/dff_tot * ln(dff_bin/dff_tot) * p_bin,
        where dff_bin and p_bin are the avg calcium activity and fraction of time spent in that bin and dff_tot the avg
        dff across all bins. The SI for each neuron is the sum of all bin SIs.
        :param bin_width: int, width of the bins in cm
        :return:
        """
        # get activity and position data
        data_all = getattr(self.cnmf.estimates, trace).T
        position_all = []
        for trial in range(len(self.behavior)):
            position_all.append(self.behavior[trial][np.where(self.behavior[trial][:, 3] == 1), 1])
        position_all = np.hstack(position_all).T

        # calculate number of bins
        if bin_width is None and n_bins is None:
            n_bins = 40
        elif bin_width is not None and n_bins is None:
            track_length = self.params['track_length']
            if track_length % bin_width == 0:
                n_bins = int(track_length / bin_width)
            else:
                raise Exception(f'Bin_width has to be a divisor of track length ({track_length} cm)!')
        elif bin_width is not None and n_bins is not None:
            track_length = self.params['track_length']
            if track_length % bin_width == 0:
                n_bin_test = int(track_length / bin_width)
            else:
                raise Exception(f'Bin_width has to be a divisor of track length ({track_length} cm)!')
            if n_bin_test != n_bins:
                raise Exception('Bin_width and n_bins result in contradicting bin numbers!')

        if remove_stationary:
        # remove samples where the mouse was stationary (less than 30 movement per frame)
            behavior_masks = []
            for trial in self.behavior:
                behavior_masks.append(
                    np.ones(int(np.nansum(trial[:, 3])), dtype=bool))  # bool list for every frame of that trial
                frame_idx = np.where(trial[:, 3] == 1)[0]  # find sample_idx of all frames
                for i in range(len(frame_idx)):
                    if i != 0:
                        # make index of the current frame False if the mouse didnt move much during the frame
                        if np.nansum(trial[frame_idx[i - 1]:frame_idx[i], 4]) > -30:
                            behavior_masks[-1][i] = False
                    else:
                        if trial[0, 4] > -30:
                            behavior_masks[-1][i] = False
            behavior_mask = np.hstack(behavior_masks)
            data = data_all[behavior_mask]
            position = position_all[behavior_mask]
        else:
            data = data_all
            position = position_all

        # remove data points where decoding was nan (beginning and end)
        nan_mask = np.isnan(data[:, 0])
        data = data[~nan_mask]
        position = position[~nan_mask]

        # bin data into n_bins, get mean event rate per bin
        bin_borders = np.linspace(-10, 110, n_bins)
        idx = np.digitize(position, bin_borders)  # get indices of bins

        # get fraction of bin occupancy
        unique_elements, counts_elements = np.unique(idx, return_counts=True)
        bin_freq = np.array([x / np.sum(counts_elements) for x in counts_elements])

        # get mean spikes/s for each bin
        bin_mean_act = np.zeros((n_bins, data.shape[1]))
        for bin_nr in range(n_bins):
            curr_bin_idx = np.where(idx == bin_nr + 1)[0]
            bin_act = data[curr_bin_idx]
            # bin_mean_act[bin_nr, :] = np.sum(bin_act, axis=0) / (bin_act.shape[0] / 30)
            bin_mean_act[bin_nr, :] = np.mean(bin_act, axis=0)
        # total_firing_rate = np.sum(data, axis=0) / (data.shape[0] / 30)
        total_firing_rate = np.mean(data, axis=0)

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
                    bin_si[i] = curr_trace[i] / tot_act * np.log(curr_trace[i] / tot_act) * bin_freq[i]
            spatial_info[cell] = np.sum(bin_si)

        if save:
            self.spatial_info = spatial_info
        else:
            return spatial_info


    #%% Visualization

    def load_gui(self):
        gui.run_gui(data=self.cnmf)

    def plot_separate_pc_contours(self):
        """
        Plots the contours of all place cells in separate plots. Clicking on a place cell prints out its index so it can
        be removed in case it is no real cell or a duplicate.
        """

        pc_idx = [x[0] for x in self.place_cells]
        lcm = self.cnmf.estimates.Cn
        n_cols = 6
        n_rows = ceil(len(pc_idx)/n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(15, 8))

        i = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if i < len(pc_idx):
                    curr_ax = ax[row, col]
                    com = draw_single_contour(ax=curr_ax, spatial=self.cnmf.estimates.A[:, pc_idx[i]], template=lcm)
                    curr_ax.set_picker(True)
                    curr_ax.set_url(i)
                    i += 1
                    plt.show()

        def onpick(event):
            this_plot = event.artist  # save artist (axis) where the pick was triggered
            print(plt.getp(this_plot, 'url'))

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.tight_layout()
        plt.show()

    def plot_binned_neurons(self, idx=None, sliced=False):
        if idx:
            if sliced:
                if type(idx) == int:
                    traces = self.bin_avg_activity[:idx]
                elif type(idx) == list and len(idx) == 2:
                    traces = self.bin_avg_activity[idx[0], idx[2]]
                else:
                    print('Idx has to be either int or list. If sliced, list has to have length 2.')
                    #return
            else:
                traces = self.bin_avg_activity[idx]
        else:
            traces = self.bin_avg_activity
        if (len(traces.shape) > 1 and traces.shape[0] < 30) or len(traces.shape) == 1:
            fig, ax = plt.subplots(nrows=traces.shape[0], ncols=2, sharex=True, figsize=(20, 12))
            for i in range(traces.shape[0]):
                im = ax[i, 1].pcolormesh(traces[i, np.newaxis])
                ax[i, 0].plot(traces[i])
                if i == ax[:, 0].size - 1:
                    ax[i, 0].spines['top'].set_visible(False)
                    ax[i, 0].spines['right'].set_visible(False)
                    ax[i, 0].set_yticks([])
                    ax[i, 1].spines['top'].set_visible(False)
                    ax[i, 1].spines['right'].set_visible(False)
                else:
                    ax[i, 0].axis('off')
                    ax[i, 1].axis('off')
                ax[i, 0].set_title(f'{i + 1}', x=-0.02, y=-0.4)
                fig.colorbar(im, ax=ax[i, 0])
            ax[i, 0].set_xlim(0, self.params['n_bins'])
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        else:
            print(f'Too many neurons to plot ({traces.shape[0]}).')

    def plot_individual_neuron(self, idx, vr_aligned=True, show_reward_zones=True, save=False):
        """
        Plots all trials of a single cell in a line graph and pcolormesh. If the cell is a place cell, the location of
        the accepted place fields of the cell is shaded red in the line plot.
        :param idx: Index of the to-be-plotted cell (following the indexing of Caiman, same idx as displayed in
                    plot_all_place_cells())
        :param show_reward_zones: bool flag whether reward zones should be shown as a grey shaded area in the line graph
        :param save: bool flag whether the figure should be automatically saved.
        :return:
        """
        if vr_aligned:
            traces = self.bin_activity[idx]
        else:
            traces = self.session[idx]

        # Get global y-axis scaling
        max_y = 0.05 * ceil(traces.max() / 0.05)
        min_y = 0.05 * floor(traces.min() / 0.05)

        # Get positions of accepted/potential place fields, remember if place cell is accepted or not
        place_field_idx = [x[1] for x in self.place_cells if x[0] == idx]
        bad_place_field_idx = [x[1] for x in self.place_cells_reject if x[0] == idx]
        if len(place_field_idx) > 0:
            place_field_idx = place_field_idx[0]
            accepted = True
        elif len(bad_place_field_idx) > 0:
            place_field_idx = bad_place_field_idx[0]
            accepted = False
        else:
            accepted = False

        if show_reward_zones:
            if is_session_novel(self.params['root']):
                zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
                self.params['novel'] = True
            else:
                zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])
                self.params['novel'] = False

            # Transform coordinates from VR-coordinates to bin indices
            zone_borders = zone_borders+10                              # Change scaling from -10-110 to 0-120 VR coords
            zone_borders = zone_borders/(120/self.params['n_bins'])     # Apply scale from VR coordinates to bins
            zone_length = int(np.round(zone_borders[0, 1] - zone_borders[0, 0]))    # Get length of reward zones
            zone_borders[:, 0] = np.array(zone_borders[:, 0], dtype=int)            # Round down first zone bin
            zone_borders[:, 1] = zone_borders[:, 0] + zone_length                   # Add RZ length to first bin idx
            self.params['zone_borders'] = np.array(zone_borders, dtype=int)         # Transform to int and save param

        fig, ax = plt.subplots(nrows=len(traces), ncols=2, sharex=True, figsize=(20, 12))
        for i in range(len(traces)):
            ax[i, 0].plot(traces[i])
            ax[i, 0].set_ylim(min_y, max_y)
            img = ax[i, 1].pcolormesh(traces[i, np.newaxis], vmax=max_y, vmin=min_y, cmap='jet')
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].set_yticks([])
            ax[i, 0].set_ylabel(f'Trial {i+1}', rotation=0, labelpad=30)
            ax[i, 1].set_yticks([])
            ax[i, 1].spines['top'].set_visible(False)
            ax[i, 1].spines['right'].set_visible(False)

            # draw place field
            for field in place_field_idx:
                if accepted:
                    ax[i, 0].axvspan(field[0], field[-1], color='r', alpha=0.3)
                else:
                    ax[i, 0].axvspan(field[0], field[-1], color='orange', alpha=0.3)

            # shade locations of reward zones
            if show_reward_zones:
                for zone in self.params['zone_borders']:
                    ax[i, 0].axvspan(zone[0], zone[1], facecolor='grey', alpha=0.2)

        if vr_aligned:
            ax[-1, 0].set_xlim(0, traces.shape[1])
            x_locs, labels = plt.xticks()
            plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
            plt.sca(ax[-1, 0])
            plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)

            ax[i, 0].set_xlim(0, traces.shape[1])
            ax[i, 0].set_xlabel('VR position')
            ax[i, 1].set_xlabel('VR position')

        # plot color bar
        fraction = 0.10  # fraction of original axes to use for colorbar
        half_size = int(np.round(ax.shape[0] / 2))  # plot colorbar in half of the figure
        cbar = fig.colorbar(img, ax=ax[half_size:, 1], fraction=fraction, label=r'$\Delta$F/F')  # draw color bar
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.label.set_size(15)

        fig.subplots_adjust(left=0.1, right=1 - (fraction + 0.05), top=0.9, bottom=0.1)
        fig.suptitle(f'Neuron {idx}', fontsize=18)

        if save:
            plt.savefig(os.path.join(self.params['root'], f'neuron_{idx}.png'))
            plt.close()

    def plot_pc_location(self, save=False, color='r', display_numbers=False):
        """
        Plots the contours of all place cells on the local correlation image via CaImAns plot_contours.
        :return:
        """
        place_cell_idx = [x[0] for x in self.place_cells]
        plt.figure()
        out=visualization.plot_contours(self.cnmf.estimates.A[:, place_cell_idx],
                                        self.cnmf.estimates.Cn, display_numbers=display_numbers, colors=color)
        ax = plt.gca()
        sess = self.params['session']
        ax.set_title(f'All place cells of session {sess}')
        ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.params['root'], 'place_cell_contours.png'))
            plt.close()

    def plot_all_place_cells(self, save=False, show_neuron_id=False, show_place_fields=True, sort='field',
                             show_reward_zones=True, fname='place_cells', global_lims=True):
        """
        Plots all place cells in the data set by line graph and pcolormesh.
        :param save: bool flag whether the figure should be automatically saved in the root and closed afterwards.
        :param show_neuron_id: bool flag whether neuron ID should be plotted next to the line graphs
        :param show_place_fields: bool flag whether place fields should be marked red in the line graph
        :param sort: str, how should the place cells be sorted? 'Max' sorts them for the earliest location of the
        maximum in each trace, 'field' sorts them for the earliest place field.
        :param show_reward_zones: bool flag whether reward zones should be shown as a grey shaded area in the line graph
        :param fname: str, file name of the .png file if save=True.
        :return:
        """

        place_cell_idx = [x[0] for x in self.place_cells]
        #todo: remove cells that have a outlier-high activity maximum?
        #place_cell_idx.remove(206)
        # place_cell_idx.remove(340)

        traces = self.bin_avg_activity[place_cell_idx]
        n_neurons = traces.shape[0]

        # Get regions of reward zones
        if show_reward_zones:
            if is_session_novel(self.params['root']):
                zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
                self.params['novel'] = True
            else:
                zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])
                self.params['novel'] = False

            # Transform coordinates from VR-coordinates to bin indices
            zone_borders = zone_borders+10                              # Change scaling from -10-110 to 0-120 VR coords
            zone_borders = zone_borders/(120/self.params['n_bins'])     # Apply scale from VR coordinates to bins
            zone_length = int(np.round(zone_borders[0, 1] - zone_borders[0, 0]))    # Get length of reward zones
            zone_borders[:, 0] = np.array(zone_borders[:, 0], dtype=int)            # Round down first zone bin
            zone_borders[:, 1] = zone_borders[:, 0] + zone_length                   # Add RZ length to first bin idx
            self.params['zone_borders'] = np.array(zone_borders, dtype=int)         # Transform to int and save param

        if n_neurons > 0:
            # figure out y-axis limits by rounding the maximum value in traces up to the next 0.05 step
            if global_lims:
                max_y = 0.05 * ceil(traces.max() / 0.05)
                min_y = 0.05 * floor(traces.min() / 0.05)

            # sort neurons after different criteria
            bins = []
            if sort == 'max':
                for i in range(n_neurons):
                    bins.append((i, np.argmax(traces[i, :])))
            elif sort == 'field':
                for i in range(n_neurons):
                    bins.append((i, self.place_cells[i][1][0][0])) # get the first index of the first place field
            else:
                print(f'Cannot understand sorting command {sort}.')
                for i in range(n_neurons):
                    bins.append((i, i))
            bins_sorted = sorted(bins, key=lambda tup: tup[1])

            trace_fig, trace_ax = plt.subplots(nrows=n_neurons, ncols=2, sharex=True, figsize=(18, 10))
            if n_neurons == 1:
                trace_ax = np.array(trace_ax)[np.newaxis]
            mouse = self.params['mouse']
            session = self.params['session']
            network = self.params['network']
            trace_fig.suptitle(f'All place cells of mouse {mouse}, session {session}, network {network}', fontsize=16)
            for i in range(n_neurons):
                curr_neur = bins_sorted[i][0]
                curr_trace = traces[curr_neur, np.newaxis]
                if not global_lims:
                    max_y = max(traces[curr_neur])
                    min_y = min(traces[curr_neur])
                img = trace_ax[i, 1].pcolormesh(curr_trace, vmax=max_y, vmin=min_y, cmap='jet')
                trace_ax[i, 0].plot(traces[curr_neur])
                trace_ax[i, 0].set_ylim(bottom=min_y, top=max_y)

                # plot place fields as shaded red area
                if show_place_fields:
                    curr_place_fields = self.place_cells[curr_neur][1]
                    for field in curr_place_fields:
                        trace_ax[i, 0].plot(field, traces[curr_neur][field], color='red')

                # plot reward zones as shaded grey areas
                if show_reward_zones:
                    for zone in self.params['zone_borders']:
                        trace_ax[i, 0].axvspan(zone[0], zone[1], facecolor='grey', alpha=0.2)

                # clean up axes
                # if i == trace_ax[:, 0].size - 1:
                trace_ax[i, 0].spines['top'].set_visible(False)
                trace_ax[i, 0].spines['right'].set_visible(False)
                #trace_ax[i, 0].tick_params(axis='y', which='major', labelsize=15)
                trace_ax[i, 0].set_yticks([])
                trace_ax[i, 1].set_yticks([])
                trace_ax[i, 1].spines['top'].set_visible(False)
                trace_ax[i, 1].spines['right'].set_visible(False)
                # else:
                #     trace_ax[i, 0].axis('off')
                #     trace_ax[i, 1].axis('off')
                if show_neuron_id:
                    trace_ax[i, 0].set_ylabel(f'Neuron {place_cell_idx[curr_neur]}', rotation=0, labelpad=30)

            # set x ticks to VR position, not bin number
            trace_ax[-1, 0].set_xlim(0, traces.shape[1])
            x_locs, labels = plt.xticks()
            plt.xticks(x_locs, (x_locs*self.params['bin_length']).astype(int), fontsize=15)
            plt.sca(trace_ax[-1, 0])
            plt.xticks(x_locs, (x_locs*self.params['bin_length']).astype(int), fontsize=15)

            # set axis labels and tidy up graph
            trace_ax[-1, 0].set_xlabel('VR position [cm]', fontsize=15)
            trace_ax[-1, 1].set_xlabel('VR position [cm]', fontsize=15)

            if global_lims:
                # plot color bar
                fraction = 0.10  # fraction of original axes to use for colorbar
                half_size = int(np.round(trace_ax.shape[0]/2))  # plot colorbar in half of the figure
                cbar = trace_fig.colorbar(img, ax=trace_ax[half_size:, 1],
                                          fraction=fraction, label=r'$\Delta$F/F')  # draw color bar
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.yaxis.label.set_size(15)

                # align all plots
                trace_fig.subplots_adjust(left=0.1, right=1-(fraction+0.05), top=0.9, bottom=0.1)
            plt.show()

            if save:
                plt.savefig(os.path.join(self.params['root'], f'{fname}.png'))
                plt.close()
        else:
            plt.figure()
            mouse = self.params['mouse']
            session = self.params['session']
            network = self.params['network']
            plt.title(f'All place cells of mouse {mouse}, session {session}, network {network}', fontsize=16)
            if save:
                plt.savefig(os.path.join(self.params['root'], f'{fname}.png'))
                plt.close()

    def plot_spatial_info(self, data=None, verbose=True, show_reject=False, save=False, overwrite=False):
        """
        Plots spatial information content of neurons. Data from self.spatial_info or provided externally
        (from self.get_spatial_info with save=False).
        :param data: 1D np.array holding mean spatial information for every neuron.
        :param verbose: boolean flag, whether p-value of significance test should be written out or reported as stars.
        :return:
        """
        if not hasattr(self, 'spatial_info'):
            if data is not None:
                spatial_info = data
            else:
                raise Exception('Spatial information data not found!')
        else:
            if data is not None and self.spatial_info != data:
                raise Exception('Spatial info from object and from externally provided array do not match. Choose one!')
            else:
                spatial_info = self.spatial_info

        # classify data points into place cells and non-place cells
        pc_label = []
        pc_idx = [x[0] for x in self.place_cells]
        pc_rej = [x[0] for x in self.place_cells_reject]
        for cell_idx in range(len(spatial_info)):
            if show_reject:
                if cell_idx in pc_idx:
                    pc_label.append('accepted')
                elif cell_idx in pc_rej:
                    pc_label.append('rejected')
                else:
                    pc_label.append('no')
            else:
                if cell_idx in pc_idx:
                    pc_label.append('yes')
                else:
                    pc_label.append('no')

        # add sample size to labels
        no_count = pc_label.count('no')
        if show_reject:
            acc_count = pc_label.count('accepted')
        else:
            acc_count = pc_label.count('yes')
        rej_count = pc_label.count('rejected')
        pc_label = [f'no (n={no_count})' if x == 'no' else x for x in pc_label]
        pc_label = [f'yes (n={acc_count})' if x == 'yes' else x for x in pc_label]
        pc_label = [f'rejected (n={rej_count})' if x == 'rejected' else x for x in pc_label]

        df = pd.DataFrame(data={'SI': spatial_info, 'Place cell': pc_label, 'dummy': np.zeros(len(spatial_info))})

        mouse = self.params['mouse']
        session = self.params['session']
        plt.figure()
        plt.title(f'Spatial info {mouse}, {session}')
        ax = sns.barplot(x='Place cell', y='SI', data=df)

        # perform statistical tests and show results on plot
        if verbose and not show_reject:
            results = add_stat_annotation(ax, data=df, x='Place cell', y='SI', text_format='full',
                                          box_pairs=[(f'yes (n={acc_count})', f'no (n={no_count})')],
                                          test='Mann-Whitney')
        elif verbose and show_reject:
            results = add_stat_annotation(ax, data=df, x='Place cell', y='SI', text_format='full',
                                          box_pairs=[(f'accepted (n={acc_count})', f'no (n={no_count})')],
                                          test='Mann-Whitney')
        elif not verbose and not show_reject:
            results = add_stat_annotation(ax, data=df, x='Place cell', y='SI', text_format='stars',
                                          box_pairs=[(f'yes (n={acc_count})', f'no (n={no_count})')],
                                          test='Mann-Whitney')
        elif not verbose and show_reject:
            results = add_stat_annotation(ax, data=df, x='Place cell', y='SI', text_format='stars',
                                          box_pairs=[(f'accepted (n={acc_count})', f'no (n={no_count})')],
                                          test='Mann-Whitney')

        sns.stripplot(x='Place cell', y='SI', data=df, linewidth=1)

        if save:
            file_dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\spatial information'
            file_name = f'spatial_info_{mouse}_{session}_no_rest.png'
            save_path = os.path.join(file_dir, file_name)
            if os.path.isfile(save_path) and not overwrite:
                answer = None
                while answer not in ("y", "n", 'yes', 'no'):
                    answer = input(f"File [...]{save_path[-40:]} already exists!\nOverwrite? [y/n] ")
                    if answer == "yes" or answer == 'y':
                        plt.savefig(os.path.join(file_dir, file_name))
                        print(f'Plot successfully saved at {save_path}.')
                        return save_path
                    elif answer == "no" or answer == 'n':
                        print('Saving cancelled.')
                        return None
                    else:
                        print("Please enter yes/y or no/n.")
            else:
                plt.savefig(os.path.join(file_dir, file_name))

