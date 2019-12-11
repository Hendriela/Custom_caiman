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
from math import ceil
import os
import re
from behavior_import import progress


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
        self.session_trans = None       # Same as session, just with significant transients only (rest is 0)
        self.behavior = None            # Array containing behavioral data and frame time stamps (for alignment)
        self.bin_activity = None        # Same structure as session, but binned activity normalized to VR position
        self.bin_avg_activity = None    # List of neurons and their binned activity averaged across trials
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
                       'bin_length': 2,         # length in cm VR distance of each bin in which to group the dF/F traces
                       'bin_window_avg': 3,     # sliding window of bins (left and right) for trace smoothing
                       'bin_base': 0.25,        # fraction of lowest bins that are averaged for baseline calculation
                       'place_thresh': 0.25,    # threshold of being considered for place fields, calculated
                                                # from difference between max and baseline dF/F
                       'min_pf_size_cm': 10,    # minimum size in cm for a place field (should be 15-20 cm)
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
            else:
                print(f'Parameter {key} was not recognized!')

        if self.params['root'] is None:
            print(f'Essential parameter root has not been provided upon initialization.')
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
        self.params['min_bin_size'] = int(ceil(self.params['min_pf_size_cm'] / self.params['bin_length']))

        # find directories, files and frame counts
        self.params['frame_list'] = []
        for trial in self.params['trial_list']:
            if len(glob.glob(trial+'//*.mmap')) != 0:
                self.params['frame_list'].append(int(glob.glob(trial+'//*.mmap')[0].split('_')[-2]))
            else:
                print(f'No memmap files found at {trial}. Run motion correction before initializing PCF object!')

        # find mouse number and session
        try:
            self.params['mouse'] = self.params['root'].split(os.sep)[-3]
            self.params['session'] = self.params['root'].split(os.sep)[-2]
            self.params['network'] = self.params['root'].split(os.sep)[-1]
        except IndexError:
            self.params['mouse'] = self.params['root'].split('/')[-3]
            self.params['session'] = self.params['root'].split('/')[-2]
            self.params['network'] = self.params['root'].split('/')[-1]

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
        n_neuron = self.cnmf.estimates.F_dff.shape[0]
        session = list(np.zeros(n_neuron))

        for neuron in range(n_neuron):
            curr_neuron = list(np.zeros(n_trials))  # initialize neuron-list
            session_trace = self.cnmf.estimates.F_dff[neuron]  # temp-save DF/F session trace of this neuron

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

        self.session = session
        self.params['n_neuron'] = n_neuron
        self.params['n_trial'] = n_trials

        print('\nSuccessfully separated traces into trials and sorted them by neurons.'
              '\nResults are stored in pcf.session.\n')

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
                sigma = self.get_noise_fwhm(trial)
                self.params['sigma'][neuron][i] = sigma     # save noise level of current trial in the params dict
                # get time points where the signal is more than 4x sigma (Koay et al., 2019)
                if sigma == 0:
                    idx = []
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
                            new_stop = np.where(trial[blocks[j][0]:] <= thresh[1] * sigma)[0][0]
                            blocks[j] = np.arange(blocks[j][0], new_stop+blocks[j][0]+1)

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

    def import_behavior_and_align_traces(self):
        """
        Imports behavioral data (merged_behavior.txt) and aligns the calcium traces to the VR position.
        Behavioral data is saved in the 'behavior' list that includes one array per trial with the following structure:
            universal time stamp -- VR position -- lick sensor -- 2p trigger -- encoder (speed)
        Frames per bin are saved in bin_frame_count, an array of shape [n_bins x n_trials] showing the number of frames
        that have to be averaged for each bin in every trial (stored in params).
        Binned calcium data is saved in two formats:
            - as bin_activity, a list of neurons that consist of an array with shape [n_bins x n_trials] and stores
              the binned dF/F for every trial.
            - as bin_avg_activity, an array of shape [n_neuron x n_bins] that contain the dF/F for each bin of every
              neuron, averaged across trials. This is what will mainly be used for place cell analysis.

        :return: Updated PCF object with behavior and binned data
        """
        behavior = []
        is_faulty = False
        count = 0
        if self.params['trial_list'] is not None:
            for trial in self.params['trial_list']:
                behavior.append(np.loadtxt(trial+'//merged_behavior.txt', delimiter='\t'))
                count_list = int(self.params['frame_list'][count])
                count_imp = int(np.sum(behavior[-1][:, 3]))
                if count_imp != count_list:
                    print(f'Contradicting frame counts in trial {trial} (no. {count}):\n'
                          f'\tExpected {count_list} frames, imported {count_imp} frames...')
                    is_faulty = True
                count += 1
            if is_faulty:
                raise Exception('Frame count mismatch detected, stopping analysis.')
        else:
            raise Exception('You have to provide trial_list before aligning data to VR position!')

        bin_frame_count = np.zeros((self.params['n_bins'], self.params['n_trial']), 'int')
        for trial in range(len(behavior)):  # go through vr data of every trial and prepare it for analysis

            # bin data in distance chunks
            bin_borders = np.linspace(-10, 110, self.params['n_bins'])
            idx = np.digitize(behavior[trial][:, 1], bin_borders)  # get indices of bins

            # check how many frames are in each bin
            for i in range(self.params['n_bins']):
                bin_frame_count[i, trial] = np.sum(behavior[trial][np.where(idx == i + 1), 3])

        # double check if number of frames are correct
        for i in range(len(self.params['frame_list'])):
            frame_list_count = self.params['frame_list'][i]
            if frame_list_count != np.sum(bin_frame_count[:, i]):
                print(f'Frame count not matching in trial {i+1}: Frame list says {frame_list_count}, import says {np.sum(bin_frame_count[:, i])}')

        self.behavior = behavior
        self.params['bin_frame_count'] = bin_frame_count
        print('\nSuccessfully aligned traces with VR position.')

        # bin the activity for every neuron to the VR position, construct bin_activity and bin_avg_activity
        self.bin_activity = []
        self.bin_avg_activity = np.zeros((self.params['n_neuron'], self.params['n_bins']))
        for neuron in range(self.params['n_neuron']):
            neuron_bin_activity, neuron_bin_avg_activity = self.bin_activity_to_vr(self.session[neuron])
            self.bin_activity.append(neuron_bin_activity)
            self.bin_avg_activity[neuron, :] = neuron_bin_avg_activity
        print('\nSuccessfully aligned calcium data to the VR position bins.'
              '\nResults are stored in pcf.bin_activity and pcf.bin_avg_activity.\n')

    def bin_activity_to_vr(self, neuron_traces):
        """
        Takes bin_frame_count and bins the dF/F traces of all trials of one neuron to achieve a uniform trial length
        for place cell analysis. Procedure for every trial: Algorithm goes through every bin and extracts the
        corresponding frames according to bin_frame_count.
        :param neuron_traces: list of arrays that contain the dF/F traces of a neuron. From self.session[n_neuron]
        :return: bin_activity (list of trials), bin_avg_activity (1D array) for this neuron
        """
        bin_activity = np.zeros((self.params['n_trial'], self.params['n_bins']))
        for trial in range(self.params['n_trial']):
            curr_trace = neuron_traces[trial]
            curr_bins = self.params['bin_frame_count'][:, trial]
            curr_act_bin = np.zeros(self.params['n_bins'])
            for bin_no in range(self.params['n_bins']):
                # extract the trace of the current bin from the trial trace
                if len(curr_trace) > curr_bins[bin_no]:
                    trace_to_avg, curr_trace = curr_trace[:curr_bins[bin_no]], curr_trace[curr_bins[bin_no]:]
                elif len(curr_trace) == curr_bins[bin_no]:
                    trace_to_avg = curr_trace
                else:
                    raise Exception('Something went wrong during binning...')
                if trace_to_avg.size > 0:
                    curr_act_bin[bin_no] = np.mean(trace_to_avg)
                else:
                    curr_act_bin[bin_no] = 0
            bin_activity[trial] = curr_act_bin

        # Get average activity across trials of this neuron for every bin
        bin_avg_activity = np.mean(bin_activity, axis=0)

        return bin_activity, bin_avg_activity

    def find_place_cells(self):
        """
        Wrapper function that checks every neuron for place fields by calling find_place_field_neuron().
        Also initializes params['place_results'], where results from place cell checks are stored for each neuron.
        Order of 'place_results': pre_screening -- is_large_enough -- is_strong_enough -- has_transients -- p<0.05
        :return: updated PCF object with place cell results
        """
        self.params['place_results'] = np.zeros((self.params['n_neuron'], 5), dtype='bool')
        for i in range(self.bin_avg_activity.shape[0]):
            self.find_place_field_neuron(self.bin_avg_activity[i, :], i)
            progress(i, self.bin_avg_activity.shape[0] - 1, status='Processing neurons...', percent=False)
        print(f'Done! {len(self.place_cells)} place cells found in total!')

    def find_place_field_neuron(self, data, neuron_id):
        """
        Performs place field analysis (smoothing, pre-screening and criteria application) on a single neuron data set.

        :param data: 1D array, binned and across-trial-averaged dF/F data of one neuron, e.g. from bin_avg_activity
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
        for trial in range(self.params['bin_frame_count'].shape[1]):
            # get the start and end frame for the current place field from the bin_frame_count array that stores how
            # many frames were pooled for each bin
            curr_place_frames = (np.sum(self.params['bin_frame_count'][:place_field[0], trial]),
                                 np.sum(self.params['bin_frame_count'][:place_field[-1] + 1, trial]))
            # attach the transient-only trace in the place field during this trial to the array
            place_frames_trace.append(self.session_trans[neuron_id][trial][curr_place_frames[0]:curr_place_frames[1] + 1])

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
            bin_act, bin_avg_act = self.bin_activity_to_vr(shuffle)

            # perform place cell analysis on binned and trial-averaged activity of shuffled neuron trace
            smooth_trace = self.smooth_trace(bin_avg_act)
            pot_place_blocks = self.pre_screen_place_fields(smooth_trace)

            # if the trace has passed pre-screening, continue with place cell criteria
            if len(pot_place_blocks) > 0:
                place_fields_passed = self.apply_pf_criteria(
                    smooth_trace, pot_place_blocks, neuron_id, save_results=False)
                # if the shuffled trace contained a place cell that passed all 3 criteria, count it
                if len(place_fields_passed) > 0:
                    p_counter += 1

        return p_counter/1000   # return p-value of this neuron (number of place fields after 1000 shuffles)

    def save(self, file_name='pcf_results', overwrite=False):
        """
        Saves PCF object as a pickled file to the root directory.
        :param file_name: str; name of the saved file, defaults to 'pcf_results'
        :param overwrite: bool, overwrites files automatically if there is one
        :return:
        """
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

#%% Visualization
    def plot_binned_neurons(self, idx=None, sliced=False):
        if idx:
            if sliced:
                if type(idx) == int:
                    traces = self.bin_avg_activity[:idx]
                elif type(idx) == list and len(idx) == 2:
                    traces = self.bin_avg_activity[idx[0], idx[2]]
                else:
                    print('Idx has to be either int or list. If sliced, list has to have length 2.')
                    return
            else:
                traces = self.bin_avg_activity[idx]
        else:
            traces = self.bin_avg_activity
        if traces.shape[0] < 30:
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

    def plot_individual_neuron(self, idx, vr_aligned=True):
        if vr_aligned:
            traces = self.bin_activity[idx]
        else:
            traces = self.session[idx]
        fig, ax = plt.subplots(nrows=len(traces), ncols=1, sharex=True, figsize=(20, 12))
        for i in range(len(traces)):
            ax[i].plot(traces[i])
            ax[i].set_title(f'{i + 1}', x=-0.02, y=-0.4)
        if vr_aligned:
            ax[i].set_xlim(0, self.params['n_bins'])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    def plot_single_place_cell(self, idx):
        """
        Plots all trials of a single place cell in a line graph and pcolormesh.
        :param idx: Index of the to-be-plotted place cell (following the indexing of self.place_cells, not cnm indexing,
                    i.e. idx=0 shows the first place cell, not the first extracted component)
        :return: figure
        """
        if type(idx) != int:
            return 'Idx has to be a single digit!'

        traces = self.bin_activity[self.place_cells[idx][0]]

        # plot components
        if len(traces.shape) == 1:
            nrows = 1
        else:
            nrows = traces.shape[0]

        trace_fig, trace_ax = plt.subplots(nrows=nrows, ncols=2, sharex=True, figsize=(18, 10))
        trace_fig.suptitle(f'Neuron {self.place_cells[idx][0]}', fontsize=16)
        for i in range(traces.shape[0]):
            curr_trace = traces[i, np.newaxis]
            trace_ax[i, 1].pcolormesh(curr_trace)
            trace_ax[i, 0].plot(traces[i])
            if i == trace_ax[:, 0].size - 1:
                trace_ax[i, 0].spines['top'].set_visible(False)
                trace_ax[i, 0].spines['right'].set_visible(False)
                trace_ax[i, 0].set_yticks([])
                trace_ax[i, 1].spines['top'].set_visible(False)
                trace_ax[i, 1].spines['right'].set_visible(False)
            else:
                trace_ax[i, 0].axis('off')
                trace_ax[i, 1].axis('off')
            trace_ax[i, 0].set_title(f'Trial {i + 1}', x=-0.1, y=0.3)
        trace_ax[i, 0].set_xlim(0, traces.shape[1])
        trace_ax[i, 0].set_xlabel('VR position', fontsize=12)
        trace_ax[i, 1].set_xlabel('VR position', fontsize=12)
        trace_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # trace_fig.tight_layout()
        plt.show()

    def plot_all_place_cells(self, save=False):
        """
        Plots all place cells in the data set by line graph and pcolormesh.
        :param save: bool flag whether the figure should be automatically saved in the root and closed afterwards.
        :return:
        """
        # todo: mark positions of place fields in graph
        # todo: make x-axis labels reflect actual VR position, not bins
        # TODO: scale bars?

        place_cell_idx = [x[0] for x in self.place_cells]
        traces = self.bin_avg_activity[place_cell_idx]
        n_neurons = traces.shape[0]

        max_bins = []
        for i in range(n_neurons):
            max_bins.append((i, np.argmax(traces[i, :])))

        max_bins_sorted = sorted(max_bins, key=lambda tup: tup[1])

        trace_fig, trace_ax = plt.subplots(nrows=n_neurons, ncols=2, sharex=True, figsize=(18, 10))
        mouse = self.params['mouse']
        session = self.params['session']
        network = self.params['network']
        trace_fig.suptitle(f'All place cells of mouse {mouse}, session {session}, network {network}', fontsize=16)
        for i in range(n_neurons):
            curr_neur = max_bins_sorted[i][0]
            curr_trace = traces[curr_neur, np.newaxis]
            trace_ax[i, 1].pcolormesh(curr_trace)
            trace_ax[i, 0].plot(traces[curr_neur])
            if i == trace_ax[:, 0].size - 1:
                trace_ax[i, 0].spines['top'].set_visible(False)
                trace_ax[i, 0].spines['right'].set_visible(False)
                trace_ax[i, 0].set_yticks([])
                trace_ax[i, 1].spines['top'].set_visible(False)
                trace_ax[i, 1].spines['right'].set_visible(False)
            else:
                trace_ax[i, 0].axis('off')
                trace_ax[i, 1].axis('off')
            trace_ax[i, 0].set_title(f'Neuron {place_cell_idx[curr_neur]}', x=-0.1, y=0)
        trace_ax[i, 0].set_xlim(0, traces.shape[1])
        trace_ax[i, 0].set_xlabel('VR position', fontsize=12)
        trace_ax[i, 1].set_xlabel('VR position', fontsize=12)
        trace_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        if save:
            plt.savefig(os.path.join(self.params['root'], 'place_cells.png'))
            plt.close()

