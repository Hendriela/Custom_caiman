#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Hendrik Heiser
# created on 11 October 2019

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import os
import glob


class PlaceCellFinder:
    """
    Class that holds all the data, parameters and results to perform place cell analysis.
    The analysis steps and major parameters are mainly adapted from Dombeck2007/2010, Hainmüller2018 and Koay2019.
    """
    def __init__(self, cnmf, params=None):
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
        :param params: dictionary that holds the pipeline's parameters. Can be initialized now or filled later.
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

        if params is not None:
            self.params = params
        else:
            self.params = {'root': None,            # main directory of that session
                           'trans_length': 0.5,     # minimum length in seconds of a significant transient
                           'trans_thresh': 4,       # factor of sigma above which a transient is significant; int or
                                                    # tuple of int (for different start and end of transients)
                           'n_bins': 100,           # number of bins per trial in which to group the dF/F traces
                           'bin_window_avg': 3,     # sliding window of bins (left and right) for trace smoothing
                           'bin_base': 0.25,        # fraction of lowest bins that are averaged for baseline calculation
                           'place_thresh': 0.25,    # threshold of being considered for place fields, calculated
                                                    # from difference between max and baseline dF/F
                           'min_bin_size': 10,      # minimum size in bins for a place field (should correspond to 15-20 cm) #TODO make it accept cm values and calculate n_bins through track length
                           'fluo_infield': 7,       # factor above which the mean DF/F in the place field should lie compared to outside the field
                           'trans_time': 0.2,       # fraction of the (unbinned!) signal while the mouse is located in
                                                    # the place field that should consist of significant transients
                           'n_splits': 10,          # segments the binned DF/F should be split into for bootstrapping. Has to be a divisor of n_bins
                           'track_length': None,    # length of the VR corridor track during this trial in cm

            # The following parameters are calculated during analysis and do not have to be set by the user
                           'frame_list': None,      # list of number of frames in every trial in this session
                           'trial_list': None,  # list of trial folders in this session
                           'n_neuron': None,        # number of neurons that were detected in this session
                           'n_trial': None,         # number of trials in this session
                           'sigma': None,           # array[n_neuron x n_trials], noise level (from FWHM) of every trial
                           'bin_frame_count': None} # array[n_bins x n_trials], number of frames averaged in each bin

        # find directories, files and frame counts
        self.params['frame_list'] = []
        for trial in self.params['trial_list']:
            if len(glob.glob(trial+'//*.mmap')) != 0:
                self.params['frame_list'].append(int(glob.glob(trial+'//*.mmap')[0].split('_')[-2]))
            else:
                print(f'No memmap files found at {trial}. Run motion correction before initializing PCF object!')

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

        print('\nSuccessfully separated traces into trials and sorted them by neurons.\nResults are stored in pcf.session.\n')
        #return self

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

    def align_to_vr_position(self):
        """
        Imports behavioral data (merged_vr_licks.txt) and aligns the traces to it. Then bins the traces to
        achieve a uniform trial length and subsequent place cell analysis.
        Behavioral data is saved in the 'behavior' list that includes one array per trial with the following structure:
            time stamp -> VR position -- time stamp -> lick sensor -- 2p trigger -- time stamp -> encoder (speed)
        Binned data is saved is saved in three formats:
            - as bin_frame_count, an array of shape [n_bins x n_trials] showing the number of frames that have to be
              averaged for each bin in every trial (stored in params)
            - as bin_activity, a list of neurons that consist of an array with shape [n_bins x n_trials] and stores
              the binned dF/F for every trial.
            - as bin_avg_activity, an array of shape [n_neuron x n_bins] that contain the dF/F for each bin of every
              neuron, averaged across trials. This is what will mainly be used for place cell analysis.
        :return: Updated PCF object with behavior and binned trial data
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
                print(f'Frame count not matching in trial {i}: Frame list says {frame_list_count}, import says {np.sum(bin_frame_count[:, i])}')

        # Average dF/F for each neuron for each trial for each bin
        # goes through every trial, extracts frames according to current bin size, averages it and puts it into
        # the data structure "bin_activity", a list of neurons, with every neuron having an array of shape
        # (n_trials X n_bins) containing the average dF/F activity of this bin of that trial
        bin_activity = list(np.zeros(self.params['n_neuron']))
        for neuron in range(self.params['n_neuron']):
            curr_neur_act = np.zeros((self.params['n_trial'], self.params['n_bins']))
            for trial in range(self.params['n_trial']):
                curr_trace = self.session[neuron][trial]
                curr_bins = bin_frame_count[:, trial]
                curr_act_bin = np.zeros(self.params['n_bins'])
                for bin_no in range(self.params['n_bins']):
                    # extract the trace of the current bin from the trial trace
                    if len(curr_trace) > curr_bins[bin_no]:
                        trace_to_avg, curr_trace = curr_trace[:curr_bins[bin_no]], curr_trace[curr_bins[bin_no]:]
                    elif len(curr_trace) == curr_bins[bin_no]:
                        trace_to_avg = curr_trace
                    else:
                        trace_to_avg = np.nan
                        raise Exception('Something went wrong during binning...')
                    if trace_to_avg.size > 0:
                        curr_act_bin[bin_no] = np.mean(trace_to_avg)
                    else:
                        curr_act_bin[bin_no] = 0
                curr_neur_act[trial] = curr_act_bin
            bin_activity[neuron] = curr_neur_act

        # Get average activity across trials of every neuron for every bin
        bin_avg_activity = np.zeros((self.params['n_neuron'], self.params['n_bins']))
        for neuron in range(self.params['n_neuron']):
            bin_avg_activity[neuron] = np.mean(bin_activity[neuron], axis=0)

        self.behavior = behavior
        self.params['bin_frame_count'] = bin_frame_count
        self.bin_activity = bin_activity
        self.bin_avg_activity = bin_avg_activity

        print('\nSuccessfully aligned traces with VR position and binned them to an equal length.\n' +
              'Results are stored in pcf.bin_activity and pcf.bin_avg_activity.\n')

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

    def find_place_cells(self):
        """
        Wrapper function that checks every neuron for place fields by calling find_place_field_neuron().
        :return:
        """
        pass

    def find_place_field_neuron(self, data, neuron_id):
        """
        Performs place field analysis (smoothing, pre-screening and criteria application) on a single neuron data set.
        #TODO remove other potential place fields from outside field dF/F value
        :param data: 1D array, binned and across-trial-averaged dF/F data of one neuron, e.g. from bin_avg_activity
        :param neuron_id: ID of the neuron that the data belongs to (its index in session list)
        :return:
        """
        # smoothing binned data by averaging over adjacent bins
        smooth_trace = self.smooth_trace(data)

        # pre-screening for potential place fields
        pot_place_blocks = self.pre_screen_place_fields(smooth_trace)

        # if the trace has above-threshold values, continue with place cell criteria
        if len(pot_place_blocks) > 0:
            place_fields_passed = self.apply_pf_criteria(smooth_trace, pot_place_blocks, neuron_id)
            # if this neuron has one or more place fields that passed all three criteria, validate via bootstrapping
            if len(place_fields_passed) > 0:
                p_value = self.bootstrapping(data, neuron_id)
                # if the p_value is lower than 0.05, accept the current cell as a place cell
                if p_value < 0.05:
                    self.place_cells.append((neuron_id, place_fields_passed, p_value))
                # if the p_value is higher than 0.05, save it in a separate list
                else:
                    self.place_cells_reject.append((neuron_id, place_fields_passed, p_value))

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
        f_thresh = (f_max - f_base) * self.params['place_thresh']
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

    def apply_pf_criteria(self, trace, place_blocks, neuron_id):
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
        :return: list of place field arrays that passed all three criteria (empty if none passed).
        """
        place_field_passed = []
        for pot_place in place_blocks:
            if self.is_large_enough(pot_place) and self.is_strong_enough(trace, pot_place) and self.has_enough_transients(neuron_id, pot_place):
                place_field_passed.append(pot_place)

        return place_field_passed

    def is_large_enough(self, place_field):
        """
        Checks if the potential place field is large enough according to 'min_bin_size' (criterion 1).

        :param place_field: 1D array of indices of data points that form the potential place field
        :return: boolean value whether the criterion is passed or not
        """
        return place_field.size >= self.params['min_bin_size']

    def is_strong_enough(self, trace, place_field):
        """
        Checks if the place field has a mean dF/F that is 'fluo_infield'x higher than outside the field (criterion 2).

        :param trace: 1D array of the trace data
        :param place_field: 1D array of indices of data points that form the potential place field
        :return: boolean value whether the criterion is passed or not
        """
        pot_place_idx = np.in1d(range(trace.shape[0]), place_field) # get an idx mask for the potential place field
        return np.mean(trace[pot_place_idx]) >= self.params['fluo_infield'] * np.mean(trace[~pot_place_idx])

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
            # get the start and end frame for the current place field from the bin_frame_count array that stores how many
            # frames were pooled for each bin
            curr_place_frames = (np.sum(self.params['bin_frame_count'][:place_field[0], trial]),
                                 np.sum(self.params['bin_frame_count'][:place_field[-1] + 1, trial]))
            # attach the transient-only trace in the place field during this trial to the array
            # TODO: not working with the current behavior data, make it work for behavior&frame-trigger data
            place_frames_trace.append(self.session_trans[neuron_id][trial][curr_place_frames[0]:curr_place_frames[1] + 1])

        # create one big 1D array that includes all frames where the mouse was located in the place field
        # as this is the transient-only trace, we make it boolean, with False = no transient and True = transient
        place_frames_trace = np.hstack(place_frames_trace).astype('bool')
        # check if at least 'trans_time' percent of the frames are part of a significant transient
        return np.sum(place_frames_trace) >= self.params['trans_time'] * place_frames_trace.shape[0]

    def bootstrapping(self, trace, neuron_id):
        """
        Performs bootstrapping on a trace and returns p-value for a place cell in this trace.
        #TODO implement Bartos analysis: shuffle dF/F unbinned data and then bin new
        The trace split in "n_splits" parts which are randomly shuffled 1000 times. Then, place cell detection is
        performed on each shuffled trace. The p-value is defined as the ratio of place cells detected in the shuffled
        traces versus number of shuffles (1000; see Dombeck et al., 2010). If this neuron's trace gets a p-value
        of p < 0.05 (place fields detected in less than 50 shuffles), the place field is accepted.

        :param trace: 1D array, trace data that is to be shuffled
        :param neuron_id: 1D array of index of the current neuron in the session list
        :return: place_field_p, p-value of place fields in this trace/neuron
        """
        try:
            split_trace = np.split(trace, self.params['n_splits'])
        except ValueError:
            raise ValueError('The number of splits for bootstrapping has to be a divisor of the number of bins! Consider changing n_bins or n_splits.') from None

        p_counter = 0
        for i in range(1000):
            # shuffle trace 1000 times and perform place cell analysis on every shuffled trace
            curr_shuffle = random.sample(split_trace, len(split_trace))
            smooth_trace = self.smooth_trace(curr_shuffle)                  # smoothing
            pot_place_fields = self.pre_screen_place_fields(smooth_trace,)  # pre-screening
            if len(pot_place_fields) > 0:                                   # criteria testing
                place_fields = self.apply_pf_criteria(smooth_trace, pot_place_fields, neuron_id)
            if len(place_fields):
                p_counter += 1      # if the shuffled trace contained a place cell that passed all 3 criteria, count it

        return p_counter/1000   # return p-value of this neuron
