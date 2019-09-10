#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import seaborn as sns
#import tkinter as tk
#from tkinter import messagebox as ms


#%%

def save_results(run_name=''):
    ### SAVES CNMF, DECONV AND CNN RESULTS ####
    # If called before estimates.select_components, all components with corresponding idx_comp are saved. #
    # If called afterwards, all available components (only good ones) are saved. #

    # comp_available = True  # assume that component selection is available

    dirname = fnames[0][:-4] + "_analysis"
    os.makedirs(os.path.dirname(dirname), exist_ok=True)  # create directory for caiman results if necessary

    timestamp = str(datetime.now())
    curr_time = timestamp[:4] + timestamp[5:7] + timestamp[8:10] + '_' + timestamp[11:13] + timestamp[
                                                                                            14:16] + timestamp[17:19]
    if run_name:
        run_dir = dirname + '/' + run_name
    else:
        run_dir = dirname + '/run_' + curr_time  # set runname to timestamp if none was provided
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)  # create directory for current run

    # save parameters of the current run
    filename = run_dir + '/params.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as param_file:
        param_file.write(str(cnm2.params))

    # all components are being saved, together with indices of good and bad components to untangle it later
    # save denoised and deconvolved neural activity
    filename = run_dir + '/denoise_deconv_act.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.C)
    # save df/f normalized neural activity
    filename = run_dir + '/df_f_norm.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.F_dff)
    # save deconvolved spikes
    filename = run_dir + '/spikes.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.S)
    if cnm2.estimates.idx_components:
        # save good/bad component indices
        filename = run_dir + '/idx_comp_good.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, cnm2.estimates.idx_components)
        print('Good and bad components with idx_comp_good indices are saved!\n')
    else:
        print('Only good components are saved!')


def load_results(run_dir):
    # run_dir = '/Users/hheiser/Desktop/testing data/file_00020_no_motion/raw_uncut_analysis/run_20190618_160555'
    filename = run_dir + '/denoise_deconv_act.gz'
    activity = np.loadtxt(filename)

    filename = run_dir + '/df_f_norm.gz'
    dff_norm = np.loadtxt(filename)

    filename = run_dir + '/spikes.gz'
    spikes = np.loadtxt(filename)


def plot_component_traces(comp_array='', param='F_dff'):
    # plots components in traces and color-graded
    # comp_array is a 2D array with single components in rows and measurements in columns

    if comp_array:
        traces = comp_array
    else:
        try:
            traces = getattr(cnm2.estimates, param)
        except NameError:
            print('Could find no component file! Run pipeline or load old results!/n')
            return

    # plot components
    trace_fig, trace_ax = plt.subplots(nrows=traces.shape[0], ncols=2, sharex=True, figsize=(20, 12))
    for i in range(traces.shape[0]):
        curr_trace = traces[i, np.newaxis]
        trace_ax[i, 1].pcolormesh(curr_trace)
        trace_ax[i, 0].plot(traces[i])
        if i == trace_ax[:, 0].size - 1:
            trace_ax[i, 0].spines['top'].set_visible(False)
            trace_ax[i, 0].spines['right'].set_visible(False)
            trace_ax[i, 1].spines['top'].set_visible(False)
            trace_ax[i, 1].spines['right'].set_visible(False)
        else:
            trace_ax[i, 0].axis('off')
            trace_ax[i, 1].axis('off')
    trace_ax[i, 0].set_xlim(0, 1000)
    trace_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # trace_fig.tight_layout()
    plt.show()


def check_correlation(thresh, param='F_dff'):
    # Check highly correlated components to see if they are actually two separate components or the same unit

    # # plot traces of correlated components under each other
    # plot_correlated_traces(thresh=thresh, param=param)
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Correlated component pairs')

    # get indices of highly correlating components
    high_corr = np.where(half_correlation_matrix(correlate_components(param)) >= thresh)

    # plot both component
    corr_fig = plt.figure()
    corr_fig.canvas.set_window_title('Correlating components')
    plt1 = plt.subplot(1, 2, 1)
    cm.utils.visualization.plot_contours(cnm2.estimates.A[:, [high_corr[0][0]]], Cn=Cn, display_numbers=True)
    plt1.title(f'Main component no. {high_corr[0][0] + 1}')
    plt2 = plt.subplot(1, 2, 2)
    cm.utils.visualization.plot_contours(cnm2.estimates.A[:, high_corr[1][0]], Cn=Cn, display_numbers=True)
    plt2.title(f'Reference component no. {high_corr[1][0] + 1}')

    # Initialize callback functions
    class Index(object):
        pressed = 'init'

        def pressed_left(self, event):
            process_press(button='left')

        def pressed_right(self, event):
            process_press(button='right')

        def pressed_both(self, event):
            process_press(button='both')

        def pressed_none(self, event):
            process_press(button='none')

    callback = Index()

    ####### DRAW FIGURE ########
    # make space for the buttons under the figure
    plt.subplots_adjust(bottom=0.2)
    # create the axes for the buttons
    ax_text = plt.axes([0.1, 0.05, 0.1, 0.075])
    ax_text.axis('off')
    ax_left = plt.axes([0.48, 0.05, 0.1, 0.075])
    ax_right = plt.axes([0.59, 0.05, 0.1, 0.075])
    ax_both = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_none = plt.axes([0.81, 0.05, 0.1, 0.075])
    ax_text.text(1, 0, 'Which component(s) \nshould be DELETED?')
    b_left = Button(ax_left, 'Left')
    b_left.on_clicked(callback.pressed_left)
    b_right = Button(ax_right, 'Right')
    b_right.on_clicked(callback.pressed_right)
    b_both = Button(ax_both, 'Both')
    b_both.on_clicked(callback.pressed_both)
    b_none = Button(ax_none, 'None')
    b_none.on_clicked(callback.pressed_none)

    # if i == 0:
    #     plt.show()
    # else:
    #     plt.draw()

    pressed_buttons = []

    def process_press(button=None):
        global pressed
        if button is not None:
            pressed_buttons.append(button)


def correlate_components(param='F_dff'):
    traces = pd.DataFrame(np.transpose(getattr(cnm2.estimates, param)))
    trace_corr = half_correlation_matrix(traces.corr())
    return trace_corr


def half_correlation_matrix(matrix):
    # halves a full correlation matrix to remove double values
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    masked_matrix = pd.DataFrame(np.ma.masked_array(matrix, mask=mask))
    return masked_matrix


def plot_comp_correlation(param='F_dff', half=True):
    # plots correlation matrix of the estimates-component parameter param
    # half: whether correlation matrix should be plotted completely or half (no double values)
    trace_corr = correlate_components(param)
    if half:
        sns.heatmap(half_correlation_matrix(trace_corr))
    else:
        sns.heatmap(trace_corr)


def plot_correlated_traces(thresh, param='F_dff'):
    # plots components that are correlated by at least thresh-value
    # corr can be a full or half correlation matrix

    corr = correlate_components(param)
    traces = getattr(cnm2.estimates, param)

    # find all component pairs that have a correlation coefficient higher than thresh
    high_corr = np.where(corr >= thresh)
    pair_idx = tuple(zip(high_corr[0], high_corr[1]))

    # plot
    corr_plot, corr_ax = plt.subplots(nrows=len(high_corr[0]), ncols=2, sharex=True,
                                      figsize=(50, 12))  # create figure+subplots
    for i in range(corr_ax.size):
        # For each correlating component-pair...
        curr_idx_1 = pair_idx[i][0]
        curr_idx_2 = pair_idx[i][1]
        # ... first plot the calcium traces on the same figure
        corr_ax[i, 0].plot(traces[curr_idx_1], lw=1)
        corr_ax[i, 0].plot(traces[curr_idx_2], lw=1)

        # ... then plot the color-graded activity below each other
        trace_pair = np.vstack((traces[curr_idx_1], traces[curr_idx_2]))
        corr_ax[i, 1].pcolormesh(trace_pair)
        corr_ax[i, 1].set_ylim(0, 2)

        # adjust graph layout
        if i == corr_ax[:, 0].size - 1:
            corr_ax[i, 0].spines['top'].set_visible(False)
            corr_ax[i, 0].spines['right'].set_visible(False)
            corr_ax[i, 1].spines['top'].set_visible(False)
            corr_ax[i, 1].spines['right'].set_visible(False)
        else:
            corr_ax[i, 0].axis('off')
            corr_ax[i, 1].axis('off')
        corr_ax[i, 0].title.set_text(f'Component {curr_idx_1} & {curr_idx_2} (r = {corr[curr_idx_2][curr_idx_1]}')
    corr_ax[i, 0].set_xlim(0, 1000)
    corr_plot.constrained_layout()
    plt.show()


# def show_window():
#     master = tk.Tk()
#     master.title('Component selection')
#
#     pressed = ''
#
#     question = ttk.Label(master, text='Which of these neurons do you want to delete?')
#     question.grid(column=0,row=0)
#
#     def pressed_left():
#         pressed = 'left'
#     def pressed_right():
#         pressed = 'right'
#     def pressed_both():
#         pressed = 'both'
#     def pressed_none():
#         pressed = 'none'
#
#     left_button = ttk.Button(master, text='Left', command=pressed_left)
#     #left_button.pack()
#     left_button.grid(column = 0,row = 2)
#     right_button = ttk.Button(master, text='Left', command=pressed_right)
#     #right_button.pack()
#     right_button.grid(column=1, row=2)
#     both_button = ttk.Button(master, text='Left', command=pressed_both)
#     #both_button.pack()
#     both_button.grid(column=2, row=2)
#     none_button = ttk.Button(master, text='Left', command=pressed_none)
#     #none_button.pack()
#     none_button.grid(column=3, row=2)
#
#     master.mainloop()
#
#     return pressed


# def show_window():
#     window = tk.Tk()
#     window.wm_withdraw()
#     result = ms.askquestion('Component selection', 'Do you want to DELETE this component?')
#     return result
