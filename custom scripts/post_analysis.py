import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
import math
import caiman as cm

#%% INITIAL RESULTS SCREENING

def check_eval_results(cnm, idx):
    """Checks results of component evaluation and determines why the component got rejected or accepted

    Args:
        cnm:                caiman CNMF object containing estimates and evaluate_components() results

        idx:                int or iterable (array, list...)
                            index or list of indices of components to be checked

    Returns:
        printout of evaluation results
    """
    try:
        iter(idx)
        idx = list(idx)
    except:
        idx = [idx]

    snr_min = cnm.params.quality['SNR_lowest']
    snr_max = cnm.params.quality['min_SNR']
    r_min = cnm.params.quality['rval_lowest']
    r_max = cnm.params.quality['rval_thr']
    cnn_min = cnm.params.quality['cnn_lowest']
    cnn_max = cnm.params.quality['min_cnn_thr']

    for i in range(len(idx)):
        snr = cnm.estimates.SNR_comp[idx[i]]
        r = cnm.estimates.r_values[idx[i]]
        cnn = cnm.estimates.cnn_preds[idx[i]]
        cnn_round = str(round(cnn, 2))

        red_start = '\033[1;31;49m'
        red_end = '\033[0;39;49m'

        green_start = '\033[1;32;49m'
        green_end = '\033[0;39;49m'

        upper_thresh_failed = 0
        lower_thresh_failed = False

        print(f'Checking component {idx[i]+1}...')
        if idx[i] in cnm.estimates.idx_components:
            print(green_start+f'\nComponent {idx[i]+1} got accepted, all lower threshold were passed!'+green_end+'\n\n\tUpper thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds threshold of {snr_max}\n'+green_end)
            else:
                print(f'\tSNR of {round(snr,2)} does not exceed threshold of {snr_max}\n')

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds threshold of {r_max}\n'+green_end)
            else:
                print(f'\tR-value of {round(r,2)} does not exceed threshold of {r_max}\n')

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            else:
                print('\tCNN-value of '+cnn_round+f' does not exceed threshold of {cnn_max}\n')
            print(f'\n')

        else:
            print(f'\nComponent {idx[i] + 1} did not get accepted. \n\n\tChecking thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds upper threshold of {snr_max}\n'+green_end)
            elif snr >= snr_min and snr < snr_max:
                print(f'\tSNR of {round(snr,2)} exceeds lower threshold of {snr_min}, but not upper threshold of {snr_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+f'\tSNR of {round(snr,2)} does not pass lower threshold of {snr_min}\n'+red_end)
                lower_thresh_failed = True

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds upper threshold of {r_max}\n'+green_end)
            elif r >= r_min and r < r_max:
                print(f'\tR-value of {round(r,2)} exceeds lower threshold of {r_min}, but not upper threshold of {r_max}\n')
                upper_thresh_failed += 1
            else:
                print(f'\tR-value of {round(r,2)} does not pass lower threshold of {r_min}\n')
                lower_thresh_failed = True

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            elif cnn >= cnn_min and cnn < cnn_max:
                print('\tCNN-value of '+cnn_round+f' exceeds lower threshold of {cnn_min}, but not upper threshold of {cnn_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+'\tCNN-value of '+cnn_round+f' does not pass lower threshold of {cnn_min}\n'+red_end)
                lower_thresh_failed = True

            if lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it failed at least one lower threshold!\n\n'+red_end)
            elif upper_thresh_failed == 3 and not lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it met all lower, but no upper thresholds!\n\n'+red_end)
            else:
                print('This should not appear, check code logic!\n\n')


def plot_component_traces(cnm, idx=None, param='F_dff'):
    # plots components in traces and color-graded
    # comp_array is a 2D array with single components in rows and measurements in columns

    try:
        traces = getattr(cnm.estimates, param)
        if idx is not None:
            traces = traces[idx]
    except NameError:
        print('Could find no component data! Run pipeline or load old results!')
        return

    # plot components
    if len(traces.shape) == 1:
        nrows = 1
    else:
        nrows = traces.shape[0]
    trace_fig, trace_ax = plt.subplots(nrows=nrows, ncols=2, sharex=True, figsize=(20, 12))
    trace_fig.suptitle(f'Parameter {param} of selected components', fontsize=16)
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
        trace_ax[i, 0].set_title(f'{i+1}', x=-0.02, y=-0.4)
    trace_ax[i, 0].set_xlim(0, 1000)
    trace_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # trace_fig.tight_layout()
    plt.show()


def reject_component(cnm, idx):
    """
    Re-assigns component from the good to the bad component array

    :param cnm: cnm object that stores the estimates object
    :param idx: index this component has in the idx_components list
    """
    # add component to the rejected list
    cnm.estimates.idx_components_bad = np.append(cnm2.estimates.idx_components_bad, cnm2.estimates.idx_components[idx])
    # remove component from the accepted list
    cnm.estimates.idx_components = np.delete(cnm.estimates.idx_components, idx)
    return cnm


def accept_component(cnm, idx):
    """
    Re-assigns component from the bad to the good component array

    :param cnm: cnm object that stores the estimates object
    :param idx: index this component has in the idx_components_bad list
    """
    # add component to the accepted list
    cnm.estimates.idx_components = np.append(cnm.estimates.idx_components, cnm.estimates.idx_components_bad[idx])
    # remove component from the rejected list
    cnm.estimates.idx_components_bad = np.delete(cnm.estimates.idx_components_bad, idx)
    return cnm


def get_noise_fwhm(data):
    """
    Returns noise level as standard deviation (sigma) from a dataset using full-width-half-maximum (Koay et al. 2019)
    :param data: 1D array of data that you need the noise level from
    :return: noise: float, sigma of given dataset
    """
    if np.all(data == 0): # catch trials without data
        sigma = 0
    else:
        x_data, y_data = sns.distplot(data).get_lines()[0].get_data()
        plt.close()
        y_max = y_data.argmax()  # get idx of half maximum
        # get points above/below y_max that is closest to max_y/2 by subtracting it from the data and
        # looking for the minimum absolute value
        nearest_above = (np.abs(y_data[y_max:] - max(y_data) / 2)).argmin()
        nearest_below = (np.abs(y_data[:y_max] - max(y_data) / 2)).argmin()
        # get FWHM by subtracting the x-values of the two points
        FWHM = x_data[nearest_above + y_max] - x_data[nearest_below]
        # return noise level as FWHM/2.3548
        sigma = FWHM/2.3548
    return sigma


#%% CORRELATION FUNCTIONS


def half_correlation_matrix(matrix):
    # halves a full correlation matrix to remove double values
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    masked_matrix = pd.DataFrame(np.ma.masked_array(matrix, mask=mask))
    return masked_matrix


def correlate_components(param='F_dff'):
    traces = pd.DataFrame(np.transpose(getattr(cnm2.estimates, param)))
    trace_corr = half_correlation_matrix(traces.corr())
    return trace_corr


def check_correlation(thresh, param='F_dff'): #TODO: GET THIS TO WORK
    # Check highly correlated components to see if they are actually two separate components or the same unit

    # # plot traces of correlated components under each other
    # plot_correlated_traces(thresh=thresh, param=param)
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Correlated component pairs')

    comp_count = 0
    # get indices of highly correlating components
    high_corr = np.where(half_correlation_matrix(correlate_components(param)) >= thresh)
    comp_to_del = {i: False for i in np.unique(np.concatenate((high_corr[0], high_corr[1])))}

    # Initialize callback functions
    class Index(object):

        def pressed_left(self, event):
            process_press(button='left')

        def pressed_right(self, event):
            process_press(button='right')

        def pressed_both(self, event):
            process_press(button='both')

        def pressed_none(self, event):
            process_press(button='none')

    # define what happens when you press a button
    def process_press(button=None):
        # The component(s) are marked for deletion when the corresponding button is pressed
        # Notice: one component can correlate with several others; if this component is marked once for deletion,
        # this is irreversible!!
        global comp_count
        global high_corr
        left_comp = high_corr[0][comp_count]
        right_comp = high_corr[1][comp_count]
        if button == 'left':
            comp_to_del[left_comp] = True
        elif button == 'right':
            comp_to_del[right_comp] = True
        elif button == 'both':
            comp_to_del[right_comp] = True
            comp_to_del[left_comp] = True
        elif button == 'none':
            pass

        # plot new components
        comp_count += 1
        plot_next_pair(high_corr[0][comp_count], high_corr[1][comp_count])

    callback = Index()

    # how to plot a pair of correlated components
    def plot_next_pair():
        global plt1
        global plt2
        global high_corr
        global comp_count
        # plot current correlation pairs
        if comp_count < high_corr[0].size:
            plt1.title(f'Main component no. {high_corr[0][comp_count] + 1}')
            cm.utils.visualization.plot_contours(cnm2.estimates.A[:, [high_corr[0][comp_count]]], Cn=Cn, display_numbers=True)
            plt2.title(f'Reference component no. {high_corr[1][comp_count] + 1}')
            cm.utils.visualization.plot_contours(cnm2.estimates.A[:, high_corr[1][comp_count]], Cn=Cn, display_numbers=True)
            if comp_count == 0:
                plt1.show()
                plt2.show()
            else:
                plt1.draw()
                plt2.draw()
        # when all components have been looped through, give a final overview about all to-be-deleted components
        else:
            corr_fig.close()
            # get a list of to-be-deleted components
            del_comp_list = np.transpose(np.asarray([k for k, v in comp_to_del.items() if v]))
            ######## simple command prompt solution
            answer = None
            while answer not in ('yes', 'y', 'no', 'n'):
                answer = input(f'These components were selected for deletion:/n/n/t {del_comp_list}./n/n/t Do you confirm? [y/n]')
                if answer == "yes" or answer == 'y':
                    pass
                elif answer == "no" or answer == 'n':
                    pass
                else:
                    print("Please enter yes or no.")

            ######## fancy plotting solution
            nrows = int(np.sqrt(del_comp_list.size))
            confirm_fig, conf_axs = plt.subplots(nrows=nrows,ncols=nrows)
            confirm_fig.canvas.set_window_title('Confirmation')
            comp_counter = 0

            # plot all backgrounds with label
            for curr_ax in conf_axs.ravel():
                curr_ax.text(0.05, 0.10, f'{del_comp_list[comp_counter]}', transform=curr_ax.transAxes, fontsize=10,
                        verticalalignment='top')
                plt.axes(curr_ax)
                cm.utils.visualization.plot_contours(cnm2.estimates.A[:, [del_comp_list[comp_counter]]], Cn=Cn, display_numbers=True)
                curr_ax.axis('off')
                comp_counter += 1

            # initialize buttons
            class IndexConf(object):

                def pressed_yes(self, event):
                    process_press_conf(button='yes')

                def pressed_no(self, event):
                    process_press_conf(button='no')

            def process_press_conf(button=None):
                if button == 'yes':
                    idx_bad = np.setdiff1d(np.arange(cnm2.estimates.A.shape[-1]),del_comp_list)
                    cnm2.estimates.select_components(idx_components=idx_bad)
                else:
                    print('Deletion cancelled, all components are still there.')
                    confirm_fig.close()


            callback_conf = IndexConf()
            plt.subplots_adjust(bottom=0.2)
            ax_quest = plt.axes([0.1, 0.05, 0.1, 0.075])
            ax_text.axis('off')
            ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
            ax_quest.text(1, 0, 'These components were flagged/n for deletion! Confirm?')
            b_yes = Button(ax_yes, 'yes')
            b_yes.on_clicked(callback_conf.pressed_yes)
            b_no = Button(ax_no, 'Right')
            b_no.on_clicked(callback_conf.pressed_no)




    ####### DRAW FIGURE ########
    corr_fig = plt.figure()
    corr_fig.canvas.set_window_title('Correlating components')
    plt1 = plt.subplot(1, 2, 1)
    plt2 = plt.subplot(1, 2, 2)
    # make space for the buttons under the figure
    plt.subplots_adjust(bottom=0.2)

    # Initialize buttons and loop through the components via button clicks (see process_press() and plot_next_pair())
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


def plot_component_correlation(param='F_dff', half=True):
    # plots correlation matrix of the estimates-component parameter param
    # half: whether correlation matrix should be plotted completely or half (no double values)
    trace_corr = correlate_components(param)
    if half:
        sns.heatmap(half_correlation_matrix(trace_corr))
    else:
        sns.heatmap(trace_corr)


def plot_correlated_traces(thresh, param='F_dff',corr=''):
    # plots components that are correlated by at least thresh-value
    # corr can be a full or half correlation matrix

    # correlate the components if no correlation matrix has been provided
    if corr == '':
        corr = correlate_components(param)
        print('Using pairwise Pearson"s correlation coefficient...\n')
    else:
        print('Using correlation provided by the user...\n')

    # load the component traces
    traces = getattr(cnm2.estimates, param)

    # find all component pairs that have a correlation coefficient higher than thresh
    high_corr = np.where(corr >= thresh)
    pair_idx = tuple(zip(high_corr[0], high_corr[1]))

    # plot
    corr_plot, corr_ax = plt.subplots(nrows=len(high_corr[0]), ncols=2, sharex=True,
                                      figsize=(50, 12))  # create figure+subplots
    for i in range(len(high_corr[0])):
        # For each correlating component-pair...
        curr_idx_1 = pair_idx[i][0]
        curr_idx_2 = pair_idx[i][1]
        # ... first plot the calcium traces on the same figure
        corr_ax[i, 0].plot(traces[curr_idx_1], lw=1)
        corr_ax[i, 0].plot(traces[curr_idx_2], lw=1)
        plt.sca(corr_ax[i,0])
        plt.text(-200, 0, f'Comp {curr_idx_1} & {curr_idx_2}\n(r = {round(corr[curr_idx_1][curr_idx_2],2)})')
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
    corr_ax[i, 0].set_xlim(0, 1000)
    #corr_plot.constrained_layout()
    plt.show()


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    C = np.column_stack([C, np.ones(C.shape[0])]) # add a column of 1s as an intercept
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_j)
            res_i = C[:, i] - C[:, idx].dot(beta_i)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def partial_cross_correlation():
    # calculates cross correlation of trace pairs, but correcting for activity of all other neurons to eliminate
    # general population changes

    # initialize correlation matrix
    traces = cnm2.estimates.F_dff
    n_comp = traces.shape[0]
    part_corr = np.zeros((n_comp,n_comp))

    # calculate the partial correlation for each pair of neurons while controlling for the activity of all other neurons
    for i in range(n_comp):
        for j in range(n_comp):
            # this is done for every pair of neurons (also with itself)

            # get mean fluorescence of the remaining neuron population
            pop_mean = np.delete(traces,[i,j],axis=0).mean(axis=0)

            # calculate partial correlation of i and j, while controlling for pop_mean and put it in the corresponding
            # place in the array
            corr_in = np.transpose(np.vstack((traces[i],traces[j],pop_mean)))
            part_corr[i,j] = partial_corr(corr_in)[1,0]

    sns.heatmap(half_correlation_matrix(part_corr))

    plot_correlated_traces(thresh=0.4,corr=np.asarray(half_correlation_matrix(part_corr)))


def plot_cross_correlation(thresh):
    # plots cross correlation of trace pairs as a function of lag

    corr_coef = correlate_components()
    #high_corr_idx = np.where(corr_coef >= thresh)
    high_corr_idx = (np.random.randint(1,30,10),np.random.randint(1,30,10))
    corr_fig = plt.figure(figsize=(50,12))
    outer = gridspec.GridSpec(math.ceil(len(high_corr_idx[0])/2),2)
    for i in range(len(high_corr_idx[0])):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        trace_1 = cnm2.estimates.F_dff[high_corr_idx[0][i]]
        trace_2 = cnm2.estimates.F_dff[high_corr_idx[1][i]]

        # calculate the cross-correlation
        npts = len(trace_1)
        sample_rate = 1/30 # in Hz
        #lags = np.arange(start=-(npts*sample_rate)+sample_rate, stop=npts*sample_rate-sample_rate, step=sample_rate)
        lags = np.arange(-npts + 1, npts)
        # remove sample means
        trace_1_dm = trace_1 - trace_1.mean()
        trace_2_dm = trace_2 - trace_2.mean()
        # calculate correlation
        trace_cov = np.correlate(trace_1_dm,trace_2_dm,'full')
        # normalize against std
        trace_corr = trace_cov / (npts * trace_1.std() * trace_2.std())

        for j in range(3):
            ax = plt.Subplot(corr_fig, inner[j])
            if j == 0: # plot trace 1 on top
                ax.plot(trace_1)
                ax.axis('off')
            elif j == 2: # plot trace 2 on the bottom
                ax.plot(trace_2)
                ax.axis('off')
            elif j == 1: # plot correlation trace in the middle
                ax.plot(lags, trace_corr)

            corr_fig.add_subplot(ax)

    corr_fig.show()


def plot_neuron_trials(traces):
    """
    Plots the traces for all trials of this neuron. Traces are not normalized against track position,
    and will have different length dependent on the time of the trial
    :param traces: list of trial data (trial data in 1D array with n_frames length,
    use session[n_neuron] for all trials of one neuron or bin_avg_activity[n_neuron] for averaged trials across neurons)
    :return: plot
    """
    n_trials = len(traces)
    fig, ax = plt.subplots(nrows=n_trials)
    for i in range(n_trials):
        ax[i].plot(traces[i])
    plt.tight_layout()

def plot_activity_track(traces):
    """
    Plots traces of a neuron normalized against track position
    :param traces: 2D array containing bin-averaged activity of (n_trials X n_bins; bin_activity[n_neuron])
    :return:
    """
    n_trials = traces.shape[0]
    fig, ax = plt.subplots(nrows=n_trials, ncols=2)
    for i in range(n_trials):
        ax[i,0].plot(traces[i])
        ax[i,1].pcolormesh(traces[i,np.newaxis])
        if i == ax[:, 0].size - 1:
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].set_yticks([])
            ax[i, 1].spines['top'].set_visible(False)
            ax[i, 1].spines['right'].set_visible(False)
            ax[i, 1].set_xticks(list(np.linspace(0,100,13)),[np.linspace(-10,110,13)])
        else:
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
    plt.tight_layout()
