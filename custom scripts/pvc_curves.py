import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import gc

# params to set
file_ids = ["M32_20200322", "M32_20200323" , "M41_20200322", "M41_20200323"]
file_folder = '../../CNM_Basic/'


def plot_pvc_curve(y_vals, bin_size=5, show=False):
    """Plots the pvc curve

        Parameters
        ----------
        y_vals : array-like
            data points of the pvc curve (idx = bin distance)
        bin_size : bool, optional
        show : bool, optional

       Returns
       -------
       fig: figure object
           a figure object of the pvc curve
    """
    fig = plt.figure()
    x_axis = np.arange(0., len(y_vals)* bin_size, bin_size)  # bin size
    plt.plot(x_axis, y_vals, figure=fig)
    plt.ylim(bottom=0); plt.ylabel('Mean PVC')
    plt.xlim(left=0); plt.xlabel('Offset Distances (cm)')
    if show:
        plt.show(block=True)
    return fig


def pvc_curve(activity_matrix, plot=True, max_delta_bins=30):
    """Calculate the mean pvc curve

        Parameters
        ----------
        activity_matrix : 2D array containing (float, dim1 = trials, dim2 = neurons)
        plot: bool, optional
        max_delta_bins: int, optional
            max difference in bin distance

       Returns
       -------
       curve_yvals:
           array of mean pvc curve (idx = delta_bin)
    """
    num_bins = np.size(activity_matrix,0)
    num_neurons = np.size(activity_matrix,1)
    curve_yvals = np.empty(max_delta_bins + 1)

    for delta_bin in range(max_delta_bins + 1):
        mean_pvc_delta_bin = 0
        for offset in range(num_bins - delta_bin):
            idx_x = offset
            idx_y = offset + delta_bin
            pvc_xy_num = pvc_xy_den_term1 = pvc_xy_den_term2 = 0
            for neuron in range(num_neurons):
                pvc_xy_num += activity_matrix[idx_x][neuron] * activity_matrix[idx_y][neuron]
                pvc_xy_den_term1 += activity_matrix[idx_x][neuron]*activity_matrix[idx_x][neuron]
                pvc_xy_den_term2 += activity_matrix[idx_y][neuron]*activity_matrix[idx_y][neuron]
            pvc_xy = pvc_xy_num / (math.sqrt(pvc_xy_den_term1*pvc_xy_den_term2))
            mean_pvc_delta_bin += pvc_xy / (num_bins - delta_bin)
        curve_yvals[delta_bin] = mean_pvc_delta_bin

    if plot:
        plot_pvc_curve(curve_yvals, show=True)

    return curve_yvals



def main():
    """ Save plots of the mean pvc curves of the
        (1) average activity and (2) for the activity of each trial of the session

        [Sessions set above in 'file_ids']
    """
    for file_id in file_ids:
        infile = open(file_folder + '/' + file_id + '/' + file_id + '.pickle', 'rb')
        data_complete = pickle.load(infile)
        infile.close(); gc.collect()
        bin_avg_activity = data_complete.bin_avg_activity
        bin_activity = np.transpose(data_complete.bin_activity, (1,2,0))
        # plot pvc curve for average activity for session
        bin_avg_activity = np.transpose(bin_avg_activity, (1,0))
        fig_avg = plot_pvc_curve(pvc_curve(bin_avg_activity, plot=False))
        fig_avg.savefig(file_folder + '/' + file_id + '/' + file_id + '_sessionave_pvc_curve.png')
        plt.close(fig_avg)
        # plot pvc curve for each trial separately
        for trial_num, bin_activity_trial in enumerate(bin_activity, start=1):
            fig_trial = plot_pvc_curve(pvc_curve(bin_activity_trial, plot=False))
            fig_trial.savefig(file_folder + '/' + file_id + '/' + file_id + '_trial' + str(trial_num) + '_pvc_curve.png')
            plt.close(fig_trial)

if __name__ == "__main__":
    main()