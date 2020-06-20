import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import gc

# params to set
MD = yaml.load(open("custom scripts/metadata.yaml"), Loader=yaml.FullLoader) #metadata

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
        activity_matrix : 2D array containing (float, dim1 = bins, dim2 = neurons)
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


def across_mice(block, max_delta_bins=30 ):
    """Save mean-pvc curve (of a block) averaged across mice

        Parameters
        ----------
        block : string name of block e.g. "block_2" (see metadata.yaml for names)

       Returns
       -------
       curve_yvals:
           saves plot of mean pvc curve to file
    """

    data_path = MD["config"]["data_path_batch3"]
    md_block = MD["data"][block]
    num_mice = len(md_block["mice"])
    final_yvals = np.zeros(max_delta_bins + 1)
    for mouse in md_block["mice"]:
        mouse_yvals = []
        print(mouse)
        for session in md_block["sessions"]:
            if mouse in md_block["sessions"][session]["mice"]:  # skip session if it wasn't recorded for this mouse
                activity_matrix = np.load(data_path + "\\" + mouse + "\\" + str(session) + "\\" + "bin_avg_activity.npy")
                session_yvals = pvc_curve(np.transpose(activity_matrix,(1,0)), plot=False)
                mouse_yvals.append(list(session_yvals))
                fig_session = plot_pvc_curve(session_yvals, show=True)
                fig_session.savefig(data_path + "\\" + mouse + "\\" + str(session) + "\\" + "pvc_curve" + ".png")
                plt.close(fig_session)
        final_yvals += np.array(mouse_yvals).mean(axis=0) / num_mice
        #fig_trial = plot_pvc_curve(final_yvals, show=True)
    fig_block = plot_pvc_curve(final_yvals)
    print(data_path + "\\" + "pvc_curve" + block + ".png")
    fig_block.savefig(data_path + "\\" + "pvc_curve_" + block + ".png")
    plt.close(fig_block)


def main():
    across_mice("block_2")


if __name__ == "__main__":
    main()
