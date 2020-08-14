import numpy as np
import standard_pipeline.place_cell_pipeline as pipe
import matplotlib.pyplot as plt

#%%


def align_traces(data, camera_frames, window_size=2):
    """
    Takes a data matrix containing session traces of neurons (from cnm.estimates.F_dff), a list of frame numbers at
    which grasps happened, aligns a trace window to each grasp and averages the trace across grasps. The result is a
    matrix holding the average activity of every neuron around grasps.

    Args:
        data (numpy array of floats)           :    Shape (n_neurons, n_frames), holds traces of all neurons for the
                                                    whole session. Should be from cnm.estimates.F_dff.

        camera_frames (list or 1D array of int) :   List of camera frames at which grasps happened in this session.

        window_size (int)                      :    Half-size of time window in seconds. Traces "window_size" seconds
                                                    before and after the grasp will be returned.

    Returns:
        mean_aligned (numpy array of floats)   :    Shape (n_neurons, window), holds average activity of every neuron
                                                    before and after grasps. The grasp itself happens at frame index
                                                    window/2.

    """
    # Calculate frame size of window
    MICROSCOPE_FR = 30
    window = window_size*MICROSCOPE_FR

    # Transform camera frames into microscope frames
    CAMERA_FR = 59                                              # Get camera frame rate (hard coded)
    grasp_frame_list = []
    for video_nb, video in enumerate(camera_frames):
        video = np.array(video) - 4                 # Subtract the first 4 frames where microscope is off
        # Divide frames by frame rate to get the elapsed time, which you multiply by the microscope frame rate to get the
        # number of microscopy frames that were taken during that time
        micro_frames = (video/CAMERA_FR) * MICROSCOPE_FR
        grasp_frames = np.round(micro_frames).astype(int)      # Round to integers to get frame indices
        grasp_frames += video_nb*2000
        grasp_frame_list.append(grasp_frames)
    grasp_frames = np.hstack(grasp_frame_list)

    # Initialize matrix that stores aligned traces of all grasps
    aligned = np.zeros((len(data), window*2, len(grasp_frames)))
    # Go through all grasps and save the traces of all neurons around the grasping frame
    for grasp_nb, frame in enumerate(grasp_frames):
        window_start = frame - window
        window_end = frame + window
        if window_start < 0:
            window_trace = data[:, :frame+window]                               # Get the first available data points
            nan_padding = np.empty((len(data), abs(window_start)))              # Create empty array of missing frames
            nan_padding[:] = np.nan                                             # Fill it with NaNs to ignore them later
            aligned[:, :, grasp_nb] = np.hstack((nan_padding, window_trace))    # Add NaN-padded window to the matrix
        elif window_end > data.shape[1]:
            window_trace = data[:, frame-window:]                               # Get the last available data points
            nan_padding = np.empty((len(data), abs(window_end-data.shape[1])))  # Create empty numpy array
            nan_padding[:] = np.nan                                             # Fill it with NaNs
            aligned[:, :, grasp_nb] = np.hstack((window_trace, nan_padding))    # Create NaN-padded window
        else:
            aligned[:, :, grasp_nb] = data[:, frame-window:frame+window]
    # Average the trace of every neuron across grasps while ignoring NaNs
    mean_aligned = np.nanmean(aligned, axis=2)

    return mean_aligned


def get_modulated_cells(data, threshold=4, pre_window=None, post_window=None):
    """
    Filters dataset for grasp-modulated neurons by comparing the maximum post-grasp activity with a pre-grasp baseline.
    The threshold is set by a multiple of the baseline standard deviation.

    Args:
        data (numpy array of floats)           :    Shape (n_neurons, n_frames), holds traces of all neurons around the
                                                    aligned frame.

        camera_frames (list or 1D array of int) :   List of camera frames at which grasps happened in this session.

        window_size (int)                      :    Half-size of time window in seconds. Traces "window_size" seconds
                                                    before and after the grasp will be returned.

    Returns:
        mean_aligned (numpy array of floats)   :    Shape (n_neurons, window), holds average activity of every neuron
                                                    before and after grasps. The grasp itself happens at frame index
                                                    window/2.

    """
    if pre_window is None:
        pre_window = (0, int(data.shape[1]/4))                 # Defaults to the first half of the pre-grasp period
    if post_window is None:
        post_window = (int(data.shape[1]/2), data.shape[1])   # Defaults to the whole post-grasp period

    # Get pre-grasp mean and standard deviation
    pre_mean = np.mean(data[:, pre_window[0]:pre_window[1]], axis=1)
    pre_std = np.std(data[:, pre_window[0]:pre_window[1]], axis=1)

    # Get a boolean mask for cells that have a post-grasp maximum dF/F >= than the threshold
    mod_cells = np.max(data[:, post_window[0]:post_window[1]], axis=1) >= (pre_mean + threshold * pre_std)

    return data[mod_cells]





#%% Demo functions

cnm = pipe.load_cnmf(r'W:\Neurophysiology-Storage1\Wahl\Jithin\imaging\M12_Pre_Stroke_Caudal\25\Caudal')
data_full = cnm.estimates.F_dff
timestamps = [[1192, 1915], [560]]

align = align_traces(data, timestamps)

plt.pcolormesh(align)

mod_cells = get_modulated_cells(align)

plt.figure()
for cell in range(len(mod_cells)):
    plt.plot(mod_cells[cell] + cell*0.08)

cell = mod_cells[2]
plt.figure()
plt.plot(cell)
