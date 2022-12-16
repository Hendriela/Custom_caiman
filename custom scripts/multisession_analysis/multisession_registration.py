from standard_pipeline import place_cell_pipeline as pipe
from glob import glob
from caiman import load_memmap
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from skimage.registration import phase_cross_correlation
from math import ceil
from scipy.ndimage import zoom
import os
import pandas as pd
from div import file_manager as fm
from copy import deepcopy
import tifffile as tiff

try:
    from point2d import Point2D
except ModuleNotFoundError:
    print('Could not import point2d, multisession_registration.py might not work.')

#%% LOADING AND AUTOMATICALLY ALIGNING MULTISESSION DATA


def plot_all_aligned(spatial, templates, data_only=False):
    rois = []
    n_sess = len(spatial)
    fig, ax = plt.subplots(ncols=n_sess)
    for i in range(n_sess):
        plt.sca(ax[i])
        out = visualization.plot_contours(spatial[i], templates[i], display_numbers=False)
        curr_sess = dir_list[i].split('\\')[-2]
        ax[i].title.set_text(f'{curr_sess}')
        if data_only:
            rois.append(out)
    if data_only:
        plt.close()
        return rois


def align_multisession_caiman(dir_list):
    spatial, templates, dims, pcf_objects = load_multisession_data(dir_list)
    print('Import complete. Looking for aligned components...')
    spatial_union, assignments, matchings = register_multisession(spatial, dims=dims, templates=templates)

    return spatial_union, assignments, matchings, spatial, templates, pcf_objects


def load_multisession_data(dir_list, place_cell_mode=True):
    """
    Loads CNMF objects from multiple PCFs for multisession registration.
    :param dir_list: list of paths to session folders that should be aligned
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :returns spatial: list of csc matrices from cnmf.estimates.A (one per session) with spatial data of components
    :returns templates: list of template images from cnmf.estimates.Cn (one per session)
    :returns dim: tuple, x and y dimensions of FOV, from templates[0].shape
    :returns pcf_objects: list of PCF objects without CNMF data to save memory, contains place cell info
    """
    spatial_list = []  # list of csc matrices (# pixels X # component ROIs, from cnmf.estimates.A) with spatial info for all components
    templates_list = []  # list of template images for each session (local correlation map; cnmf.estimates.Cn)
    pcf_objects_list = []  # list of pcf objects that contain all other infos about the place cells

    count = 0
    for folder in dir_list:
        count += 1
        if place_cell_mode:
            curr_pcf = pipe.load_pcf(folder)    # load pcf object that includes the cnmf object
            spatial_list.append(curr_pcf.cnmf.estimates.A)
        else:
            curr_pcf = pipe.load_cnmf(folder)
            spatial_list.append(curr_pcf.estimates.A)
        try:
            if place_cell_mode:
                templates_list.append(curr_pcf.cnmf.estimates.Cn)
            else:
                templates_list.append(curr_pcf.estimates.Cn)
        except AttributeError:
            print(f'No local correlation image found in {folder}...')
            movie_file = glob(folder+'memmap__*.mmap')
            if len(movie_file) == 1:
                print(f'\tFound mmap file, local correlation image is being calculated.')
                Yr, dims, T = load_memmap(movie_file[0])
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                if place_cell_mode:
                    curr_pcf.cnmf.estimates.Cn = pipe.get_local_correlation(images)
                    curr_pcf.save(overwrite=True)
                    templates_list.append(curr_pcf.cnmf.estimates.Cn)
                else:
                    curr_pcf.estimates.Cn = pipe.get_local_correlation(images)
                    pipe.save_cnmf(curr_pcf, overwrite=True)
                    templates_list.append(curr_pcf.estimates.Cn)
            elif len(movie_file) > 1:
                print('\tResult ambiguous, found more than one mmap file, no calculation possible.')
            else:
                print('\tNo mmap file found. Maybe you have to re-motioncorrect the movie?')

        pcf_objects_list.append(curr_pcf)    # add rest of place cell info to the list
        sep = '\\'
        print(f'Successfully loaded data from {sep.join(folder.split(os.path.sep)[-3:])} ({count}/{len(dir_list)}).')

    dimensions = templates_list[0].shape  # dimensions of the FOV, can be gotten from templates

    return spatial_list, templates_list, dimensions, pcf_objects_list


def plot_all_cells_multisession(direct_list, spatial_list, template_list):
    sess_dates = []
    for path in direct_list:
        sess_dates.append(path.split(os.path.sep)[-2])
    n_sess = len(spatial_list)
    if n_sess < 5:
        n_cols = n_sess
        n_rows = 1
    else:
        n_cols = 4
        n_rows = ceil(n_sess / n_cols)  # number of rows needed to plot into 5 columns

    fig1, ax1 = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    count = 0
    for rows in range(n_rows):
        for cols in range(n_cols):
            plt.sca(ax1[rows, cols])
            try:
                visualization.plot_contours(spatial_list[count], template_list[count], display_numbers=False)
                ax1[rows, cols].set_title(sess_dates[count])
            except IndexError:
                pass
            count += 1

#%% SEMI-MANUALLY ALIGNING PLACE CELLS ACROSS SESSIONS


def save_alignment(directory, align_array, ref_sess_date, pcf_list, place_cell_mode=True):
    """
    Saves manual alignment array to a txt file with session dates as headers for each column. NaNs (no match found)
    are replaced by -10 to maintain visibility.
    :param align_array: data array, shape (n_placecells, n_sessions), from manual_place_cell_alignment()
    :param ref_sess_date: str, date of reference session (where place cells are from)
    :param pcf_list: list of paths to session folders
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :return:
    """
    sess_dates = []
    header = 'Neur_ID_'
    if place_cell_mode:
        mouse = pcf_list[0].params["mouse"]
        for pcf in pcf_list:
            sess_dates.append(pcf.params['session'])
            header = header + f'{pcf.params["session"]}\t'
    else:
        mouse = pcf_list[0].mmap_file.split(sep=os.path.sep)[-3]
        for cnm in pcf_list:
            sess_dates.append(cnm.mmap_file.split(sep=os.path.sep)[-4])
            header = header + f'{cnm.mmap_file.split(sep=os.path.sep)[-4]}\t'
    file_name = f'pc_alignment_{mouse}_{ref_sess_date}.txt'
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path):
        answer = None
        while answer not in ("y", "n", 'yes', 'no'):
            answer = input(f"File [...]{file_path[-40:]} already exists!\nOverwrite? [y/n] ")
            if answer == "yes" or answer == 'y':
                print(f'Saving alignment table to {file_path}...')
                align_fix = np.nan_to_num(align_array, nan=-10).astype('int')
                np.savetxt(file_path, align_fix, delimiter='\t', header=header, fmt='%d')
            elif answer == "no" or answer == 'n':
                print('Saving cancelled.')
                return None
            else:
                print("Please enter yes/y or no/n.")
    else:
        print(f'Saving alignment table to {file_path}...')
        align_fix = np.nan_to_num(align_array, nan=-10).astype('int')
        np.savetxt(file_path, align_fix, delimiter='\t', header=header, fmt='%d')


def load_alignment(file_paths):
    """
    Loads one or more alignment.txt files from the list in a Pandas DataFrame
    :param file_paths: list of alignment.txt files (from save_alignment())
    :return: Pandas Dataframe with all aligned cells from all text files
    """
    cell_list = [np.loadtxt(x) for x in file_paths]
    with open(file_paths[0]) as file:
        dates = file.readline().strip().split('\t')
        dates[0] = dates[0].split('_')[-1]
    dates.insert(0, 'ref_date')
    dtypes = [[x, 'float64'] for x in dates]
    dtypes[0][1] = 'int32'
    dtypes = dict(dtypes)
    ref_dates = [os.path.splitext(os.path.split(x)[1])[0].split('_')[-1] for x in file_paths]
    ref_idx = []
    for i in range(len(cell_list)):
        ref_idx.extend([ref_dates[i]]*len(cell_list[i]))
    ref_idx = np.array(ref_idx)[..., np.newaxis]
    df = pd.DataFrame(np.hstack((ref_idx, np.vstack(cell_list))), columns=dates)
    return df.astype(dtypes)


def align_traces(mousepath, alignment, ignore=None):
    """
    Load PCF objects of the sessions that were aligned in the alignment array
    :param mousepath: str, path to the folder of the corresponding mouse
    :param alignment: pd.DataFrame from load_alignment
    :param ignore: optional list holding sessions that should not be loaded (e.g. first sess after stroke w/o behavior)
    :return: data (dict): PCF objects of all aligned sessions, session dates as keys
    :return: traces (np.array): binned traces of each cell and every session with shape (#neurons, #sessions, #bins)
    :return: unique (pd.DataFrame): unique tracked cells with respective neuron IDs for every session
    """

    data = {}
    if len(check_alignment(alignment)) > 0:
        raise ValueError('Still some mismatched cells. Run check_alignment() again.')
    # Get paths of all sessions
    dates = alignment.columns[1:]
    # Load PCF objects of the aligned sessions
    for sess in dates:
        if (ignore is None) or (ignore is not None and sess not in ignore):
            data[sess] = pipe.load_pcf(os.path.join(mousepath, sess))

    # Filter alignment IDs for unique cells
    cell_ids = alignment[dates]
    unique = cell_ids[~alignment[dates].duplicated()]

    # Put VR-aligned traces of each cell and every session in a 3D array with shape (#neurons, #sessions, #bins)
    traces = np.zeros((len(unique), len(dates), data[dates[0]].params['n_bins']))+np.nan
    for date_idx, date in enumerate(dates):
        found_cells = unique[date].loc[unique[date] != -10]
        # Check that there are no mismatched cells
        if len(found_cells.unique()) != len(found_cells):
            raise ValueError(f'Double Neuron IDs in session {date}, check alignments again.')

        # Go through all cells in that session and put the traces for each cell into the array
        for cell_idx, cell in enumerate(unique[date]):
            if int(cell) != -10:
                traces[cell_idx, date_idx, :] = data[date].bin_avg_activity[int(cell)]

    return data, traces, unique


def piecewise_fov_shift(ref_img, tar_img, n_patch=8):
    """
    Calculates FOV-shift map between a reference and a target image. Images are split in n_patch X n_patch patches, and
    shift is calculated for each patch separately with phase correlation. The resulting shift map is scaled up and
    missing values interpolated to ref_img size to get an estimated shift value for each pixel.
    :param ref_img: np.array, reference image
    :param tar_img: np.array, target image to which FOV shift is calculated. Has to be same dimensions as ref_img
    :param n_patch: int, root number of patches the FOV should be subdivided into for piecewise phase correlation
    :return: two np.arrays containing estimated shifts per pixel (upscaled x_shift_map, upscaled y_shift_map)
    """
    img_dim = ref_img.shape
    patch_size = int(img_dim[0]/n_patch)

    shift_map_x = np.zeros((n_patch, n_patch))
    shift_map_y = np.zeros((n_patch, n_patch))
    for row in range(n_patch):
        for col in range(n_patch):
            curr_ref_patch = ref_img[row*patch_size:row*patch_size+patch_size, col*patch_size:col*patch_size+patch_size]
            curr_tar_patch = tar_img[row*patch_size:row*patch_size+patch_size, col*patch_size:col*patch_size+patch_size]
            patch_shift = phase_cross_correlation(curr_ref_patch, curr_tar_patch, upsample_factor=100, return_error=False)
            shift_map_x[row, col] = patch_shift[0]
            shift_map_y[row, col] = patch_shift[1]
    shift_map_x_big = zoom(shift_map_x, patch_size, order=3)
    shift_map_y_big = zoom(shift_map_y, patch_size, order=3)
    return shift_map_x_big, shift_map_y_big


def plot_single_contour(ax, spatial, template, half_size=50, color='w', verbose=False):
    """
    Draws the contour of one component and focuses the template image around its center.
    :param ax: Axes in which the plot should be drawn
    :param spatial: spatial information of the component, acquired by indexing spatial[session][:,neuron_id]
    :param template: background template image on which to draw the contour
    :param half_size: int, half size in pixels of the final area
    :param color: color of the contour
    :param verbose: bool flag whether the contour threshold should be printed upon plotting
    :returns: CoM of the drawn contour
    """

    def set_lims(com, ax, half_size, max_lims):
        """
        Sets axis limits of Axes around the center of mass of a neuron to create a 50x50 area centered on the CoM.
        :param com: tuple, center of mass X and Y coordinates of a neuron
        :param half_size: int, half size in pixels of the final area
        :param max_lims: tuple, maximum values of X and Y axis, from template.shape
        """

        def fit_lims(coord, half_size, max):
            """
            Takes a coordinate and calculates axis limits of 'half_size' points around it. Caps at template shapes.
            :param coord: int, center of mass coordinate
            :param half_size: int, half size of the final region
            :param max: maximum value of this axis, from template.shape
            :return lim: tuple, axis limits
            """
            lim = [coord - half_size, coord + half_size]
            if lim[0] < 0:
                lim[1] = lim[1] - lim[0]
                lim[0] = 0
            elif lim[1] > max:
                diff = lim[1] - max
                lim[1] = lim[1] - diff
                lim[0] = lim[0] - diff
            return tuple(lim)

        lim_x = fit_lims(com[1], half_size, max_lims[1])  # X is second coordinate
        lim_y = fit_lims(com[0], half_size, max_lims[0])

        ax.set_xlim(lim_x[0], lim_x[1])
        ax.set_ylim(lim_y[1], lim_y[0])  # Y limits have to be swapped because image origin is top-left

    plt.sca(ax)
    plt.cla()       # clear axes from potential previous data
    out = visualization.plot_contours(spatial, template, display_numbers=False, colors=color, verbose=verbose)
    com = (int(np.round(out[0]['CoM'][0])), int(np.round(out[0]['CoM'][1])))
    set_lims(com, ax, half_size, template.shape)
    return com


def shift_com(com, shift, dims):
    """
    Shifts a center-of-mass coordinate point by a certain step size. Caps coordinates at 0 and dims limits
    :param com: iterable, X and Y coordinates of the center of mass
    :param shift: iterable, amount by which to shift com, has to be same length as com
    :param dims: iterable, dimensions of the FOV, has to be same length as com
    :return: shifted com
    """
    # shift the CoM by the given amount (subtract because shifts have been calculated from ref vs tar
    com_shift = [com[0] - shift[0], com[1] - shift[1]]
    # cap CoM at 0 and dims limits
    com_shift = [0 if x < 0 else x for x in com_shift]
    for coord in range(len(com_shift)):
        if com_shift[coord] > dims[coord]:
            com_shift[coord] = dims[coord]
    return com_shift


def get_ref_idx(pcf_sessions, ref_sess_str, place_cell_mode):
    """
    Finds the index of the PCF object with the matching session date.
    :param pcf_sessions: list of PCF objects
    :param ref_sess_str: string of the session date, format 'YYYMMDD'
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :return: ref_sess_idx, integer index of the corresponding PCF object
    """
    # Get index of reference session
    if place_cell_mode:
        ref_sess_idx = None
        for idx, pcf in enumerate(pcf_sessions):
            if pcf.params['session'] == ref_sess_str:
                if ref_sess_idx is None:
                    ref_sess_idx = idx
                else:
                    raise IndexError(f'More than one PCF object has the session date {ref_sess_str}!')
        if ref_sess_idx is None:
            raise IndexError(f'No PCF object has the session date {ref_sess_str}!')
    else:
        ref_sess_idx = None
        for idx, cnm in enumerate(pcf_sessions):
            # Jithins structure has the session date in the third parent folder
            if ref_sess_str in cnm.mmap_file:
                if ref_sess_idx is None:
                    ref_sess_idx = idx
                else:
                    raise IndexError(f'More than one CNM object has the session "{ref_sess_str}"!')
        if ref_sess_idx is None:
            raise IndexError(f'No CNM object has the session date "{ref_sess_str}"!')

    return ref_sess_idx


def prepare_manual_alignment_data(pcf_sessions, ref_sess, place_cell_mode=True, shift_from_mean=True, session_list=None):
    """
    Prepares PCF and CNMF data for manual alignment tool. Initializes alignment array and calculates contours and
    shifts from all cells in all sessions.
    :param pcf_sessions: list of PCF objects to be aligned
    :param ref_sess: str, date of session to be used as a reference (place cells will be taken from this session)
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :param shift_from_mean: bool flag whether the FOV shift is calculated from mean intensity or local correlation image
    :param session_list: list of session paths, only necessary if place_cell_mode=False
    :return: alignment array, all_contours list (list of sessions, each session is list of neuron contours),
            all_shifts (list of sessions, each session is 2 arrays of x_shift and y_shift for every pixel)
    """

    ref_session = get_ref_idx(pcf_sessions, ref_sess, place_cell_mode)
    target_sess = [x for j, x in enumerate(pcf_sessions) if j != ref_session]

    if place_cell_mode:
        # get indices of place cells from first session
        place_cell_idx = [x[0] for x in pcf_sessions[ref_session].place_cells]
    else:
        # if not in place cell mode, give all neurons to align
        place_cell_idx = np.arange(pcf_sessions[ref_session].estimates.F_dff.shape[0])

    # initialize alignment array (#place cells X #sessions)
    alignment = np.full((len(place_cell_idx), len(pcf_sessions)), -1.0)  # +1 due to the reference session being popped

    # get contour data of all cells and FOV shifts between reference and the other sessions with phase correlation
    all_contours = []
    all_shifts = []
    for sess_idx in range(len(pcf_sessions)):
        if sess_idx != ref_session:
            sess = pcf_sessions[sess_idx]
            if place_cell_mode:
                if shift_from_mean:
                    ref_im = tiff.imread(os.path.join(pcf_sessions[ref_session].params['root'],
                                                      'mean_intensity_image.tif'))
                    tar_im = tiff.imread(os.path.join(sess.params['root'], 'mean_intensity_image.tif'))
                else:
                    ref_im = pcf_sessions[ref_session].cnmf.estimates.Cn
                    tar_im = sess.cnmf.estimates.Cn

            # For Jithin, files have to taken from session_list because CNM file paths are changed after creation
            else:
                if shift_from_mean:
                    ref_im = tiff.imread(os.path.join(session_list[ref_session], 'mean_intensity_image.tif'))
                    tar_im = tiff.imread(os.path.join(session_list[sess_idx], 'mean_intensity_image.tif'))
                else:
                    ref_im = pcf_sessions[ref_session].estimates.Cn
                    tar_im = sess.cnmf.estimates.Cn

            curr_shifts_x, curr_shifts_y = piecewise_fov_shift(ref_im, tar_im)
            all_shifts.append((curr_shifts_x, curr_shifts_y))

            plt.figure()
            if place_cell_mode:
                all_contours.append(visualization.plot_contours(sess.cnmf.estimates.A, sess.cnmf.estimates.Cn))
            else:
                all_contours.append(visualization.plot_contours(sess.estimates.A, sess.estimates.Cn))
            plt.close()

    return target_sess, place_cell_idx, alignment, all_contours, all_shifts


def manual_place_cell_alignment(pcf_sessions, target_sessions, cell_idx, alignment, all_contours, all_shifts, dim,
                                ref_sess, place_cell_mode=True, show_neuron_id=False):
    """
    Master function that produces and updates the interactive figure. The reference cell is drawn on the left graph.
    The candidate target cells are plotted on the right side, with the cell closest to the reference cell first.
    IDs are constantly updated in the alignment array and refer to the same neuron in different sessions.
    If alignment is already partly filled (not -1 in all fields), the alignment picks up at the first not-aligned cell.
    The arguments for this function should be the output of the prepare_manual_alignment_data() function!
    :param pcf_sessions: list of all PCF objects that should be aligned
    :param target_sessions: list of PCF objects excluding the reference session
    :param cell_idx: list of cell indices in the reference session that should be aligned
    :param alignment: numpy array that saves the corresponding cell IDs across sessions
    :param all_contours: list of sessions holding the contour coordinates for every neuron
    :param all_shifts: list of sessions holding the FOV shifts in x and y direction
    :param dim: dimensions of the FOV in pixel
    :param ref_sess: str of date of the reference session
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :param show_neuron_id: bool flag whether the IDs of the candidate neurons should be displayed
    :return: alignment: filled np.array with the shape (#cells, #sessions)
    """
    def targ_to_real_idx(idx):
        """
        Adjusts non-reference pcf_sessions idx to real pcf_sessions idx to avoid referring to the reference session.
        :param idx: Index of non-reference session
        :return: adj_idx, actual index of requested session in pcf_sessions
        """
        adj_idx = idx
        if adj_idx >= ref_session:
            adj_idx += 1
        return adj_idx

    def real_to_targ_idx(idx):
        """
        Adjusts real pcf_sessions idx to non-reference pcf_sessions idx to avoid IndexError when indexing target list.
        :param idx: Index of complete pcf_sessions list
        :return: adj_idx, index of requested session in pcf_sessions without reference session
        """
        adj_idx = idx
        if adj_idx > ref_session:
            adj_idx -= 1
        elif adj_idx == ref_session:
            raise IndexError('Cant find index of reference session in target list.')
        return adj_idx

    def draw_reference(ax, cnm, idx):
        """
        Draws reference cell into the provided Axes and returns its contour data.
        :param ax: Axes where to draw the cell, here ref_ax
        :param cnm: CNM object that contains the component data, here pcf_sessions[session]
        :param idx: int, index of the neuron that should be drawn, here taken from place_cell_idx
        :return: 1-length list containing dictionary of contour data
        """
        com = plot_single_contour(ax=ax, spatial=cnm.estimates.A[:, cell_idx[idx]],
                                  template=cnm.estimates.Cn)
        if place_cell_mode:
            plt.setp(ax, url=idx, title=f'Session {ref_sess} (Index {ref_session}), Neuron {cell_idx[idx]} (Index {idx})')
        else:
            plt.setp(ax, url=idx, title=f'{ref_sess} (Index {ref_session}), Neuron {cell_idx[idx]} (Index {idx})')
        return com

    def find_target_cells(reference_com, session_contours, dims, fov_shift, max_dist=25):
        """
        Finds cells in tar_sess that are close to the reference cell
        :param reference_com: tuple, (x,y) of center of mass of reference cell
        :param session_contours: list of contour dicts holding contour information of all target cells
        :param dims: tuple, (x,y) of FOV dimensions
        :param fov_shift: np.array of FOV shifts between reference and target session (from all_shifts)
        :param max_dist: int, maximum radial distance between reference and target cell to be considered 'nearby'
        :return: list of contours of nearby cells sorted by distance
        """
        # Find shift of reference CoM by indexing fov_shift
        # Correct reference center-of-mass by the shift to be better aligned with the FOV of the second session
        reference_com_shift = shift_com(reference_com,
                                        (fov_shift[0][reference_com[0], reference_com[1]],
                                         fov_shift[1][reference_com[0], reference_com[1]]), dims)

        # transform CoM into a Point2D that can handle radial distances
        ref_point = Point2D(reference_com_shift[0], reference_com_shift[1])

        # Go through all cells and select the ones that have a CoM near the reference cell, also remember their distance
        near_contours = []
        distances = []
        for contour in session_contours:
            new_point = Point2D(contour['CoM'][0], contour['CoM'][1])
            rad_dist = (new_point - ref_point).r
            if rad_dist <= max_dist:
                near_contours.append(contour)
                distances.append(rad_dist)

        # return contours sorted by their distance to the reference CoM
        return [x for _, x in sorted(zip(distances, near_contours), key=lambda pair: pair[0])]

    def draw_target_cells(outer_grid, near_contours_sort, idx, ref_cell_idx):
        """
        Draws target cells in the right part of the figure.
        :param outer_grid: GridSpec object where the target cells are to be drawn
        :param near_contours_sort: list, contains contour data of target cells (from find_target_cells)
        :param idx: int, index of the target session the target cells belong to
        :param ref_cell_idx: index of reference cell, needed to label axes
        :return:
        """
        n_cols = 5
        real_idx = targ_to_real_idx(idx)  # exchange the target-session specific idx into a pcf_sessions index
        n_plots = len(near_contours_sort)+1
        # draw possible matching cells in the plots on the right
        if n_plots < 15:  # make a 5x3 grid for up to 14 nearby cells + 1 'No Match' plot
            n_rows = ceil(n_plots/n_cols)    # number of rows needed to plot into 5 columns
        else:
            n_rows = 4      # if there are more than 14 cells, make 4 rows and extend columns as much as necessary
            n_cols = ceil(n_plots/n_rows)
        candidates = outer_grid[1].subgridspec(n_rows, n_cols)  # create grid layout
        counter = 0                                     # counter that keeps track of plotted contour number
        for row in range(n_rows):
            for column in range(n_cols):
                curr_ax = fig.add_subplot(candidates[row, column], picker=True)  # picker enables clicking subplots
                try:
                    # -1 because the neuron_id from visualization.plot_contours starts counting at 1
                    curr_neuron = near_contours_sort[counter]['neuron_id']-1
                    # plot the current candidate
                    if place_cell_mode:
                        curr_cont = plot_single_contour(ax=curr_ax,
                                                        spatial=target_sessions[idx].cnmf.estimates.A[:, curr_neuron],
                                                        template=target_sessions[idx].cnmf.estimates.Cn)
                    else:
                        curr_cont = plot_single_contour(ax=curr_ax,
                                                        spatial=target_sessions[idx].estimates.A[:, curr_neuron],
                                                        template=target_sessions[idx].estimates.Cn)
                    if show_neuron_id:
                        t = curr_ax.text(0.5, 0.5, f'{curr_neuron}', va='center', ha='center', transform=curr_ax.transAxes)
                    # the url property of the Axes is used as a tag to remember which neuron has been clicked
                    # as well as which target session it belonged to
                    plt.setp(curr_ax, url=(ref_cell_idx, real_idx, curr_neuron))
                    curr_ax.tick_params(labelbottom=False, labelleft=False)
                    counter += 1

                # if there are no more candidates to plot, make plot into a "no matches" button and mark it with -10
                except IndexError:
                    dummy = np.ones(dim)
                    dummy[0, 0] = 0
                    curr_ax.imshow(dummy, cmap='gray')
                    t = curr_ax.text(0.5, 0.5, 'No Matches', va='center', ha='center',  transform=curr_ax.transAxes)
                    plt.setp(curr_ax, url=(ref_cell_idx, real_idx, -10))
                    curr_ax.tick_params(labelbottom=False, labelleft=False)
                if row == 0 and column == int(n_cols/2):
                    if place_cell_mode:
                        curr_session = target_sessions[idx].params["session"]
                        curr_ax.set_title(f'Session {curr_session} (Index {real_idx})')
                    else:
                        curr_ax.set_title(f'Session {real_idx+1} (Index {real_idx})')

    def draw_both_sides(ref_idx, targ_sess_idx):
        fig.clear()  # remove potential previous layouts
        outer_grid = grid.GridSpec(1, 2)  # initialize outer structure (two fields horizontally)

        ref = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])  # initialize reference plot
        ref_ax = fig.add_subplot(ref[0])  # draw reference plot

        # first draw the reference cell
        if place_cell_mode:
            ref_com = draw_reference(ref_ax, pcf_sessions[ref_session].cnmf, ref_idx)
        else:
            ref_com = draw_reference(ref_ax, pcf_sessions[ref_session], ref_idx)

        # Find cells in the next session(s) that have their center of mass near the reference cell
        nearby_contours = find_target_cells(reference_com=ref_com,
                                            session_contours=all_contours[targ_sess_idx],
                                            dims=dim,
                                            fov_shift=all_shifts[targ_sess_idx])

        # Draw target cells in the right plots
        draw_target_cells(outer_grid, nearby_contours, targ_sess_idx, ref_idx)

##########################################################################################
################ START OF PLOTTING #######################################################

    ref_session = get_ref_idx(pcf_sessions, ref_sess, place_cell_mode)

    # see if the alignment array has already been (partly) filled to skip processed cells
    if len(np.unique(alignment)) != 1:
        untagged_cells = np.where(alignment == -1)
        if len(untagged_cells) == 0:
            return print("All cells aligned!")
        else:
            start_ref = untagged_cells[0][0]   # row of first -1 shows with which reference cell to start
            start_real = untagged_cells[1][0]   # col of first -1 shows with which target session to start
        if start_real == ref_session:
            start_tar = 0
        else:
            start_tar = real_to_targ_idx(start_real)
    else:   # otherwise start with the first cell
        start_ref = 0
        start_tar = 0

    # build figure
    fig = plt.figure(figsize=(18, 8))  # draw figure

    # First drawing
    draw_both_sides(start_ref, start_tar)  # Draw the first reference (place) cell and the target cells

    # Define what happens when a plot was clicked: update alignment accordingly and draw the next set of cells
    def onpick(event):
        this_plot = event.artist                 # save artist (axis) where the pick was triggered
        id_s = plt.getp(this_plot, 'url')        # get the IDs of the neuron that was clicked and of the current session
        ref_id = id_s[0]
        real_sess_id = id_s[1]
        targ_sess_id = real_to_targ_idx(id_s[1])
        neuron_id = id_s[2]
        print(f'Reference cell {ref_id}, clicked neuron {id_s}.')

        # update the alignment array with the clicked assignment
        alignment[ref_id, ref_session] = cell_idx[ref_id]   # which ID did this place cell have?
        if neuron_id == -1:  # '-1' means "no match" has been clicked, fill spot with nan
            alignment[ref_id, real_sess_id] = np.nan
        else:
            # assign the clicked neuron_id to the 'ref_id' reference cell in the 'sess_id' session
            alignment[ref_id, real_sess_id] = neuron_id

        # if this is the last session, draw a new reference cell
        if targ_sess_id+1 == len(pcf_sessions)-1:
            draw_both_sides(ref_id+1, 0)
        # else, only draw the target cells of the next session
        else:
            draw_both_sides(ref_id, targ_sess_id+1)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.show()

    return alignment


def show_whole_fov(reference_session, target_session, ref_cell_id, place_cell_mode=True, move_together=True):
    """
    Shows the whole FOV of the reference session with the contour of the reference cell, as well as the FOV of the
    target session with the contours and IDs of all cells.
    CAREFUL: Caiman starts labelling cells with 1, meaning that you have to subtract 1 from the ID plotted on the left
    graph if you want to use it for the alignment.
    :param reference_session: PCF object of the reference session (from pcf_objects)
    :param target_session: PCF object of the target session (from pcf_objects)
    :param ref_cell_id: ID number of the reference cell (from the title in the main figure)
    :param place_cell_mode: bool flag whether to load PCF objects or CNM objects (for Jithin)
    :param move_together: bool flag whether the reference and target FOVs should move together or independently
    :return:
    """
    # Initialize figure
    if move_together:
        fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharex='all', sharey='all')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    if place_cell_mode:
        reference = reference_session.cnmf
        target = target_session.cnmf
        ref_curr_session = reference_session.params['session']
        tar_curr_session = target_session.params['session']
    else:
        reference = reference_session
        target = target_session
        ref_curr_session = reference.mmap_file.split(sep=os.path.sep)[-4]
        tar_curr_session = target.mmap_file.split(sep=os.path.sep)[-4]

    # Plot reference cell on the left side
    plt.sca(ax[0])
    out = visualization.plot_contours(reference.estimates.A[:, ref_cell_id], reference.estimates.Cn,
                                      display_numbers=False, colors='r', verbose=False)
    ax[0].set_title('Session {}, reference neuron {}'.format(ref_curr_session, ref_cell_id))

    # Plot all other cells on the right side
    plt.sca(ax[1])
    out = visualization.plot_contours(target.estimates.A, target.estimates.Cn,
                                      display_numbers=True, colors='r', verbose=False)
    ax[1].set_title('Session {}, all other neurons'.format(tar_curr_session))


def check_alignment(alignments, save=None):
    """
    Checks alignment arrays for cells that received different IDs in the same session. The results are stored in a list
    of DataFrames, each DF holding the IDs of one cell across sessions. The results can also be saved as a text file.
    :param alignments: pandas DataFrame with all alignment.txt data
    :return:
    """
    dates = alignments.columns[1:]
    misalign = []
    breakpoint = False
    for sess in dates:
        # Get all unique cell IDs (and remove un-assigned -10s)
        unique = np.unique(alignments[sess])
        unique = np.delete(unique, np.where(unique == -10))
        unique = np.delete(unique, np.where(unique == -1))
        for cell in unique:
            # Get the cross-session IDs of the same cells
            rows = alignments.loc[alignments[sess] == cell]
            exists = [rows.equals(x) for x in misalign]
            has_mismatch = [True if len(rows[date].unique())>1 else False for date in dates]
            # if it has not been checked before and the cell was tracked more than 1 time, save it
            if not any(exists) and len(rows) > 1 and any(has_mismatch):
                misalign.append(rows)

    if save is not None:
        with open(save, 'w') as f:
            misalign[0].to_csv(f)
            f.write('\n')
        for i in range(1, len(misalign)):
            with open(save, 'a') as f:
                misalign[i].to_csv(f, header=True)
                f.write('\n')

    return misalign

    # Old confusing method
    # dates = alignments.columns[1:]
    # ref_dates = alignments['ref_date'].unique().astype(str)
    # sess_dic = {k: {} for k in dates}
    # change_dic = {k: deepcopy(sess_dic) for k in ref_dates}
    # test = []
    # for sess in dates:
    #     # Get all unique cell IDs (and remove un-assigned -10s)
    #     unique = np.unique(alignments[sess])
    #     unique = np.delete(unique, np.where(unique == -10))
    #     for cell in unique:
    #         # Get all IDs of the same cells
    #         rows = alignments.loc[alignments[sess] == cell]
    #         wrong = []
    #         exists = [rows.equals(x) for x in test]
    #         if not any(exists) and len(rows) > 1:
    #             test.append(rows)
    #         # Go through sessions and look for sessions where the ID was not the same
    #         for session in dates:
    #             if len(np.unique(rows[session])) > 1:
    #                 wrong.append((session, rows[session]))
    #         # If there were mismatched IDs, print them out together with their session dates
    #         if len(wrong) > 0:
    #             # print(f'\nThe cell with ID {cell} on session {sess} has different IDs in other sessions:')
    #             for i in wrong:
    #                 # print(f'\tFor session {i[0]}:')
    #                 new_dic = {}
    #                 for j in range(len(i[1])):
    #                     new_dic[rows["ref_date"].iloc[j]] = i[1].iloc[j]
    #                 for key in new_dic:
    #                     ref_cell_idx = str(int(rows.loc[rows['ref_date']==key, sess].unique()))
    #                     if ref_cell_idx not in change_dic[str(key)][i[0]]:
    #                         change_dic[str(key)][i[0]][ref_cell_idx] = new_dic
    #                         # print(f'\t\tIn reference session {rows["ref_date"].iloc[j]}:', i[1].iloc[j])


def reset_misaligned_cells(misalignment, file_list):
    """
    OLD FUNCTION, NOT RECOMMENDED! USE "MISALIGN" LIST FROM CHECK_DOUBLE_CELLS!
    Takes misalignment list from check_alignment() and resets the IDs in the .txt files to -1 so it can be checked
    again.
    :param misalignment:
    :param file_list:
    :return:
    """
    # load the alignment files and save a copy
    align = {}
    for file in file_list:
        date = os.path.splitext(file)[0].split('_')[-1]
        with open(file) as f:
            header = f.readline()
        align[date] = np.loadtxt(file)
        # Get next filename and save the array
        new_file = fm.get_next_filename(file)
        np.savetxt(new_file, align[date], delimiter='\t', header=header, fmt='%d')

    new_align = deepcopy(align)

    for cell in misalignment:
        # Get indices of the columns with the mismatched IDs for numpy indexing
        bad_cols = [cell.columns.get_loc(session)-1 for session in cell.columns[1:] if len(cell[session].unique()) > 1]
        for date in cell['ref_date']:
            arr = align[str(date)]
            # Find row of the cell in each np alignment array
            row = np.where((arr == np.array(cell.loc[cell['ref_date'] == int(date), cell.columns[1:]]))
                           .all(axis=1))[0][0]
            # Change the value of the cell at the session to -1
            new_align[str(date)][row, bad_cols] = -1

    # save new alignment files
    for file in file_list:
        date = os.path.splitext(file)[0].split('_')[-1]
        with open(file) as f:
            header = f.readline()
        np.savetxt(file, new_align[date], delimiter='\t', header=header, fmt='%d')



def plot_aligned_cells(cell_list, pcf_objects_list, ref_dates, color=False, colbar=False, sort=True, show_neuron_id=False):
    """
    :param cell_list: list of assignments, one array (element) per session. Can be e.g. alignments
    :param pcf_objects_list:
    :param ref_dates: list of index showing which session the place cells were from. Same length as cell_list.
    :param color:
    :param colbar:
    :param sort:
    :return:
    """

    ref_list = []
    for k in range(len(ref_dates)):
        for n_cells in range(cell_list[k].shape[0]):
            ref_list.append(ref_dates[k])

    all_cell_array = np.vstack(cell_list)

    nrows = len(ref_list)
    ncols = len(pcf_objects_list)


    # order place cells (first cell in each row) by the start of its place cell
    # sort neurons after different criteria
    bins = []
    for m in range(len(cell_list)):
        this_sess_bins = []
        for p in range(cell_list[m].shape[0]):
            curr_pc_id = np.where(np.array([x[0] for x in pcf_objects_list[ref_dates[m]].place_cells]) ==
                                  int(cell_list[m][p, ref_dates[m]]))[0][0]
            this_sess_bins.append((p, pcf_objects_list[ref_dates[m]].place_cells[curr_pc_id][1][0][0]))  # get the first index of the first place field
        bins.append(this_sess_bins)

    all_bins_sort = []
    for bin_list in bins:
        bins_sorted = sorted(bin_list, key=lambda tup: tup[1])
        if sort:
            bins_sort = [x[0] for x in bins_sorted]
        else:
            bins_sort = [x[0] for x in bin_list]
        all_bins_sort.append(bins_sort)

    combined_bins_sort = []
    n_cells = 0
    for n in range(len(all_bins_sort)):
        if n != 0:
            combined_bins_sort.append([x + (len(all_bins_sort[n-1]) + n_cells) for x in all_bins_sort[n]])
            n_cells = len(all_bins_sort[n-1]) + n_cells
        else:
            combined_bins_sort.append(all_bins_sort[n])
    comb_bins_sort = [item for sublist in combined_bins_sort for item in sublist]

    # get activity data of all cells
    all_cells = []
    for nrow in range(nrows):
        this_cell = np.ones((ncols, 80))
        for col in range(ncols):
            curr_neur_id = int(all_cell_array[comb_bins_sort[nrow], col])
            #if curr_neur_id == -10 or col == 3 and curr_neur_id == 346: # if the ID is -10 (not recognized), set values to 0 to be filtered out later
            if curr_neur_id == -10 or col == 3 and curr_neur_id == 646 or curr_neur_id == 565:  # if the ID is -10 (not recognized), set values to 0 to be filtered out later
                this_cell[col] = np.zeros((1, 80))
            else:
                this_cell[col] = pcf_objects_list[col].bin_avg_activity[curr_neur_id]
        all_cells.append(this_cell)

    # set global y-axis limits over all cells
    if color and colbar:
        glob_scale = True
        all_data = np.vstack(all_cells)
        cell_max = all_data.max()
        cell_min = all_data.min()
    else:
        glob_scale = False

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Place cells in session {pcf_objects_list[ref_dates[0]].params["session"]} tracked over time', fontsize=22)
    for nrow in range(nrows):
        if not glob_scale:
            cell_max = np.max(all_cells[nrow])
            cell_min = np.min(all_cells[nrow])
        for col in range(ncols):
            if show_neuron_id:
                ax[nrow, col].text(0.5, 0.5, int(all_cell_array[comb_bins_sort[nrow], col]),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   fontsize=12, color='red',
                                   transform=ax[nrow, col].transAxes)

            if color:
                curr_data = all_cells[nrow][col, np.newaxis]
            else:
                curr_data = all_cells[nrow][col]
            if nrow == 0:
                ax[nrow, col].set_title(pcf_objects_list[col].params['session'])
            elif nrow == nrows-1:
                # set x ticks to VR position, not bin number
                ax[nrow, col].set_xlim(0, len(all_cells[nrow][col]))
                ax[nrow, col].get_xaxis().set_ticks([0, 40, 80])
                x_locs, labels = plt.xticks()
                plt.sca(ax[nrow, col])
                plt.xticks(x_locs, (x_locs * pcf_objects_list[col].params['bin_length']).astype(int), fontsize=12)

            if color:
                if nrow != nrows - 1:
                    if nrow != int(nrows / 2):
                        ax[nrow, col].axis('off')
                    else:
                        ax[nrow, col].spines['top'].set_visible(False)
                        ax[nrow, col].spines['right'].set_visible(False)
                        ax[nrow, col].spines['left'].set_visible(False)
                        ax[nrow, col].spines['bottom'].set_visible(False)
                        ax[nrow, col].get_yaxis().set_ticks([])
                        ax[nrow, col].get_xaxis().set_ticks([])
                else:
                    ax[nrow, col].spines['top'].set_visible(False)
                    ax[nrow, col].spines['right'].set_visible(False)
                    ax[nrow, col].spines['left'].set_visible(False)
                    ax[nrow, col].get_yaxis().set_ticks([])
            else:
                # if col == 0:
                #     ax[nrow, col].spines['top'].set_visible(False)
                #     ax[nrow, col].spines['right'].set_visible(False)
                # else:
                ax[nrow, col].spines['top'].set_visible(False)
                ax[nrow, col].spines['right'].set_visible(False)
                ax[nrow, col].spines['left'].set_visible(False)
                ax[nrow, col].get_yaxis().set_ticks([])

            if not np.any(curr_data):
                if nrow != nrows-1:
                    ax[nrow, col].axis('off')
            else:
                if color:
                    img = ax[nrow, col].pcolormesh(curr_data, vmax=cell_max, vmin=cell_min, cmap='jet')
                else:
                    ax[nrow, col].plot(curr_data)
                    ax[nrow, col].set_ylim(bottom=cell_min, top=cell_max)
                    pc_idx = np.where(np.array([x[0] for x in pcf_objects_list[col].place_cells]) ==
                                      int(all_cell_array[comb_bins_sort[nrow], col]))[0]
                    if len(pc_idx) == 0:
                        pc_idx = None
                    else:
                        pc_idx = pc_idx[0]
                    # pc_idx = find_pc_id(pcf_objects_list[col], int(all_cell_array[comb_bins_sort[nrow], col]))
                    if pc_idx is not None:
                        curr_place_fields = pcf_objects_list[col].place_cells[pc_idx][1]
                        for field in curr_place_fields:
                            ax[nrow, col].axvspan(field.min(), field.max(), facecolor='r', alpha=0.2)
                            #ax[nrow, col].plot(field, all_cells[nrow][col][field], color='red')

    # draw color bar
    if color and colbar:
        fraction = 0.10  # fraction of original axes to use for colorbar
        half_size = int(np.round(ax.shape[0] / 3))  # plot colorbar in 1/3 of the figure
        cbar = fig.colorbar(img, ax=ax[half_size:, -1], fraction=fraction, label=r'$\Delta$F/F')  # draw color bar
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.label.set_size(15)

        # align all plots
        fig.subplots_adjust(left=0.1, right=1 - (fraction + 0.05), top=0.9, bottom=0.1)

    ax[nrows-1, int(ncols/2)].set_xlabel('Position in VR [cm]', fontsize=15)
    if color:
        ax[int(nrows / 2), 0].set_ylabel('# Neuron', fontsize=15)
    else:
        ax[int(nrows / 2), 0].set_ylabel('# Neuron\n$\Delta$F/F', fontsize=15)


def align_data():
    return None

#%%
if __name__ == '__main__':
    dir_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191122a\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191125\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191126b\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191127a\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191205\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191206\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191207\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191208\N2',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191219\N2']
    ref_session = 5

    spatial, templates, dim, pcf_objects = load_multisession_data(dir_list)


    target_session_list, place_cell_indices, alignment_array, all_contours_list, all_shifts_list = prepare_manual_alignment_data(pcf_objects, ref_session)

    alignment_array = manual_place_cell_alignment(pcf_sessions=pcf_objects,
                                                  target_sessions=target_session_list,
                                                  cell_idx=place_cell_indices,
                                                  alignment=alignment_array,
                                                  all_contours=all_contours_list,
                                                  all_shifts=all_shifts_list,
                                                  ref_session=ref_session,
                                                  show_neuron_id=True)

    save_alignment(alignment_array, ref_session, dir_list)

    #%% multi PCF anal
    for pcf in pcf_objects:
        #pcf.plot_all_place_cells(show_neuron_id=True)
        pcf_objects[0].plot_pc_location(color='r', display_numbers=True)



    #%% test CaImAn alignment tool

    spatial_union, assignments, matchings, spatial, templates, pcf_objects = align_multisession(dir_list)

    # load manual assignment
    manual = np.loadtxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191125.txt')

    # get the same cells from CaImAn tool
    caiman_results = np.zeros(manual.shape)
    for row in range(manual.shape[0]):
        curr_row = manual[row]
        caiman_results[row] = assignments[np.where(assignments[:, 0] == curr_row[0])]
    caiman_results[np.isnan(caiman_results)] = -10  # make nans to -10 to make it comparable with manual
    # compare how many hits CaImAn has
    sum_equal = np.equal(manual, caiman_results)
    performance = np.sum(sum_equal[:, 2:4])/sum_equal[:, 2:4].size



    # only take neurons that were found in all sessions
    assignments_filtered = np.array(assignments[np.sum(~np.isnan(assignments), axis=1) >= assignments.shape[1]], dtype=int)

    # Use filtered indices to select the corresponding spatial components
    spatial_filtered = []
    for i in range(len(spatial)):
        spatial_filtered.append(spatial[i][:, assignments_filtered[:, i]])

    # Plot spatial components of the selected components on the template of the last session
    contours = plot_all_aligned(spatial_filtered, templates, data_only=True)


    test1 = out[1]['coordinates']
    test1[~np.isnan(test1).any(axis=1)]

    #%% Plotting of traces of aligned cells
    sess_list = ['day -12', 'day -9', 'day -8', 'day -7', 'day 0', 'day 1', 'day 2', 'day 3', 'day 4', 'day 15']


    def find_pc_id(pcf_obj, neur_idx):
        for j in range(len(pcf_obj.place_cells)):
            if pcf_obj.place_cells[j][0] == neur_idx:
                return j


    idx_file_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191125_full.txt',
                     ]
    # load data from txt files
    idx_file_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191125.txt',
                     r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191126b.txt',
                     r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191127a.txt',
                     r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191204.txt',
                     r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191205.txt']

    alignments_all = []
    for file in idx_file_list:
        alignments_all.append(np.loadtxt(file, delimiter='\t'))
    pc_idx_list = []
    for obj in pcf_objects:
        pc_idx_list.append([x[0] for x in obj.place_cells])

    # get a list of all aligned cells
    all_aligned_cells = []
    for i in range(alignments_all[0].shape[0]):
        all_aligned_cells.append(alignments_all[0][i])


    # skip cells that didnt get recognized in all sessions
    alignments = []
    for session in alignments_all:
        alignments.append(session[np.all(session != -10, axis=1)])

    flat_list = np.vstack(alignments)

    # get cells that are place cells in all three sessions
    all_sess_pc = []
    for session in alignments:
        for row in range(session.shape[0]):
            all_pc = True
            for column in range(len(pc_idx_list)):
                if session[row, column] not in pc_idx_list[column]:
                    all_pc = False
            if all_pc:
                all_sess_pc.append(session[row])

    # get cells that are place cells in more than one session
    two_sess_pc = []
    one_sess_pc = []
    for sess_idx in range(len(alignments)):
        for row in range(alignments[sess_idx].shape[0]):
            double_pc = False
            for column in range(len(pc_idx_list)):
                if alignments[sess_idx][row, column] in pc_idx_list[column] and column != sess_idx:
                    double_pc = True
            if double_pc and not any((alignments[sess_idx][row] == x).all() for x in two_sess_pc):
                two_sess_pc.append(alignments[sess_idx][row])
            else:
                one_sess_pc.append(alignments[sess_idx][row])

    # plot contours of a cell that was found in all sessions
    for cell in range(alignments[0].shape[0]):
        fig, ax = plt.subplots(1, alignments[0].shape[1], figsize=(17, 5))
        for sess in range(alignments[0].shape[1]):
            curr_spat = pcf_objects[sess].cnmf.estimates.A[:, alignments[0][cell, sess]]
            plot_single_contour(ax[sess], curr_spat, pcf_objects[sess].cnmf.estimates.Cn, half_size=25, color='r')
            ax[sess].axis('off')
            ax[sess].set_title(sess_list[sess])
        plt.tight_layout()
        path = os.path.join(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Progress Reports\22.01.2020\cells across all sessions', f'cell_{cell}.png')
        plt.savefig(path)
        plt.close()




    plot_aligned_cells(alignments, pcf_objects, [1, 4, 5], color=True, colbar=False)


    #%% Plot example graphs for place cell 13, 20191125
    trial_borders = [0, 1783, 3946, 5639, 6936, 9454, 13678]
    self = pcf_objects[0]
    # raw trace
    plt.figure()
    plt.plot(self.cnmf.estimates.F_dff[13])
    plt.vlines(trial_borders, -0.1, 1.4, 'r')
    plt.ylabel(r'$\Delta$F/F', fontsize=15)
    plt.xlabel(r'frames', fontsize=15)
    plt.title('Calcium trace of neuron 13, 25.11.2019')
    # with noise level threshold
    noise = self.params['sigma'][13]
    last_border = 0
    for sigma in noise:
        plt.hlines(4*sigma, trial_borders[last_border], trial_borders[last_border+1], 'g')
        last_border += 1
    # significant transients
    sig_trans = np.concatenate(self.session_trans[13])
    plt.figure()
    plt.plot(sig_trans)
    plt.vlines(trial_borders, -0.1, 1.4, 'r')
    plt.ylabel(r'$\Delta$F/F', fontsize=15)
    plt.xlabel(r'frames', fontsize=15)
    plt.title('Calcium trace of neuron 13, 25.11.2019')
    # bin_activity separate trials
    fig, ax = plt.subplots(6,1, sharey=True)
    fig.suptitle('Calcium trace binned to VR position')
    for i in range(6):
        ax[i].plot(self.bin_activity[13][i])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    ax[2].set_ylabel(r'$\Delta$F/F', fontsize=15)
    ax[i].set_xlabel('bins', fontsize=15)
    # bin_avg_activity
    plt.figure()
    ax = plt.gca()
    ax.plot(self.bin_avg_activity[13])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Delta$F/F', fontsize=15)
    ax.set_xlabel('bins', fontsize=15)
    #smoothed activity
    plt.figure()
    ax = plt.gca()
    smooth_trace = self.smooth_trace(self.bin_avg_activity[13])
    ax.plot(smooth_trace)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Delta$F/F', fontsize=15)
    ax.set_xlabel('bins', fontsize=15)

    # PLACE CELL ANALYSIS
    # pre screening for place fields
    plt.figure()
    ax = plt.gca()
    smooth_trace = self.smooth_trace(self.bin_avg_activity[13])
    ax.plot(smooth_trace)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Delta$F/F', fontsize=15)
    ax.set_xlabel('bins', fontsize=15)
    # show baseline activity (mean of 25% least active bins)
    f_max = max(smooth_trace)
    f_base = np.mean(np.sort(smooth_trace)[:int(smooth_trace.size * self.params['bin_base'])])
    ax.hlines(f_base, 0, 79, 'r')
    # show threshold activity (25% difference between f_max and f_base)
    f_thresh = ((f_max - f_base) * self.params['place_thresh']) + f_base
    ax.hlines(f_thresh, 0, 79, 'g')
    # shade place field
    pot_place_blocks = self.pre_screen_place_fields(smooth_trace)
    ax.axvspan(min(pot_place_blocks[0]), max(pot_place_blocks[0]), alpha=0.3, color='red')
    # criteria
    # infield 7 times higher than outfield
    pot_place_idx = np.in1d(range(smooth_trace.shape[0]), pot_place_blocks[0])  # get an idx mask for the potential place field
    all_place_idx = np.in1d(range(smooth_trace.shape[0]), np.concatenate(pot_place_blocks))   # get an idx mask for all place fields
    mean_infield = np.mean(smooth_trace[pot_place_idx])
    mean_outfield = self.params['fluo_infield'] * np.mean(smooth_trace[~all_place_idx])
    ax.hlines(np.mean(smooth_trace[~all_place_idx]), 0, 79, 'r')
    ax.hlines(mean_outfield, 0, 79, 'g')
    ax.hlines(mean_infield, 0, 79, 'b')
    # 25% of time is significant transient
    place_field = pot_place_blocks[0]
    plt.figure()
    ax = plt.gca()
    plt.plot(sig_trans)
    plt.vlines(trial_borders, -0.1, 1.4, 'r')
    plt.ylabel(r'$\Delta$F/F', fontsize=15)
    plt.xlabel(r'frames', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    place_frames_trace = []  # stores the trace of all trials when the mouse was in a place field as one data row
    for trial in range(self.params['bin_frame_count'].shape[1]):
        # get the start and end frame for the current place field from the bin_frame_count array that stores how
        # many frames were pooled for each bin
        curr_place_frames = (np.sum(self.params['bin_frame_count'][:place_field[0], trial])+trial_borders[trial],
                             np.sum(self.params['bin_frame_count'][:place_field[-1] + 1, trial])+trial_borders[trial])
        ax.axvspan(curr_place_frames[0], curr_place_frames[1], alpha=0.3, color='g')

    # plot final trace with color code
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12,4))
    data = pcf.bin_avg_activity[13, np.newaxis]
    ax[0].plot(smooth_trace)
    img = ax[1].pcolormesh(data, cmap='jet')
    ax[1].set_yticks([])
    ax[0].set_ylabel('$\Delta$F/F', fontsize=15)
    ax[0].set_xlabel('bins', fontsize=15)
    ax[1].set_xlabel('bins', fontsize=15)
    ax[0].plot(field, smooth_trace[field], color='red')
    # set x axis labels as VR position
    ax[0].set_xlim(0, smooth_trace.shape[0])
    x_locs, labels = plt.xticks()
    plt.xticks(x_locs, (x_locs * pcf.params['bin_length']).astype(int))
    plt.sca(ax[0])
    plt.xticks(x_locs, (x_locs * pcf.params['bin_length']).astype(int))
    ax[0].set_xlabel('VR position [cm]')
    ax[1].set_xlabel('VR position [cm]')
    # plot color bar
    fraction = 0.10  # fraction of original axes to use for colorbar
    half_size = int(np.round(ax.shape[0] / 2))  # plot colorbar in half of the figure
    cbar = fig.colorbar(img, ax=ax[1], fraction=fraction, label=r'$\Delta$F/F')  # draw color bar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_size(15)

    #%% save data for Björn
    root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\alignment_data_björn'

    for file in range(len(pcf_objects)):
        curr_pcf = pcf_objects[file]
        curr_sess = curr_pcf.params['session']
        curr_data = {'dff_trace': curr_pcf.cnmf.estimates.F_dff,
                     'spatial_masks': curr_pcf.cnmf.estimates.A,
                     'contours': curr_pcf.cnmf.estimates.coordinates}
        trace_path = os.path.join(root, f'data_{curr_sess}.pickle')
        with open(trace_path, 'wb') as f:
            pickle.dump(curr_data, f)
        print(f'Done! {file+1}/{len(pcf_objects)}')
