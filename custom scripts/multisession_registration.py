import place_cell_pipeline as pipe
from glob import glob
from caiman import load_memmap
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from skimage.feature import register_translation
from math import ceil
from point2d import Point2D
from scipy.ndimage import zoom
import os

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


def align_multisession(dir_list):
    spatial, templates, dims, pcf_objects = load_multisession_data(dir_list)
    print('Import complete. Looking for aligned components...')
    spatial_union, assignments, matchings = register_multisession(spatial, dims=dims, templates=templates)

    return spatial_union, assignments, matchings, spatial, templates, pcf_objects


def load_multisession_data(dir_list):
    """
    Loads CNMF objects from multiple PCFs for multisession registration.
    :param dir_list: list of paths to session folders that should be aligned
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
        curr_pcf = pipe.load_pcf(folder)    # load pcf object that includes the cnmf object
        spatial_list.append(curr_pcf.cnmf.estimates.A)
        try:
            templates_list.append(curr_pcf.cnmf.estimates.Cn)
        except AttributeError:
            print(f'No local correlation image found in {folder}...')
            movie_file = glob(folder+'memmap__*.mmap')
            if len(movie_file) == 1:
                print(f'\tFound mmap file, local correlation image is being calculated.')
                Yr, dims, T = load_memmap(movie_file[0])
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                curr_pcf.cnmf.estimates.Cn = pipe.get_local_correlation(images)
                curr_pcf.save(overwrite=True)
                templates_list.append(curr_pcf.cnmf.estimates.Cn)
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


def save_alignment(align_array, ref_sess_id, session_list):
    """
    Saves manual alignment array to a txt file with session dates as headers for each column. NaNs (no match found)
    are replaced by -10 to maintain visibility.
    :param align_array: data array, shape (n_placecells, n_sessions), from manual_place_cell_alignment()
    :param ref_sess_id: int, index of reference session (where place cells are from) in session_list
    :param session_list: list of paths to session folders
    :return:
    """
    sess_dates = []
    header = 'Neur_ID_'
    for session in session_list:
        sess_dates.append(session.split(os.path.sep)[-2])
        header = header + f'{session.split(os.path.sep)[-2]}\t'
    file_name = f'pc_alignment_{sess_dates[ref_sess_id]}.txt'
    file_path = os.path.join(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis', file_name)
    print(f'Saving alignment table to {file_path}...')
    align_fix = np.nan_to_num(align_array, nan=-10).astype('int')
    np.savetxt(file_path, align_fix, delimiter='\t', header=header, fmt='%d')

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
            patch_shift = register_translation(curr_ref_patch, curr_tar_patch, upsample_factor=100, return_error=False)
            shift_map_x[row, col] = patch_shift[0]
            shift_map_y[row, col] = patch_shift[1]
    shift_map_x_big = zoom(shift_map_x, patch_size, order=3)
    shift_map_y_big = zoom(shift_map_y, patch_size, order=3)
    return shift_map_x_big, shift_map_y_big


def draw_single_contour(ax, spatial, template, half_size=50):
    """
    Draws the contour of one component and focuses the template image around its center.
    :param ax: Axes in which the plot should be drawn
    :param spatial: spatial information of the component, acquired by indexing spatial[session][:,neuron_id]
    :param template: background template image on which to draw the contour
    :param half_size: int, half size in pixels of the final area
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
    out = visualization.plot_contours(spatial, template, display_numbers=False)
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


def prepare_manual_alignment_data(pcf_sessions, ref_session):
    """
    Prepares PCF and CNMF data for manual alignment tool. Initializes alignment array and calculates contours and
    shifts from all cells in all sessions.
    :param pcf_sessions: list of PCF objects to be aligned
    :param ref_session: int, index of session to be used as a reference (place cells will be taken from this session)
    :return: alignment array, all_contours list (list of sessions, each session is list of neuron contours),
            all_shifts (list of sessions, each session is 2 arrays of x_shift and y_shift for every pixel)
    """
    target_sess = [x for j, x in enumerate(pcf_sessions) if j != ref_session]
    # get indices of place cells from first session
    place_cell_idx = [x[0] for x in pcf_sessions[ref_session].place_cells]

    # initialize alignment array (#place cells X #sessions)
    alignment = np.full((len(place_cell_idx), len(pcf_sessions)), -1.0)  # +1 due to the reference session being popped

    # get contour data of all cells and FOV shifts between reference and the other sessions with phase correlation
    all_contours = []
    all_shifts = []
    for sess_idx in range(len(pcf_sessions)):
        if sess_idx != ref_session:
            sess = pcf_sessions[sess_idx]
            curr_shifts_x, curr_shifts_y = piecewise_fov_shift(pcf_sessions[ref_session].cnmf.estimates.Cn,
                                                               sess.cnmf.estimates.Cn)
            all_shifts.append((curr_shifts_x, curr_shifts_y))
            plt.figure()
            all_contours.append(visualization.plot_contours(sess.cnmf.estimates.A, sess.cnmf.estimates.Cn))
            plt.close()
    return target_sess, place_cell_idx, alignment, all_contours, all_shifts


def manual_place_cell_alignment(pcf_sessions, target_sessions, place_cell_idx, alignment, all_contours, all_shifts,
                                ref_session):
    """
    IDs refer to the same neuron in different sessions
    :param pcf_sessions:
    :param target_sessions:
    :param place_cell_idx:
    :param alignment:
    :param all_contours:
    :param all_shifts:
    :param ref_session:
    :return:
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

    def draw_reference(ax, pcf, idx):
        """
        Draws reference cell into the provided Axes and returns its contour data.
        :param ax: Axes where to draw the cell, here ref_ax
        :param pcf: PCF object that contains the component data, here pcf_sessions[session]
        :param idx: int, index of the neuron that should be drawn, here taken from place_cell_idx
        :return: 1-length list containing dictionary of contour data
        """
        com = draw_single_contour(ax=ax, spatial=pcf.cnmf.estimates.A[:, place_cell_idx[idx]],
                                  template=pcf.cnmf.estimates.Cn)
        plt.setp(ax, url=idx, title=f'Session {ref_session + 1}, Neuron {place_cell_idx[idx]}')
        #ax.tick_params(labelbottom=False, labelleft=False)
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
                    curr_cont = draw_single_contour(ax=curr_ax,
                                                    spatial=target_sessions[idx].cnmf.estimates.A[:, curr_neuron],
                                                    template=target_sessions[idx].cnmf.estimates.Cn)
                    t = curr_ax.text(0.5, 0.5, f'{curr_neuron}', va='center', ha='center', transform=curr_ax.transAxes)
                    # the url property of the Axes is used as a tag to remember which neuron has been clicked
                    # as well as which target session it belonged to
                    plt.setp(curr_ax, url=(ref_cell_idx, real_idx, curr_neuron))
                    curr_ax.tick_params(labelbottom=False, labelleft=False)
                    counter += 1

                # if there are no more candidates to plot, make plot into a "no matches" button and mark it with -1
                except IndexError:
                    dummy = np.ones(dim)
                    dummy[0, 0] = 0
                    curr_ax.imshow(dummy, cmap='gray')
                    t = curr_ax.text(0.5, 0.5, 'No Matches', va='center', ha='center',  transform=curr_ax.transAxes)
                    plt.setp(curr_ax, url=(ref_cell_idx, real_idx, -1))
                    curr_ax.tick_params(labelbottom=False, labelleft=False)
                if row == 0 and column == int(n_cols/2):
                    curr_ax.set_title(f'Session {real_idx + 1}')

    def draw_both_sides(ref_idx, targ_sess_idx):
        fig.clear()  # remove potential previous layouts
        outer_grid = grid.GridSpec(1, 2)  # initialize outer structure (two fields horizontally)

        ref = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])  # initialize reference plot
        ref_ax = fig.add_subplot(ref[0])  # draw reference plot

        # first draw the reference cell
        ref_com = draw_reference(ref_ax, pcf_sessions[ref_session], ref_idx)

        # Find cells in the next session(s) that have their center of mass near the reference cell
        nearby_contours = find_target_cells(reference_com=ref_com,
                                            session_contours=all_contours[targ_sess_idx],
                                            dims=pcf_sessions[ref_session].cnmf.estimates.Cn.shape,
                                            fov_shift=all_shifts[targ_sess_idx])

        # Draw target cells in the right plots
        draw_target_cells(outer_grid, nearby_contours, targ_sess_idx, ref_idx)

##########################################################################################
################ START OF PLOTTING #######################################################

    # build figure
    fig = plt.figure(figsize=(18, 8))  # draw figure

    # see if the alignment array has already been (partly) filled to skip processed cells
    if len(np.unique(alignment_array)) != 1:
        start_ref = np.where(alignment_array == -1)[0][0]   # row of first -1 shows with which reference cell to start
        start_real = np.where(alignment_array == -1)[1][0]   # col of first -1 shows with which target session to start
        if start_real == ref_session:
            start_tar = 0
        else:
            start_tar = real_to_targ_idx(start_real)
    else:   # otherwise start with the first cell
        start_ref = 0
        start_tar = 0

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
        alignment[ref_id, ref_session] = place_cell_idx[ref_id]   # which ID did this place cell have?
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


#%%
dir_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191125\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191126b\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191127a\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191206\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191207\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191208\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191219\N2']
ref_session = 0

spatial, templates, dim, pcf_objects = load_multisession_data(dir_list)


target_session_list, place_cell_indices, alignment_array, all_contours_list, all_shifts_list = prepare_manual_alignment_data(pcf_objects, ref_session)

alignment_array = manual_place_cell_alignment(pcf_sessions=pcf_objects,
                                              target_sessions=target_session_list,
                                              place_cell_idx=place_cell_indices,
                                              alignment=alignment_array,
                                              all_contours=all_contours_list,
                                              all_shifts=all_shifts_list,
                                              ref_session=ref_session)

save_alignment(alignment_array, ref_session, dir_list)

#%%

spatial_union, assignments, matchings, spatial, templates, pcf_objects = align_multisession(dir_list)

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


def find_pc_id(pcf_obj, neur_idx):
    for j in range(len(pcf_obj.place_cells)):
        if pcf_obj.place_cells[j][0] == neur_idx:
            return j

idx_file_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191125.txt']
# load data from txt files
idx_file_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191125.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191126b.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191127a.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191206.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191207.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191208.txt',
                 r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\pc_alignment_20191219.txt']

alignments_all = []
for file in idx_file_list:
    alignments_all.append(np.loadtxt(file, delimiter='\t'))
pc_idx_list = []
for obj in pcf_objects:
    pc_idx_list.append([x[0] for x in obj.place_cells])


# skip cells that didnt get recognized in all sessions
alignments = []
for session in alignments_all:
    alignments.append(session[np.all(session != -10, axis=1)])

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
for sess_idx in range(len(alignments)):
    for row in range(alignments[sess_idx].shape[0]):
        double_pc = False
        for column in range(len(pc_idx_list)):
            if alignments[sess_idx][row, column] in pc_idx_list[column] and column != sess_idx:
                double_pc = True
                print(f'Place cell {alignments[sess_idx][row, 0]} of session is also a place cell in sess {column+1}.')
        if double_pc and not any((alignments[sess_idx][row] == x).all() for x in two_sess_pc):
            two_sess_pc.append(alignments[sess_idx][row])


# plot aligned cells
def plot_aligned_cells(cell_list, pcf_objects_list, color=False):

    nrows = len(cell_list)
    ncols = len(pcf_objects_list)
    all_cells = []

    # order place cells (first cell in each row) by the start of its place cell
    # sort neurons after different criteria
    bins = []
    for i in range(nrows):
        curr_pc_id = find_pc_id(pcf_objects_list[0], int(cell_list[i][0]))
        bins.append((i, pcf_objects_list[0].place_cells[curr_pc_id][1][0][0]))  # get the first index of the first place field
    bins_sorted = sorted(bins, key=lambda tup: tup[1])
    bins_sort = [x[0] for x in bins_sorted]

    # get activity data of all cells
    for nrow in range(nrows):
        this_cell = np.ones((ncols, 80))
        for col in range(ncols):
            curr_neur_id = int(cell_list[bins_sort[nrow]][col])
            this_cell[col] = pcf_objects_list[col].bin_avg_activity[curr_neur_id]
        all_cells.append(this_cell)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8), sharex=True)
    fig.suptitle('Place cells in session 1 tracked over time', fontsize=22)
    for nrow in range(nrows):
        cell_max = np.max(all_cells[nrow])
        cell_min = np.min(all_cells[nrow])
        for col in range(ncols):
            if nrow == 0:
                sess_date = pcf_objects_list[col].params['root'].split(os.path.sep)[-2]
                ax[nrow, col].set_title(sess_date)

                # set x ticks to VR position, not bin number
                ax[nrow, col].set_xlim(0, len(all_cells[nrow][col]))
                ax[nrow, col].get_xaxis().set_ticks([0, 40, 80])
                x_locs, labels = plt.xticks()
                plt.sca(ax[nrow, col])
                plt.xticks(x_locs, (x_locs * pcf_objects_list[col].params['bin_length']).astype(int), fontsize=15)

            if col == 0:
                ax[nrow, col].spines['top'].set_visible(False)
                ax[nrow, col].spines['right'].set_visible(False)
                if color:
                    ax[nrow, col].get_yaxis().set_ticks([])
            else:
                ax[nrow, col].spines['top'].set_visible(False)
                ax[nrow, col].spines['right'].set_visible(False)
                ax[nrow, col].spines['left'].set_visible(False)
                ax[nrow, col].get_yaxis().set_ticks([])

            if color:
                img = ax[nrow, col].pcolormesh(all_cells[nrow][col, np.newaxis],
                                               vmax=cell_max, vmin=cell_min, cmap='jet')
                if col == ncols-1:
                    fraction = 0.15  # fraction of original axes to use for colorbar
                    # half_size = int(np.round(ax.shape[0] / 2))  # plot colorbar in half of the figure
                    cbar = fig.colorbar(img, ax=ax[nrow, :], fraction=fraction, pad=0.01, aspect=5, label=r'$\Delta$F/F')  # draw color bar
                    cbar.ax.tick_params(labelsize=12)
            else:
                ax[nrow, col].plot(all_cells[nrow][col])
                ax[nrow, col].set_ylim(bottom=cell_min, top=cell_max)
                pc_idx = find_pc_id(pcf_objects_list[col], int(cell_list[bins_sort[nrow]][col]))
                if pc_idx is not None:
                    curr_place_fields = pcf_objects_list[col].place_cells[pc_idx][1]
                    for field in curr_place_fields:
                        ax[nrow, col].axvspan(field.min(), field.max(), facecolor='r', alpha=0.2)
                        #ax[nrow, col].plot(field, all_cells[nrow][col][field], color='red')
    ax[nrows-1, int(ncols/2)].set_xlabel('Position in VR [cm]', fontsize=15)
    if not color:
        ax[int(nrows / 2), 0].set_ylabel(r'$\Delta$F/F', fontsize=15)


plot_aligned_cells(two_sess_pc, pcf_objects, color=True)