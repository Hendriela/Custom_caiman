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
from copy import deepcopy

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
    spatial = []  # list of csc matrices (# pixels X # component ROIs, from cnmf.estimates.A) with spatial info for all components
    templates = []  # list of template images for each session (local correlation map; cnmf.estimates.Cn)
    pcf_objects = []  # list of pcf objects that contain all other infos about the place cells

    count = 0
    for folder in dir_list:
        count += 1
        curr_pcf = pipe.load_pcf(folder)    # load pcf object that includes the cnmf object
        spatial.append(curr_pcf.cnmf.estimates.A)
        try:
            templates.append(curr_pcf.cnmf.estimates.Cn)
        except AttributeError:
            print(f'No local correlation image found in {folder}...')
            movie_file = glob(folder+'memmap__*.mmap')
            if len(movie_file) == 1:
                print(f'\tFound mmap file, local correlation image is being calculated.')
                Yr, dims, T = load_memmap(movie_file[0])
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                curr_pcf.cnmf.estimates.Cn = pipe.get_local_correlation(images)
                curr_pcf.save(overwrite=True)
                templates.append(curr_pcf.cnmf.estimates.Cn)
            elif len(movie_file) > 1:
                print('\tResult ambiguous, found more than one mmap file, no calculation possible.')
            else:
                print('\tNo mmap file found. Maybe you have to re-motioncorrect the movie?')

        pcf_objects.append(curr_pcf)    # add rest of place cell info to the list
        print(f'Successfully loaded data from {folder[-18:]} ({count}/{len(dir_list)}).')

    dim = templates[0].shape  # dimensions of the FOV, can be gotten from templates

    return spatial, templates, dim, pcf_objects
#%% SEMI-MANUALLY ALIGNING PLACE CELLS ACROSS SESSIONS

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
    com_shift = [com[0] + shift[0], com[1] + shift[1]]  # shift the CoM by the given amount
    # cap CoM at 0 and dims limits
    com_shift = [0 if x < 0 else x for x in com_shift]
    for coord in range(len(com_shift)):
        if com_shift[coord] > dims[coord]:
            com_shift[coord] = dims[coord]
    return com_shift


def manual_place_cell_alignment(pcf_sessions, ref_session=0):

    def adj_idx(idx):
        """
        Adjusts index of pcf_sessions list to avoid referring to the reference session.
        :param idx: Index of non-reference session
        :return: adj_idx, actual index of requested session in pcf_sessions
        """
        adj_idx = idx
        if adj_idx >= ref_session:
            adj_idx += 1
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
        plt.setp(ax, url=idx, title=f'Session {ref_session + 1}')
        ax.tick_params(labelbottom=False, labelleft=False)
        return com

    def find_target_cells(reference_com, session_contours, dims, fov_shift, max_dist=25):
        """
        Finds cells in tar_sess that are close to the reference cell
        :param reference_com: tuple, (x,y) of center of mass of reference cell
        :param session_contours: list of contour dicts holding contour information of all target cells
        :param dims: tuple, (x,y) of FOV dimensions
        :param fov_shift: (x,y) of FOV shift between reference and target session (from all_shifts)
        :param max_dist: int, maximum radial distance between reference and target cell to be considered 'nearby'
        :return: list of contours of nearby cells sorted by distance
        """
        # Correct reference center-of-mass by the shift to be better aligned with the FOV of the second session
        reference_com_shift = shift_com(reference_com, fov_shift, dims)
        ref_point = Point2D(reference_com_shift[0],
                            reference_com_shift[1])  # transform CoM into a Point2D that can handle radial distances

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

    def draw_target_cells(near_contours_sort, idx, update=False):
        """
        Draws target cells in the right part of the figure.
        :param near_contours_sort: list, contains contour data of target cells (from find_target_cells)
        :param idx: int, index of the target session the target cells belong to
        :return:
        """
        real_idx = adj_idx(idx) # exchange the target-session specific idx into a pcf_sessions index
        # draw possible matching cells in the plots on the right
        if len(near_contours_sort)+1 <= 9:  # the +1 is for the button "no matches"
            n_rows = ceil((len(near_contours_sort)+1)/3)    # number of rows needed to plot near contours into 3 columns
            candidates = outer[1].subgridspec(n_rows, 3)    # create grid layout
            counter = 0                                     # counter that keeps track of plotted contour number
            for row in range(n_rows):
                for column in range(3):
                    curr_ax = fig.add_subplot(candidates[row, column], picker=True)  # picker enables clicking subplots
                    if row == 0 and column == 1:
                        curr_ax.set_title(f'Session {real_idx+1}')
                    try:
                        curr_neuron = near_contours_sort[counter]['neuron_id']
                        # plot the current candidate
                        curr_cont = draw_single_contour(ax=curr_ax,
                                                        spatial=target_sessions[idx].cnmf.estimates.A[:, curr_neuron],
                                                        template=target_sessions[idx].cnmf.estimates.Cn)
                        # the url property of the Axes is used as a tag to remember which neuron has been clicked
                        # as well as which target session it belonged to
                        plt.setp(curr_ax, url=(real_idx, curr_neuron))
                        curr_ax.tick_params(labelbottom=False, labelleft=False)
                        counter += 1
                    # if there are no more candidates to plot, make plot into a "no matches" button and mark it with -1
                    except IndexError:
                        t = curr_ax.text(0.5, 0.5, 'No Matches', va='center', ha='center')
                        plt.setp(curr_ax, url=(real_idx, -1))
                        curr_ax.tick_params(labelbottom=False, labelleft=False)
        else:
            #TODO Fix plotting of many cells, this is just a placeholder
            candidates = outer[1].subgridspec(1, 1)    # create grid layout
            curr_ax = fig.add_subplot(candidates, picker=True)  # picker enables clicking subplots
            t = curr_ax.text(0.5, 0.5, 'More than 9 cells! Fix plotting!', va='center', ha='center')
            plt.setp(curr_ax, url=-1)
            curr_ax.tick_params(labelbottom=False, labelleft=False)

    def draw_both_sides(ref_idx, targ_sess_idx):

        # build figure
        fig = plt.figure(figsize=(15, 8))  # draw figure
        outer = grid.GridSpec(1, 2)  # initialize outer structure (two fields horizontally)
        ref = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])  # initialize reference plot
        ref_ax = fig.add_subplot(ref[0])  # draw reference plot

        # first draw the reference cell
        ref_com = draw_reference(ref_ax, pcf_sessions[ref_session], ref_idx)

        # Find cells in the next session(s) that have their center of mass near the reference cell
        nearby_contours = find_target_cells(ref_com, all_contours[targ_sess_idx],
                                            pcf_sessions[ref_idx].cnmf.estimates.Cn.shape,
                                            all_shifts[targ_sess_idx])

        # Draw target cells in the right plots
        draw_target_cells(nearby_contours, targ_sess_idx)

        return nearby_contours, ref_ax, fig

    #######################################################################################
    ##### THIS FIGURE AND DATA PREPARATION HAS TO BE DONE ONLY ONCE PER FUNCTION CALL #####

    target_sessions = [x for j, x in enumerate(pcf_sessions) if j != ref_session]
    # get indices of place cells from first session
    place_cell_idx = [x[0] for x in pcf_sessions[ref_session].place_cells]

    # initialize alignment array (#place cells X #sessions)
    alignment = np.zeros((len(place_cell_idx), len(pcf_sessions)))  # +1 because the reference session has been popped

    # get contour data of all cells and FOV shifts between reference and the other sessions with phase correlation
    all_contours = []
    all_shifts = []
    for sess_idx in range(len(pcf_sessions)):
        if sess_idx != ref_session:
            sess = pcf_sessions[sess_idx]
            all_shifts.append(register_translation(pcf_sessions[ref_session].cnmf.estimates.Cn, sess.cnmf.estimates.Cn,
                                                   upsample_factor=100, return_error=False))
            plt.figure()
            all_contours.append(visualization.plot_contours(sess.cnmf.estimates.A, sess.cnmf.estimates.Cn))
            plt.close()

    # First drawing
    near_contours, ref_ax, fig = draw_both_sides(0, 0)  # This means to draw the first reference (place) cell and the target cells

    # Define what happens when a plot was clicked: update alignment accordingly and draw the next set of cells
    def onpick(event):
        global near_contours, ref_ax, fig
        this_plot = event.artist                 # save artist (axis) where the pick was triggered
        id_s = plt.getp(this_plot, 'url')        # get the IDs of the neuron that was clicked and of the current session
        ref_id = plt.getp(ref_ax, 'url')         # get the url label (idx) of the reference cell
        print(f'Reference cell {ref_id}, clicked neuron {id_s}.')

        # update the alignment array with the clicked assignment
        alignment[ref_id, 0] = place_cell_idx[ref_id]   # which ID did this place cell have?
        if id_s[1] == -1:  # '-1' means "no match" has been clicked, fill spot with nan
            alignment[ref_id, id_s[0]] = np.nan
        else:
            # assign the clicked neuron_id to the 'ref_id' reference cell in the 'sess_id' session
            alignment[ref_id, id_s[0]] = id_s[1]


        # if this is the last session, draw a new reference cell
        if id_s[0]+1 == len(pcf_sessions):
            near_contours, ref_ax, fig = draw_both_sides(ref_id+1, id_s[0]+1)
        # else, only draw the target cells of the next session
        else:
            draw_target_cells(near_contours, id_s[0]+1, update=True)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.show()



#%%
dir_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191125\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191126b\N2',
            r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191127a\N2']

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


