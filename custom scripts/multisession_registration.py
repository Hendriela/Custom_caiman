import place_cell_pipeline as pipe
from glob import glob
from caiman import load_memmap
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from skimage.feature import register_translation
from scipy.ndimage import shift as shift_img
from math import ceil
from point2d import Point2D

#%%


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


def manual_place_cell_alignment(pcf_objects):


    # get indices of place cells from first session
    place_cell_idx = [x[0] for x in pcf_objects[0].place_cells]

    fig = plt.figure(figsize=(15, 8))  # draw figure
    outer = grid.GridSpec(1, 2)        # initialize outer structure (two fields horizontally)

    # draw reference cell in the big plot on the left
    ref = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ref_ax = fig.add_subplot(ref[0])
    ref_com = draw_single_contour(ax=ref_ax,
                                  spatial=pcf_objects[0].cnmf.estimates.A[:, place_cell_idx[0]],
                                  template=pcf_objects[0].cnmf.estimates.Cn)
    ref_ax.tick_params(labelbottom=False, labelleft=False)

    ### Find cells in the next session(s) that have their center of mass near the reference cell (25 px distance) ###
    max_dist = 25  # maximum radial distance of a cell to the reference cell in pixel

    # Calculate offset of FOVs with phase correlation (skimage.feature.register_translation)
    shift = register_translation(pcf_objects[0].cnmf.estimates.Cn, pcf_objects[1].cnmf.estimates.Cn,
                                 upsample_factor=100, return_error=False)

    # Correct reference center-of-mass by the shift to be better aligned with the FOV of the second session
    ref_com_shift = shift_com(ref_com, shift, pcf_objects[0].cnmf.estimates.Cn.shape)
    ref_point = Point2D(ref_com_shift[0], ref_com_shift[1])  # transform CoM into a Point2D to handle radial distances

    # get CoM data for all cells in the second session
    plt.figure()
    all_contours = visualization.plot_contours(pcf_objects[1].cnmf.estimates.A, pcf_objects[1].cnmf.estimates.Cn)
    plt.close()

    # Go through all cells and select the ones that have a CoM near the reference cell, also remember their distance
    near_contours = []
    distances = []
    for contour in all_contours:
        new_point = Point2D(contour['CoM'][0], contour['CoM'][1])
        rad_dist = (new_point - ref_point).r
        # x_dist = abs(contour['CoM'][0] - ref_com_shift[0])
        # y_dist = abs(contour['CoM'][1] - ref_com_shift[1])
        if rad_dist <= max_dist:
            near_contours.append(contour)
            distances.append(rad_dist)

    # sort the near contours by their distance to the reference CoM
    near_contours_sort = [x for _, x in sorted(zip(distances, near_contours), key=lambda pair: pair[0])]

    # draw possible matching cells in the plots on the right
    if len(near_contours_sort)+1 <= 9:  # the +1 is for the button "no matches"
        n_rows = ceil((len(near_contours_sort)+1)/3)    # number of rows necessary to plot near contours into 3 columns
        candidates = outer[1].subgridspec(n_rows, 3)    # create grid layout
        counter = 0                                     # counter that keeps track of plotted contour number
        for row in range(n_rows):
            for column in range(3):
                curr_ax = fig.add_subplot(candidates[row, column], picker=True)  # picker enables clicking the subplot
                try:
                    curr_neuron = near_contours_sort[counter]['neuron_id']
                    # plot the current candidate
                    curr_cont = draw_single_contour(ax=curr_ax,
                                                    spatial=pcf_objects[1].cnmf.estimates.A[:, curr_neuron],
                                                    template=pcf_objects[1].cnmf.estimates.Cn)
                    # the url property of the Axes is used as a tag to remember which neuron has been clicked
                    plt.setp(curr_ax, url=curr_neuron)
                    curr_ax.tick_params(labelbottom=False, labelleft=False)
                    counter += 1
                # if there are no more candidates to plot, make plot into a "no matches" button and mark it with '-1'
                except IndexError:
                    t = curr_ax.text(0.5, 0.5, 'No Matches', va='center', ha='center')
                    plt.setp(curr_ax, url=-1)
                    curr_ax.tick_params(labelbottom=False, labelleft=False)

    def onpick(event):
        this_plot = event.artist                    # save artist (axis) where the pick was triggered
        neuron_id = plt.getp(this_plot, 'url')      # get the url label of the plot that shows which neuron was plotted
        print(neuron_id)

    plt.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.show()


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


