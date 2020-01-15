import place_cell_pipeline as pipe
from glob import glob
from caiman import load_memmap
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid

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


def manual_place_cell_alignment(pcf):

    # get indices of place cells from first session
    place_cell_idx = [x[0] for x in pcf.place_cells]

    fig = plt.figure(figsize=(15, 8))  # draw figure
    outer = grid.GridSpec(1, 2)        # initialize outer structure (two fields horizontally)

    # draw reference cell in the big plot on the left
    ref = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ref_ax = fig.add_subplot(ref[0])
    draw_single_contour(ref_ax, pcf.cnmf.estimates.A[:, place_cell_idx[0]], pcf.cnmf.estimates.Cn)
    ref_ax.tick_params(labelbottom=False, labelleft=False)

    # draw possible matching cells in the plots on the right
    candidates = outer[1].subgridspec(3, 3)

    for i in range(3):
        for j in range(3):
            curr_ax = fig.add_subplot(candidates[i, j], picker=True) # picker activates clickability of the subplot
            plt.setp(curr_ax, url=(i, j))   # the url property is used as a label to know which axis has been clicked
            t = curr_ax.text(0.5, 0.5, 'ax (%d,%d)' % (i, j), va='center', ha='center')
            curr_ax.tick_params(labelbottom=False, labelleft=False)

    def onpick(event):
        this_plot = event.artist                    # save artist (axis) where the pick was triggered
        neuron_id = plt.getp(this_plot, 'url')      # get the url label of the plot that shows which neuron was plotted
        print(neuron_id)

    plt.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.show()


def draw_single_contour(ax, spatial, template, half_size=25):
    """
    Draws the contour of one component and focuses the template image around its center.
    :param ax: Axes in which the plot should be drawn
    :param spatial: spatial information of the component, acquired by indexing spatial[session][:,neuron_id]
    :param template: background template image on which to draw the contour
    :param half_size: int, half size in pixels of the final area
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


