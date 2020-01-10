import place_cell_pipeline as pipe
from glob import glob
import caiman as cm
import numpy as np


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

    for folder in dir_list:
        curr_pcf = pipe.load_pcf(folder)    # load pcf object that includes the cnmf object
        spatial.append(curr_pcf.cnmf.estimates.A)
        try:
            templates.append(curr_pcf.cnmf.estimates.Cn)
        except AttributeError:
            print(f'No local correlation image found in {folder}...')
            movie_file = glob(folder+'memmap__*.mmap')
            if len(movie_file) == 1:
                print(f'\tFound mmap file, local correlation image is being calculated.')
                Yr, dims, T = cm.load_memmap(movie_file[0])
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                curr_pcf.cnmf.estimates.Cn = pipe.get_local_correlation(images)
                curr_pcf.save(overwrite=True)
                templates.append(curr_pcf.cnmf.estimates.Cn)
            elif len(movie_file) > 1:
                print('\tResult ambiguous, found more than one mmap file, no calculation possible.')
            else:
                print('\tNo mmap file found. Maybe you have to re-motioncorrect the movie?')

        delattr(curr_pcf, 'cnmf')       # delete cnmf attribute from pcf object to save memory
        pcf_objects.append(curr_pcf)    # add rest of place cell info to the list
        print(f'Successfully loaded data from {folder}.')

    dim = templates[0].shape  # dimensions of the FOV, can be gotten from templates

    return spatial, templates, dim, pcf_objects
