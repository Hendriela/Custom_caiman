
from caiman.source_extraction.cnmf import params as params
import place_cell_pipeline as pipe
import behavior_import as behavior

"""
Complete pipeline for place cell analysis, from motion correction to place cell finding

Condition before starting:
Tiff and behavioral files (encoder, position, frames/licks) have to be in separate folders for each trial, and trial
folders have to be grouped in one folder per field of view. Several FOVs can be grouped in a folder per session.
"""

#%% Set parameters

# dataset dependent parameters
fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds (0.4)
dxy = (1, 1)  # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here
max_shift_um = (20., 20.)  # maximum shift in um
patch_motion_um = (40., 40.)  # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = True  # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (12, 12)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fnames': None,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy'
}

opts = params.CNMFParams(params_dict=mc_dict)


#%% Set working directory

folder_list, tif_list = pipe.set_file_paths()
opts = opts.change_params({'fnames': tif_list})

#%% Perform motion correction
motion_file = pipe.motion_correction(opts, remove_f_order=True)

#%% Import behavioral data
behavior.align_files(folder_list)


