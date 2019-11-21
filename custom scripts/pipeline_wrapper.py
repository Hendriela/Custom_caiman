from caiman.source_extraction import cnmf
import place_cell_pipeline as pipe
import behavior_import as behavior
import place_cell_class as pc

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
dxy = (1.66, 1.52)  # spatial resolution in x and y in (um per pixel) [(1.66,1.52) for 1x, (0.83,0.76) for 2x]
# note the lower than usual spatial resolution here
max_shift_um = (50., 50.)  # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

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

opts = cnmf.params.CNMFParams(params_dict=mc_dict)


#%% Set working directory

root = pipe.set_file_paths()

#%% Perform motion correction
motion_file = pipe.motion_correction(root, opts, remove_f_order=True)

#%% Align behavioral data
#behavior.align_files(folder_list, performance_check=True)
behavior.align_behavior(root, performance_check=True)

# evaluate behavior
mouse_list = []
#%% CaImAn source extraction

#%% Initialize PlaceCellFinder object

cnm = cnmf.cnmf.load_CNMF(r'E:\PhD\Data\DG\M14_20191014\N2\N2_results.hdf5')
params = {'root': root,                  # main directory of this session
          'trial_list': folder_list,     # list of all trial directories in this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 2,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size_cm': 10,          # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.2,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 170,           # length in cm of the virtual reality corridor
          'split_size': 10}              # size in frames of bootstrapping segments

pcf = pc.PlaceCellFinder(cnm, params)

# split traces into trials
pcf.split_traces_into_trials()
# create significant-transient-only traces
pcf.create_transient_only_traces()
# align the frames to the VR position using merged behavioral data
pcf.import_behavior_and_align_traces()
# look for place cells
pcf.find_place_cells()
#%%

