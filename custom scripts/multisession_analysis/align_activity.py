import standard_pipeline.place_cell_pipeline as pipe
import pandas as pd
import numpy as np

session = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200627'
window_size = (2, 4)

# Load PCF
pcf = pipe.load_pcf(session)

# Set RZ borders
if pcf.params['novel']:
    zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
else:
    zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])

# Find frame count of valve openings
df_hit_list = []
df_miss_list = []
for trial_idx, trial in enumerate(pcf.behavior):
    # Recover frames where the mouse was not moving and get frame indices
    frame_idx = np.where(np.nan_to_num(trial[:, 3], nan=1) == 1)[0]
    valve_idx = np.where(trial[:, 6] == 1)[0]
    # Check if any reward zone was without reward
    for idx, zone in enumerate(zone_borders):
        zone_data = trial[np.logical_and(zone[0] < trial[:, 1], trial[:, 1] < zone[1]), :]
        if sum(zone_data[:, 6]) == 0:
            df_miss_list.append(pd.DataFrame({'trial': [trial_idx], 'zone':[idx]}))

    # For every valve opening, get the index of the next frame and save it for later alignment
    for reward in valve_idx:
        # Check if the current valve opening happened in a reward zone
        rzs = [True if (zone[0] < trial[reward, 1] < zone[1]) else False for zone in zone_borders]
        in_rz = any(rzs)
        if in_rz:
            # Check the time delay of entering the reward zone and valve opening
            time_diff = trial[reward, 0] - trial[np.where(trial[:, 1] > zone_borders[np.where(rzs)[0][0], 0])[0][0], 0]
            # Get the valve opening distance from the reward zone start
            pos_diff = trial[reward, 1] - zone_borders[np.where(rzs)[0][0], 0]

        # Get frame count differences (to find the next frame index)
        diff = frame_idx - reward
        df_hit_list.append(pd.DataFrame({'trial': [trial_idx], 'frame': [np.where(diff > 0, diff, np.inf).argmin()],
                                         'rz': [in_rz], 'pos': pos_diff, 'delay': time_diff}))
if len(df_hit_list) > 0:
    df_hit = pd.concat(df_hit_list, ignore_index=True)
if len(df_miss_list) > 0:
    df_miss = pd.concat(df_miss_list, ignore_index=True)

# For every neuron, align traces to all reward frames
align = np.zeros((len(pcf.cnmf.estimates.F_dff), (window_size[0]+window_size[1])*pcf.cnmf.params.data['fr'], len(df_hit)))
for cell_idx in range(len(pcf.session)):
    for rew_idx, rew in df_hit.iterrows():
        first_frame = rew['frame']-window_size[0]*pcf.cnmf.params.data['fr']
        last_frame = rew['frame']+window_size[1]*pcf.cnmf.params.data['fr']
        align[cell_idx, :, rew_idx] = pcf.session[cell_idx][rew['trial']][first_frame:last_frame]


