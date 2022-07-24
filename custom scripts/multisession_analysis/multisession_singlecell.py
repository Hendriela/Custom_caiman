from standard_pipeline import place_cell_pipeline as pipe
import numpy as np
from glob import glob
import pandas as pd
import os
import pickle
import seaborn as sns
from multisession_analysis import multisession_registration as multi
from multisession_analysis import singlecell as single
import matplotlib.pyplot as plt
#
# roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191122a\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191125\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191126b\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191127a\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191205\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191206\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191207\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191208\N2',
#          r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191219\N2']


#%% new version

paths = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200818.txt',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200819.txt',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200820.txt',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200821.txt',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200824.txt']


alignments = multi.load_alignment(paths)
data, traces, unique = multi.align_traces(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33', alignments)

# All cells
corr_matrix_all = single.correlate_activity(traces, unique, stroke='20200825', ignore=['20200826'], title='All cells', show_coef=True, borders=True)

# Split into place cells and non-place cells
pc_traces, non_pc_traces = single.split_place_cells(traces, data, unique)
fig, ax = plt.subplots(1,2, figsize=(16,8))
corr_matrix_pc = single.correlate_activity(pc_traces, unique, stroke='20200825', ignore=['20200826'],
                                           title='Place cells', ax=ax[0], borders=True)
corr_matrix_non_pc = single.correlate_activity(non_pc_traces, unique, stroke='20200825', ignore=['20200826'],
                                               title='Non-Place cells', ax=ax[1], borders=True)

# Save data in txt files for prism
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\correlation\all_cells.txt', corr_matrix_all, fmt='%.4f', delimiter='\t')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\correlation\only_pcs.txt', corr_matrix_pc, fmt='%.4f', delimiter='\t')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\correlation\no_pcs.txt', corr_matrix_non_pc, fmt='%.4f', delimiter='\t')

# Get spike rate from single cells
spikerate_all = single.get_single_spikerate(unique, data, split_pc=False)
spikerate_pc, spikerate_non_pc = single.get_single_spikerate(unique, data, split_pc=True)

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\spikerate_all.txt', spikerate_all, fmt='%.4f', delimiter='\t')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\spikerate_pc.txt', spikerate_pc, fmt='%.4f', delimiter='\t')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\spikerate_non_pc.txt', spikerate_non_pc, fmt='%.4f', delimiter='\t')

#%% Plot activity of all unique cells across sessions

n_rows = 2
n_cols = 5

# Extract place cell IDs before
pc_ids = {sess_idx: [pc[0] for pc in sess.place_cells] for sess_idx, sess in data.items()}

# Plot sessions for every unique cell (rows in DF)
for idx, row in unique.iterrows():

    # Create new figure for each neuron
    fig, ax = plt.subplots(n_rows, n_cols, sharey="all", sharex="all", figsize=(20, 6))
    ax_row = 0
    ax_col = 0

    ax[0, 0].set_ylabel('Prestroke', fontsize=18)
    ax[1, 0].set_ylabel('Poststroke', fontsize=18)

    # Plot data for every session
    for session_idx in row.index:
        # Skip session of poststroke day 1 because mouse was not running, data is meaningless
        if session_idx != '20200826':
            # -10 means that cell was not found in that session
            if not int(row[session_idx]) == -10:
                # Plot data
                ax[ax_row, ax_col].plot(data[session_idx].bin_avg_activity[int(row[session_idx])])

                # If the current cell is a place cell in the current session, draw place fields red
                if int(row[session_idx]) in pc_ids[session_idx]:
                    pc_idx = pc_ids[session_idx].index(int(row[session_idx]))
                    for pf in data[session_idx].place_cells[pc_idx][1]:
                        ax[ax_row, ax_col].axvspan(pf[0], pf[-1], color='r', alpha=0.3)

            # Set session date as subplot title
            ax[ax_row, ax_col].set_title(f"{session_idx} - Idx {int(row[session_idx])}")
            ax[ax_row, ax_col].spines['top'].set_visible(False)
            ax[ax_row, ax_col].spines['right'].set_visible(False)

            # Figure out coordinates of next axes
            if ax_col + 1 == n_cols:
                ax_row += 1
                ax_col = 0
            else:
                ax_col += 1

    plt.tight_layout()
    plt.savefig(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\activity_plots" + f"\\unique_cell_{idx}.png")
    plt.close(fig)


#%% Load pcf files into a dict for better indexing

pcf_dict = {}
for root in roots:
    pcf = pipe.load_pcf(root)
    pcf_dict[pcf.params['session']] = pcf

with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\pcf_results.pickle', 'rb') as file:
    pcf_dict['20191204'] = pickle.load(file)

#%% load alignment files and store data in a DataFrame
alignment_root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis'

alignment_paths = glob(alignment_root+r'\pc_alignment*.txt')
sess_list = [path.split(os.path.sep)[-2] for path in roots]

# load alignment files and enter data into a data frame with columns (data, glob_id, session, sess_id, pc_sess)
single_rows = []
glob_id = 0
for path in alignment_paths:
    file = np.loadtxt(path)
    pc_sess = path.split(os.path.sep)[-1].split('_')[-1][:-4]  # isolate session date

    # go through the table and enter data of current cell into a temporary data frame
    for cell in range(file.shape[0]):
        for sess in range(file.shape[1]):
            session = sess_list[sess]                               # get name of current session
            sess_id = int(file[cell, sess])                         # get ID of current neuron in the current session
            if sess_id != -10:
                data = pcf_dict[session].cnmf.estimates.F_dff[sess_id]  # get calcium data of that cell
                # append row to a list that will be concatenated to a single DF in the end
                single_rows.append(pd.DataFrame({'data': [data], 'glob_id': int(glob_id), 'session': session,
                                                 'sess_id': int(sess_id), 'pc_sess': pc_sess}))
        glob_id += 1

alignment_df = pd.concat(single_rows, ignore_index=True)

# calculate spatial information
cell_si = np.zeros(alignment_df.shape[0])
for cell in range(alignment_df.shape[0]):
    curr_session = alignment_df['session'][cell]
    cell_si[cell] = pipe.get_spatial_info(alignment_df['data'][cell], pcf_dict[curr_session].behavior, n_bootstrap=0)
alignment_df['SI'] = cell_si

ax = sns.lineplot(x='session', y='SI norm', data=alignment_df[alignment_df['glob_id'] == 40])

# normalize SI (divide by SI of place cell session?)
cell_si_norm = np.zeros(alignment_df.shape[0])
idx = 0
for cell in np.unique(alignment_df['glob_id']):
    cell_sis = alignment_df[alignment_df['glob_id'] == cell]
    pc_si = float(cell_sis['SI'][alignment_df['session'] == alignment_df['pc_sess']])
    si_array = np.array(cell_sis['SI'])
    for j in range(len(si_array)):
        cell_si_norm[idx] = si_array[j]/pc_si
        idx += 1
alignment_df['SI norm'] = cell_si_norm

# get ratio of pre- vs poststroke SI
prestroke_sessions = ['20191122a', '20191125', '20191126b', '20191127a', '20191204']
poststroke_sessions = ['20191205', '20191206', '20191207', '20191208', '20191219']
cell_si_ratio = np.zeros(alignment_df.shape[0])
idx = 0
for cell in np.unique(alignment_df['glob_id']):
    si_pre = []
    si_post = []
    cell_sis = alignment_df[alignment_df['glob_id'] == cell]
    si_pre = cell_sis['SI'][np.where(cell_sis['session'] == prestroke_sessions)]
    si_array = np.array(cell_sis['SI'])
    for j in range(len(si_array)):
        cell_si_ratio[idx] = si_array[j]/pc_si
        idx += 1
alignment_df['SI norm'] = cell_si_norm
