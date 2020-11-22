import standard_pipeline.place_cell_pipeline as pipe
import matplotlib.pyplot as plt
import numpy as np
import scipy
import caiman.source_extraction.cnmf.utilities as util
import caiman as cm
from caiman.utils import stats

#%% Load example dataset
cnm = pipe.load_pcf(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200625').cnmf
cell = 49
frames_window=1000

A = cnm.estimates.A
C = cnm.estimates.C
YrA = cnm.estimates.YrA
F = C + YrA
b = cnm.estimates.b
f = cnm.estimates.f
B = A.T.dot(b).dot(f)
F_dff = cnm.estimates.F_dff

Fd = scipy.ndimage.percentile_filter(F, 8, (frames_window, 1))
Fd_rev = scipy.ndimage.percentile_filter(F, 8, (1, frames_window))
Df = scipy.ndimage.percentile_filter(B, 8, (frames_window, 1))
Df_rev = scipy.ndimage.percentile_filter(B, 8, (1, frames_window))

# Auto-calc of quantile for baseline fluorescence
data_prct, val = util.df_percentile(F[:, :frames_window], axis=1)
Fd_auto = np.stack([scipy.ndimage.percentile_filter(f, prctileMin, (frames_window)) for f, prctileMin in zip(F, data_prct)])
Df_auto = np.stack([scipy.ndimage.percentile_filter(f, prctileMin, (frames_window)) for f, prctileMin in zip(B, data_prct)])

dFF_cells = (F - Fd) / (Df + Fd)
dFF_time = (F - Fd_rev) / (Df_rev + Fd_rev)
dFF_auto = (F - Fd_auto) / (Df_auto + Fd_auto)

#%% Creation graph F=C+YrA
fr = 30 # frame rate, set to 1 to plot traces against frame counts
fig, ax = plt.subplots(2, 3)
ax[0,0].plot(np.arange(len(C[cell]))/fr, C[cell])
ax[0,0].set_title('denoised (inferred) temporal trace C')
ax[0,0].axvline(5000/fr, color='r')
ax[0,0].axvline(5800/fr, color='r')
ax[0,1].plot(np.arange(len(YrA[cell]))/fr, YrA[cell])
ax[0,1].set_title('residual temporal trace R')
ax[0,1].axvline(5000/fr, color='r')
ax[0,1].axvline(5800/fr, color='r')
ax[0,2].plot(np.arange(len(C[cell]))/fr, F[cell])
ax[0,2].set_title('"filtered raw trace" F (C + R)')
ax[0,2].axvline(5000/fr, color='r')
ax[0,2].axvline(5800/fr, color='r')
ax[1,0].plot(np.arange(5000,5800)/fr, C[cell, 5000:5800])
# ax[1,0].set_title('estimates.F_dff zoom')
ax[1,1].plot(np.arange(5000,5800)/fr, YrA[cell, 5000:5800])
# ax[1,1].set_title('estimates.C zoom')
ax[1,2].plot(np.arange(5000,5800)/fr, F[cell, 5000:5800])
# ax[1,2].set_title('estimates.S zoom')
ax[1,0].set_xlabel('time [s]')
ax[1,1].set_xlabel('time [s]')
ax[1,2].set_xlabel('time [s]')
for axis in ax.ravel():
    axis.tick_params(axis='y', which='major', labelsize=15)
    axis.axhline(0, linestyle='--', color='grey')

#%% Creation graph temporal background
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(len(f[0]))/fr, f[0], label='comp 1')
ax[0].plot(np.arange(len(f[0]))/fr, f[1], label='comp 2')
ax[0].tick_params(axis='y', which='major', labelsize=15)
ax[0].legend()
ax[0].set_xlabel('time [s]')
ax[0].set_title('Global temporal background components')
ax[1].plot(np.arange(len(B[cell]))/fr, B[cell])
ax[1].tick_params(axis='y', which='major', labelsize=15)
ax[1].set_xlabel('time [s]')
ax[1].set_title('Temporal background component of example neuron\nB = A.T.dot(b).dot(f)')

#%% Creation graph F0 (baseline fluorescence)

plt.figure()
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,3)
ax3 = plt.subplot(2,2,(2,4))
ax1.plot(np.arange(len(f[0]))/fr, F[cell], label='F')
ax1.plot(np.arange(len(f[0]))/fr, Fd[cell], label='Base_F_cells')
ax1.plot(np.arange(len(f[0]))/fr, Fd_rev[cell], label='Base_F_time', linewidth=3)
ax1.plot(np.arange(len(f[0]))/fr, Fd_auto[cell], label='Base_F_auto ({:.1f})'.format(data_prct[cell]), linewidth=3)
ax1.axvline(14000/fr, color='r')
ax1.axvline(15800/fr, color='r')
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.axhline(0, linestyle='--', color='grey')
ax1.legend(fontsize=12)
ax1.set_title('Baseline of component fluorescence F (8th percentile)', fontsize=18)
ax2.plot((np.arange(len(f[0]))/fr)[14000:15800], F[cell][14000:15800], label='F')
ax2.plot((np.arange(len(f[0]))/fr)[14000:15800], Fd[cell][14000:15800], label='Base_F_cells')
ax2.plot((np.arange(len(f[0]))/fr)[14000:15800], Fd_rev[cell][14000:15800], label='Base_F_time', linewidth=3)
ax2.plot((np.arange(len(f[0]))/fr)[14000:15800], Fd_auto[cell][14000:15800], label='Base_F_auto ({:.1f})'.format(data_prct[cell]), linewidth=3)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.axhline(0, linestyle='--', color='grey')
ax2.legend(fontsize=12)
ax2.set_xlabel('time [s]', fontsize=15)
ax3.plot(np.arange(len(f[0]))/fr, B[cell], label='B')
ax3.plot(np.arange(len(f[0]))/fr, Df[cell], label='Base_B_cells')
ax3.plot(np.arange(len(f[0]))/fr, Df_rev[cell], label='Base_B_time', linewidth=3)
ax3.plot(np.arange(len(f[0]))/fr, Df_auto[cell], label='Base_B_auto ({:.1f})'.format(data_prct[cell]), linewidth=3)
ax3.legend(fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_xlabel('time [s]', fontsize=15)
ax3.set_title('Baseline of background fluorescence B (8th percentile)', fontsize=18)

#%% Plot mean and std of B
B_mean = np.mean(B, axis=0)
B_std = np.std(B, axis=0)
plt.figure();
plt.plot(np.arange(len(f[0]))/fr, B_mean, color='blue')
plt.fill_between(np.arange(len(f[0]))/fr, B_mean - B_std, B_mean + B_std, color='gray', alpha=0.2)

#%% Plot dF/F with different baselines
s1 = 5000
s2 = 5800
fig, ax = plt.subplots(2, 4)
ax[0,0].set_title('Fluorescence F (C + residuals)\nwith different baselines', fontsize=18)
ax[0,0].plot(np.arange(len(F[cell]))/fr, F[cell], label='F')
ax[0,0].plot(np.arange(len(f[0]))/fr, Fd[cell], label='F_base_cells', linewidth=2)
ax[0,0].plot(np.arange(len(f[0]))/fr, Fd_rev[cell], label='F_base_time', linewidth=2)
ax[0,0].plot(np.arange(len(f[0]))/fr, Fd_auto[cell], label='F_base_auto ({:.1f})'.format(data_prct[cell]), linewidth=2)
z = ax[0,0].legend()
ax[1,0].set_title('Background fluorescence B\nwith different baselines', fontsize=18)
ax[1,0].plot(np.arange(len(F[cell]))/fr, B[cell], label='B')
ax[1,0].plot(np.arange(len(f[0]))/fr, Df[cell], label='B_base_cells', linewidth=2)
ax[1,0].plot(np.arange(len(f[0]))/fr, Df_rev[cell], label='B_base_time', linewidth=2)
ax[1,0].plot(np.arange(len(f[0]))/fr, Df_auto[cell], label='B_base_auto ({:.1f})'.format(data_prct[cell]), linewidth=2)
ax[1,0].legend()

# Get color of baselines
cols = z.get_lines()

ax[0,1].set_title('dF/F with F_base_cells', fontsize=18)
ax[0,1].plot(np.arange(len(dFF_cells[cell]))/fr, dFF_cells[cell], color=cols[1].get_color())
ax[1,1].plot(np.arange(s1,s2)/fr, dFF_cells[cell, s1:s2], color=cols[1].get_color())
ax[0,1].axvline(s1/fr, color='r')
ax[0,1].axvline(s2/fr, color='r')

ax[0,2].set_title('dF/F with F_base_time', fontsize=18)
ax[0,2].plot(np.arange(len(dFF_time[cell]))/fr, dFF_time[cell], color=cols[2].get_color())
ax[0,2].axvline(s1/fr, color='r')
ax[0,2].axvline(s2/fr, color='r')
ax[1,2].plot(np.arange(s1,s2)/fr, dFF_time[cell, s1:s2], color=cols[2].get_color())

ax[0,3].set_title('dF/F with F_base_auto', fontsize=18)
ax[0,3].plot(np.arange(len(dFF_auto[cell]))/fr, dFF_auto[cell], color=cols[3].get_color())
ax[0,3].axvline(s1/fr, color='r')
ax[0,3].axvline(s2/fr, color='r')
ax[1,3].plot(np.arange(s1,s2)/fr, dFF_auto[cell, s1:s2], color=cols[3].get_color())

# set x label
ax[1,0].set_xlabel('time [s]', fontsize=15)
ax[1,1].set_xlabel('time [s]', fontsize=15)
ax[1,2].set_xlabel('time [s]', fontsize=15)
ax[1,3].set_xlabel('time [s]', fontsize=15)


ax[1,1].set_ylim(-0.2, 3.9)
ax[1,2].set_ylim(-0.2, 3.9)
ax[1,3].set_ylim(-0.2, 3.9)
ax[0,1].set_ylim(-0.2, 3.9)
ax[0,2].set_ylim(-0.2, 3.9)
ax[0,3].set_ylim(-0.2, 3.9)

for idx, axis in enumerate(ax.ravel()):
    axis.tick_params(axis='both', which='major', labelsize=15)
    if not idx == 4:
        axis.axhline(0, linestyle='--', color='grey')

#%% Quantile estimation via KDE
inputData1 = F[cell, :frames_window]
bandwidth1, mesh1, density1, cdf1 = stats.kde(inputData1)
data_prct1 = cdf1[np.argmax(density1)] * 100

inputData2 = F[153, :frames_window]
bandwidth2, mesh2, density2, cdf2 = stats.kde(inputData2)
data_prct2 = cdf2[np.argmax(density2)] * 100

# Plot inputs
fig, ax = plt.subplots(2, 3)
ax[0,0].plot(np.arange(frames_window)/fr, inputData1, label='Input')
ax[0,0].set_title('Input (first {:d} frames of "raw" trace)'.format(frames_window), fontsize=15)
ax[1,0].plot(np.arange(frames_window)/fr, inputData2, label='Input')
# Plot KDE results
ax[0,1].plot(density1, label='density')
ax[0,1].axvline(np.argmax(density1), c='r')
ax[0,1].set_title('Kernel density estimator', fontsize=15)
ax_twin = ax[0,1].twinx()
ax_twin.plot(cdf1, label='cumulative density', color='g')
ax_twin.axhline(cdf1[np.argmax(density1)], c='r')
ax[1,1].plot(density2, label='density')
ax[1,1].axvline(np.argmax(density2), c='r')
ax_twin = ax[1,1].twinx()
ax_twin.plot(cdf2, label='cumulative density', color='g')
ax_twin.axhline(cdf2[np.argmax(density2)], c='r')
# Plot trace with estimated baselines
ax[0,2].plot(np.arange(5000,5800)/fr, F[cell, 5000:5800], label='raw trace')
ax[0,2].axhline(data_prct1, c='r', label='quantile ({:.1f})'.format(data_prct1))
ax[1,2].plot(np.arange(11100,11900)/fr, F[153, 11100:11900], label='raw trace')
ax[1,2].axhline(data_prct2, c='r', label='quantile ({:.1f})'.format(data_prct2))
ax[0,2].legend()
ax[1,2].legend()
ax[0,2].set_title('Trace with estimated baseline', fontsize=15)
# Formatting
ax[0,0].set_xlabel('time [s]', fontsize=15)
ax[1,0].set_xlabel('time [s]', fontsize=15)
ax[0,2].set_xlabel('time [s]', fontsize=15)
ax[1,2].set_xlabel('time [s]', fontsize=15)


#%% Alternative dF/F function (using the whole movie)
Yr, dims, T = cm.load_memmap(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200625\memmap__d1_512_d2_472_d3_1_order_C_frames_29058_.mmap')
bl = cnm.estimates.bl
quantileMin = 8
frames_window = 1000
block_size = 400

nA = np.array(np.sqrt(A.power(2).sum(0)).T)
A_1 = scipy.sparse.coo_matrix(A / nA.T)
C_1 = C * nA
bl = (bl * nA.T).squeeze()
nA = np.array(np.sqrt(A_1.power(2).sum(0)).T)

dview_res = dview
AY = cm.mmapping.parallel_dot_product(Yr, A_1, dview=dview_res, block_size=block_size, transpose=True).T

bas_val = bl[None, :]
Bas = np.repeat(bas_val, T, 0).T
AA = A_1.T.dot(A_1)
AA.setdiag(0)
Cf = (C_1 - Bas) * (nA ** 2)
C2 = AY - AA.dot(C_1)

# Plot C, Cf and C2
fig, ax = plt.subplots(2, 4)
ax[0,0].plot(np.arange(len(C[cell]))/fr, C[cell])
ax[0,0].set_title('denoised (inferred) temporal trace C')
ax[0,0].axvline(5000/fr, color='r')
ax[0,0].axvline(5800/fr, color='r')
ax[0,1].plot(np.arange(len(YrA[cell]))/fr, Cf[cell])
ax[0,1].set_title('baseline-corrected temporal trace Cf')
ax[0,1].axvline(5000/fr, color='r')
ax[0,1].axvline(5800/fr, color='r')
ax[0,2].plot(np.arange(len(C[cell]))/fr, C2[cell])
# ax[0,3].plot(np.arange(len(C[cell]))/fr, C2[cell]-6000)
ax[0,2].set_title('C2 = AY () - AA.dot(C)')
ax[0,2].axvline(5000/fr, color='r')
ax[0,2].axvline(5800/fr, color='r')
ax[0,3].plot(np.arange(len(C[cell]))/fr, C[cell]+YrA[cell])
ax[0,3].set_title('C + residuals')
ax[0,3].axvline(5000/fr, color='r')
ax[0,3].axvline(5800/fr, color='r')
ax[1,0].plot(np.arange(5000,5800)/fr, C[cell, 5000:5800])
# ax[1,0].set_title('estimates.F_dff zoom')
ax[1,1].plot(np.arange(5000,5800)/fr, Cf[cell, 5000:5800])
# ax[1,1].set_title('estimates.C zoom')
ax[1,2].plot(np.arange(5000,5800)/fr, C2[cell, 5000:5800])
# ax[1,3].plot(np.arange(5000,5800)/fr, C2[cell, 5000:5800]-6000)
# ax[1,2].set_title('estimates.S zoom')
ax[1,3].plot(np.arange(5000,5800)/fr, C[cell, 5000:5800]+YrA[cell, 5000:5800])
ax[1,0].set_xlabel('time [s]', fontsize=15)
ax[1,1].set_xlabel('time [s]', fontsize=15)
ax[1,2].set_xlabel('time [s]', fontsize=15)
ax[1,3].set_xlabel('time [s]', fontsize=15)

ax[0,0].tick_params(axis='both', which='major', labelsize=15)
ax[0,0].axhline(0, linestyle='--', color='grey')
ax[1,0].tick_params(axis='both', which='major', labelsize=15)
ax[1,0].axhline(0, linestyle='--', color='grey')
ax[0,1].tick_params(axis='both', which='major', labelsize=15)
ax[0,1].axhline(0, linestyle='--', color='grey')
ax[1,1].tick_params(axis='both', which='major', labelsize=15)
ax[1,1].axhline(0, linestyle='--', color='grey')
ax[0,2].tick_params(axis='both', which='major', labelsize=15)
ax[1,2].tick_params(axis='both', which='major', labelsize=15)
ax[0,3].tick_params(axis='both', which='major', labelsize=15)
ax[0,3].axhline(0, linestyle='--', color='grey')
ax[1,3].tick_params(axis='both', which='major', labelsize=15)
ax[1,3].axhline(0, linestyle='--', color='grey')

#%% dF/F with both methods calculation with whole movie
Df_1 = scipy.ndimage.percentile_filter(C2, quantileMin, (frames_window, 1))
Df_1_rev = scipy.ndimage.percentile_filter(C2, quantileMin, (1, frames_window))

C_df = Cf / Df_1
C_df_rev = Cf / Df_1_rev

#%% Plot second method dFF
s1 = 5000
s2 = 5800
fig, ax = plt.subplots(2, 4)
ax[0,0].set_title('Fluorescence Cf with background', fontsize=18)
ax[0,0].plot(np.arange(len(F[cell]))/fr, Cf[cell], label='Cf')
ax[0,0].plot(np.arange(len(f[0]))/fr, Df_1[cell], label='Base_cells', linewidth=2)
ax[0,0].plot(np.arange(len(f[0]))/fr, Df_1_rev[cell], label='Base_time', linewidth=2)
z = ax[0,0].legend()
ax[1,0].plot(np.arange(s1,s2)/fr, Cf[cell, s1:s2], label='Cf')
ax[1,0].plot(np.arange(s1,s2)/fr, Df_1[cell, s1:s2], label='Base_cells', linewidth=2)
ax[1,0].plot(np.arange(s1,s2)/fr, Df_1_rev[cell, s1:s2], label='Base_time', linewidth=2)
ax[1,0].legend()

# Get color of baselines
cols = z.get_lines()

ax[0,1].set_title('dF/F with Base_cells', fontsize=18)
ax[0,1].plot(np.arange(len(C_df[cell]))/fr, C_df[cell], color=cols[1].get_color())
ax[1,1].plot(np.arange(s1,s2)/fr, C_df[cell, s1:s2], color=cols[1].get_color())
ax[0,1].axvline(s1/fr, color='r')
ax[0,1].axvline(s2/fr, color='r')

ax[0,2].set_title('dF/F with Base_time', fontsize=18)
ax[0,2].plot(np.arange(len(dFF_time[cell]))/fr, C_df_rev[cell], color=cols[2].get_color())
ax[0,2].axvline(s1/fr, color='r')
ax[0,2].axvline(s2/fr, color='r')
ax[1,2].plot(np.arange(s1,s2)/fr, C_df_rev[cell, s1:s2], color=cols[2].get_color())

ax[0,3].set_title('dF/F with F_base_time', fontsize=18)
ax[0,3].plot(np.arange(len(dFF_auto[cell]))/fr, dFF_time[cell])
ax[0,3].axvline(s1/fr, color='r')
ax[0,3].axvline(s2/fr, color='r')
ax[1,3].plot(np.arange(s1,s2)/fr, dFF_time[cell, s1:s2])

# set x label
ax[1,0].set_xlabel('time [s]', fontsize=15)
ax[1,1].set_xlabel('time [s]', fontsize=15)
ax[1,2].set_xlabel('time [s]', fontsize=15)
ax[1,3].set_xlabel('time [s]', fontsize=15)


ax[1,1].set_ylim(-0.2, 3.6)
ax[1,2].set_ylim(-0.2, 3.6)
ax[1,3].set_ylim(-0.2, 3.6)
ax[0,1].set_ylim(-0.2, 3.6)
ax[0,2].set_ylim(-0.2, 3.6)
ax[0,3].set_ylim(-0.2, 3.6)

for idx, axis in enumerate(ax.ravel()):
    axis.tick_params(axis='both', which='major', labelsize=15)
    if not idx == 4:
        axis.axhline(0, linestyle='--', color='grey')

