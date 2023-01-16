from standard_pipeline import place_cell_pipeline as pipe
import matplotlib.pyplot as plt
from caiman.utils import visualization
import numpy as np

#%% Load CNM
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191121b\N1'

pcf = pipe.load_pcf(root, 'pcf_results_no_resting')

#%% Plot traces
n_idx = 10

fig, ax = plt.subplots(1, 3, sharex=True)
ax[0].plot(cnm.estimates.C[n_idx])
ax[0].set_title('"Raw"')
ax[1].plot(cnm.estimates.F_dff[n_idx])
ax[1].set_title('dF/F')
ax[2].plot(pcf.cnmf.estimates.spikes[n_idx])
ax[2].set_title('spikes')

#%% spike estimation
min_idx = 11400
max_idx = 11600
spike = np.sum(pcf.cnmf.estimates.spikes[n_idx, min_idx:max_idx])
max_freq = max(pcf.cnmf.estimates.spikes[n_idx, min_idx:max_idx]) / (1/30)
print(f'Estimated {spike} spikes in this transient with a max firing frequency of {max_freq} Hz.')

#%% Plot spatial info
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(np.reshape(cnm.estimates.A[:, n_idx].toarray(), (512, 512), order='F'), cmap='gray')
plt.sca(ax[1])
out = visualization.plot_contours(cnm.estimates.A[:, n_idx], cnm.estimates.Cn, display_numbers=False, colors='r')

#%% Plot binned traces

def print_msg(msg):
# This is the outer enclosing function

    def printer():
# This is the nested function
        new_msg = msg+'new'
        print(msg)

    printer()
    print(new_msg)