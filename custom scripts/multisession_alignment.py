import pickle
import caiman as cm
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import imageio
#%% load pickled file (spatial components + mean intensity file)

infile = open('/Users/hheiser/Desktop/testing data/chronic_test/Sample data/alignment.pickle','rb')
data = pickle.load(infile)
infile.close()

spatial = data[0]
templates = data[1]

dims = templates[0].shape

spatial_union, assignments, matchings = register_multisession(spatial, dims=dims, templates=templates)

n_sess = len(spatial)
fig, ax = plt.subplots(nrows=2, ncols=3)
plt.sca(ax[0,0])
visualization.plot_contours(spatial[0], templates[0])
plt.sca(ax[0,1])
visualization.plot_contours(spatial[1], templates[1])
plt.sca(ax[0,2])
visualization.plot_contours(spatial[2], templates[2])
plt.sca(ax[1,0])
visualization.plot_contours(spatial[3], templates[3])
plt.sca(ax[1,1])
visualization.plot_contours(spatial[4], templates[4])
plt.sca(ax[1,2])
visualization.plot_contours(spatial[5], templates[5])
fig.tight_layout()
fig.show()

mean = np.mean((templates[0],templates[1]))

plt.figure()
visualization.plot_contours(spatial_union, templates[0])
plt.show()

#%% filter components by number of registrated sessions
n_reg = 6
assignments_filtered = np.array(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg], dtype=int)
spatial_filtered = spatial[0][:, assignments_filtered[:, 0]]
visualization.plot_contours(spatial_filtered, templates[0])


#%%

dir_path = '/Users/hheiser/Desktop/testing data/test'
file_ext = '.png'

file_path = dir_path + '/*' + file_ext

file_list = glob.glob(dir_path+'/*'+file_ext)

for i in range(len(file_list)):
    temp = np.asarray(imageio.imread(file_list[i]), dtype=bool)
    if i == 0:
        A = np.zeros((np.prod(temp.shape), len(file_list)), dtype=bool)
    if expand_method == 'dilation':
        temp = dilation(temp, selem=selem)
    elif expand_method == 'closing':
        temp = dilation(temp, selem=selem)

    A[:, i] = temp.flatten('F')

#%%
assignments_filtered = [5
n_reg = 3
traces = np.zeros(len(assignments_filtered))
for i in range(len(assignments_filtered):
    traces[i] = np.zeros(n_reg)
    for j in range(n_reg):
        traces[i][j] = estimate_list[i].F_dff

test = np.zeros((4,6), dtype=np.ndarray)

for i in range(test.shape[0]):
    for j in range(test.shape[1]):
        if assignments[i,j] is not NaN:
            test[i,j] = cnm2.estimates.C[int(assignments[i,j])]