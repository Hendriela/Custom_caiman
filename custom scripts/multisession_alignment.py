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
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.sca(ax[0])


fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
#ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)

ax1.imshow(shift_map_y, cmap='viridis')
# ax1.set_xlim(200, 300)
# ax1.set_ylim(200, 300)
# ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(shift_map_y_big, cmap='viridis')
# ax2.set_xlim(200, 300)
# ax2.set_ylim(200, 300)
# ax2.set_axis_off()
ax2.set_title(f'{all_shifts[0]}')

ax3.imshow(pcf_objects[2].cnmf.estimates.Cn, cmap='gray')
# ax2.set_xlim(200, 300)
# ax2.set_ylim(200, 300)
# ax3.set_axis_off()
ax3.set_title(f'{all_shifts[1]}')



n_rows = 4
n_cols = ceil((18 + 1) / n_rows)


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
assignments_filtered = [5]
n_reg = 3
traces = np.zeros(len(assignments_filtered))
for i in range(len(assignments_filtered)):
    traces[i] = np.zeros(n_reg)
    for j in range(n_reg):
        traces[i][j] = estimate_list[i].F_dff

test = np.zeros((4,6), dtype=np.ndarray)

for i in range(test.shape[0]):
    for j in range(test.shape[1]):
        if assignments[i,j] is not NaN:
            test[i,j] = cnm2.estimates.C[int(assignments[i,j])]

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# given values
xi = np.array([0.2, 0.5, 0.7, 0.9])
yi = np.array([0.3, -0.1, 0.2, 0.1])
# positions to inter/extrapolate
x = np.linspace(0, 1, 50)
# spline order: 1 linear, 2 quadratic, 3 cubic ...
order = 1
# do inter/extrapolation
s = InterpolatedUnivariateSpline(xi, yi, k=order)
y = s(x)

# example showing the interpolation for linear, quadratic and cubic interpolation
plt.figure()
plt.plot(xi, yi)
for order in range(1, 4):
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)
    plt.plot(x, y)
plt.show()