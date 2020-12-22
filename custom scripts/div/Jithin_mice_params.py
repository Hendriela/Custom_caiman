fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds (0.4)
dxy = (0.83, 0.76)  # spatial resolution (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

# extraction parameters
p = 1  # order of the autoregressive system
gnb = 2  # number of global background components (3)
merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
K = 6  # number of components per patch (10)
gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 2  # spatial subsampling during initialization
tsub = 2  # temporal subsampling during intialization

# evaluation parameters
min_SNR = 10  # signal to noise ratio for accepting a component (default 2)
SNR_lowest = 3
rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
rval_lowest = -1
cnn_thr = 0.9  # threshold for CNN based classifier (default 0.99)
cnn_lowest = 0.05  # neurons with cnn probability lower than this value are rejected (default 0.1)