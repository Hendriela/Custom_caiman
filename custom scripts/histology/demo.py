import tifffile as tiff
from PIL import Image as im
import os
import aicspylibczi
import numpy as np
import matplotlib.pyplot as plt

def prepare_stack_for_quicknii(path, output, step, start=0):
    """
    Load full brain stack from Fiji macro "assemble_full_brain" and prepare it for QuickNII
    (save series as png).
    :param path: file path of the stack
    :param step: step size of the slices, every n-th slice has been imaged
    :return:
    """

    # Load stack
    stack = tiff.imread(path)

    # Get file name
    fname = os.path.splitext(os.path.basename(path))[0]

    # Save each stack separately
    for i in range(len(stack)):
        img = im.fromarray(stack[i])
        img.save(os.path.join(output, fname+"_s{:03d}.png".format(start)))
        start += step

path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Tests\export_test\output\M41\M41__full_brain_20.852um_per_px.tif"
output = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Tests\export_test\quicknii_output"
step = 3

prepare_stack_for_quicknii(path, output, step)


#%% Trying to open multiscene mosaic CZI files

def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0])/(norms[1]-norms[0]), 0, 1)
    return i2

path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\histology\Batch3_quant\images\M41_01.czi"
czi = aicspylibczi.CziFile(path)

mosaic_data = czi.read_mosaic(C=2, scale_factor=0.1)
normed_mosaic_data = norm_by(mosaic_data[0, :, :], 5, 98)*255
plt.imshow(normed_mosaic_data)

czi.read_subblock_rect(S=1, M=100)

#%% Open OME TIFF

path = r"Z:\data\h.heiser\slidescanner_transfer_20210209_0901\M41_01.ome.tiff"
path = r"Z:\data\h.heiser\slidescanner_transfer_20210209_0901\M41_01_DAPI_8bit.ome.tiff"
ome = tiff.imread(path)
plt.imshow(ome)


#%% Try to read .flat file from QuickNII to get atlas identifier data
import struct

path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Tests\export_test\quicknii_output\M41__full_brain_20.852um_per_px_s000-Rainbow_2017.flat"

with open(path, "rb") as f:
    bytes_read = f.read()

unpack_result = struct.unpack('<II', bytes_read[1:9])

for i in range(len(bytes_read)):
    if i < 100:
        print(bytes_read[i])
