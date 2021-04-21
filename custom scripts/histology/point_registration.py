from glob import glob
import os
import re
import numpy as np
import tifffile as tiff

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\histology\Batch3_quant\annotation_results"


def fix_order(point_path):
    return

def load_points(point_path):
    """
    Load coordinates of points from QuPath output TSV files and returns it as a list of ndarrays
    :param point_path: str, path of the directory where TSV files are stored
    :return: list of ndarrays (one per file) with shape (#points, 2), first column being x, second column y coordinate.
    """

    # Get correct order from separate file (if scenes were imaged out-of-order) or from file numbering
    # try:
    #     with open(os.path.join(point_path, "order.txt")) as f:
    #         file_list = f.readlines()
    #         file_list = [x.strip() for x in file_list]
    # except:
    file_list = glob(os.path.join(point_path, "*.tsv"))
    file_list.sort(key=natural_keys)

    coords = []
    for file in file_list:
        out = np.loadtxt(file, delimiter="\t", skiprows=1, usecols=(0,1))      # Load TSV file
        if out.ndim == 1:
            out = np.reshape(out, (1, out.shape[0]))                           # Reshape array in case of only one entry
        coords.append(out)

    return coords


def load_ome(im_path):

    file_list = glob(os.path.join(im_path, "*ome.tiff"))


def make_point_mask(path):

    # Load TSV files (QuPath output) of point coordinates
    coords = load_points(path)

    # Load single-channel OME TIFFs that should be extended by the new binary channel



