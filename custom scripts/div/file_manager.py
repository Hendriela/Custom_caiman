import os
import platform
import ctypes
from glob import glob
import shutil
from pathlib import Path
from standard_pipeline.behavior_import import progress
import tifffile as tif
import numpy as np
from skimage import registration as reg
from util import motion_correction
import itertools


def get_free_space_gb(dirname):
    """Return folder/drive free space (in gigabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024 / 1024


def transfer_raw_movies(source, target, basename='file', restore=False, move_all=False, force_transfer=False):
    """
    Transfers all raw TIFF movies in the source directory and its subdirectories to the target directory while
    maintaining the source directory structure and creating new directories in the target if necessary.
    :param source: str, parent directory from where TIFF files should be taken
    :param target: str, target directory where TIFF files should be moved to
    :param basename: str, basename (before the underscore) of the TIFF files that should be transferred; default 'file'
    :param restore: bool flag whether the function is used to recover raw files back to the working directory. If True,
                        no PCF object has to be in the parent directory of transfer TIFF files. If False, raw files in
                        the source directory are deleted after successful transfer to make room on the server.
    :param move_all: bool flag whether TIF files should be moved, even if no PCF file is in the session folder.
    :param force_transfer: bool flag whether to start transfer even with insufficient disk space.
    :return:
    """
    tif_list = []
    # Find movie files that are part of a completely analysed session (PCF file exists)
    use_mem = 0
    for step in os.walk(source):
        tif_files = glob(step[0] + f'\\{basename}_00???.tif')
        if len(tif_files) > 0:
            if (len(glob(step[0].rsplit(os.sep, 1)[0] + r'\\pcf*')) > 0) or restore or move_all:
                for fname in tif_files:
                    use_mem += os.path.getsize(fname) / 1073741824
                    tif_list.append(fname[len(source):])  # remove source directory to get relative path

    if use_mem > get_free_space_gb(target) and not force_transfer:
        raise MemoryError(
            f'Not enough space on {target}! {get_free_space_gb(target)} GB available, {use_mem} GB needed.')

    answer = input(
        f'Found {len(tif_list)} files.\nThey would free up {int(use_mem * 100) / 100} GB. Do you want to transfer them? [y/n]')
    if answer == "yes" or answer == 'y':
        for idx, file in enumerate(tif_list):
            # Check the path to the current TIFF file and create any non-existing directories
            if len(tif_list) > 1:
                progress(idx, len(tif_list) - 1,
                         status=f'Transferring files to target directory...({idx + 1}/{len(tif_list)})')
            targ = Path(target + file)
            for parent in targ.parents.__reversed__():
                if not os.path.isdir(parent):
                    os.mkdir(parent)

            # Transfer the file to the new location if it does not exist already there
            if not os.path.isfile(target + file):
                shutil.copy(src=source + file, dst=target + file)
            else:
                print('File {} already exists in target directory, copying skipped.'.format(file))
            # If the files were transferred from the server to the backup hard drive, delete the files from the server after
            # successful transfer.
        if not restore:
            print('Removing files at the source directory...')
            [os.remove(source + fname) for fname in tif_list]
        print('\nDone!')
    elif answer == "no" or answer == 'n':
        print('Transfer cancelled.')
    else:
        print("Please enter yes or no.")


def remove_mmap_after_analysis(root, basename='memmap__', remove_all=False):
    """
    Removes mmap files in all subdirectories that are completely analysed (PCF file exists in the same directory).
    :param root: parent directory from whose subdirectories mmap files should be removed
    :param remove_all: bool flag whether mmap files should be moved, even if no PCF file is in the session folder.
    :return:
    """
    free_mem = 0
    n_files = 0

    del_files = []
    for step in os.walk(root):
        mmap_file = glob(step[0] + f'\\{basename}*')
        ana_file = glob(step[0] + '\\pcf_result*')
        if len(mmap_file) > 0 and (len(ana_file) > 0 or remove_all):
            for file in mmap_file:
                free_mem += os.path.getsize(file) / 1073741824  # Remember size of deleted file for later report
                n_files += 1
                del_files.append(file)

    if len(del_files) == 0:
        print('No mmap files found!')
    else:
        if remove_all:
            print(
                f'WARNING! remove_all = True, which means that also half-processed session`s .{basename} will be deleted!\n')
        print(f'Found {len(del_files)} files to be deleted:')
        if len(del_files) < 50:
            print(*del_files, sep='\n')
        else:
            print(*del_files[:49], sep='\n')
            print(f'... + {len(del_files) - 49} more files')

        answer = None
        while answer not in ("y", "n", 'yes', 'no'):
            answer = input(
                f'These files would free up {int(free_mem * 100) / 100} GB. Do you want to delete them? [y/n]')
            if answer == "yes" or answer == 'y':
                print('Deleting...')
                for file in del_files:
                    os.remove(file)
                print('Done!')
            elif answer == "no" or answer == 'n':
                print('Deleting cancelled.')
            else:
                print("Please enter yes or no.")


def clean_cnm_files(dir=r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data",
                    fnames=("cnm_pre_selection", "cnm_results", "pcf_results")):
    """
    Removes files (in order of fnames) if the next file exists in the directory
    :param dir: str, root directory from whose subdirectories files will be deleted
    :param fnames: tuple, filenames of files that should be removed (in ascending order)
    :return: int, amount of disk space that was removed
    """
    free_mem = 0
    n_files = 0

    del_files = []
    for step in os.walk(dir):
        first_files = glob(step[0] + f'\\{fnames[0]}*')
        if len(first_files) > 0:
            second_files = glob(step[0] + f'\\{fnames[1]}*')
            if len(second_files) > 0:
                for file in first_files:
                    free_mem += os.path.getsize(file) / 1073741824
                    n_files += 1
                    del_files.append(file)
                third_files = glob(step[0] + f'\\{fnames[2]}*')
                if len(third_files) > 0:
                    for file in second_files:
                        free_mem += os.path.getsize(file) / 1073741824
                        n_files += 1
                        del_files.append(file)

    print(f'Found {len(del_files)} files to be deleted:')
    print(*del_files, sep='\n')
    answer = None
    while answer not in ("y", "n", 'yes', 'no'):
        answer = input(f'These files would free up {int(free_mem * 100) / 100} GB. Do you want to delete them? [y/n]')
        if answer == "yes" or answer == 'y':
            print('Deleting...')
            for file in del_files:
                os.remove(file)
            print('Done!')
        elif answer == "no" or answer == 'n':
            print('Deleting cancelled.')
        else:
            print("Please enter yes or no.")


def average_overviews(root_dir: str, name_pattern: str = 'overview', dist_thresh: int = 5) -> None:
    """
    Make average from overview TIFFs, while excluding movement-corrupted frames. Delete original stack after
    saving average TIFF

    Args:
        root_dir    : Start directory from where TIFFs are searched for recursively.
        name_pattern: Substring that has to be in the TIFF filename to be averaged.
        dist_thresh : Maximum geometric shift distance (from phase cross correlation) between neighbouring frames

    (c) Celine Heeb
    """

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in [f for f in filenames if (name_pattern in f) and ("AVG" not in f) and ("stack" not in f)]:
            filepath = os.path.join(dirpath, filename)
            print(f'Processing file {filepath}')
            a = tif.imread(filepath)  # a -> array von image (a[0]= erster frame)

            # Adjust line shift
            a_corr = motion_correction.correct_line_shift_stack(stack=a, crop_left=0, crop_right=0,
                                                                nr_samples=a.shape[0])

            ad = np.zeros((a_corr.shape[0], a_corr.shape[0]))  # array with movement distance
            indices = np.arange(a_corr.shape[0])
            corr_pairs = [comb for comb in itertools.combinations(indices, 2)]
            for pair in corr_pairs:
                shift, _, _ = reg.phase_cross_correlation(a_corr[pair[0]], a_corr[pair[1]])
                dist = np.sqrt((shift[0]) ** 2 + (shift[1]) ** 2)
                ad[pair[0], pair[1]] = dist
                ad[pair[1], pair[0]] = dist

            ave = np.average(ad, axis=0)
            lst = []
            while len(lst) < 20:
                for idx, avg_dist in enumerate(ave):
                    if avg_dist <= dist_thresh:
                        lst.append(a_corr[idx])
                dist_thresh += 1
            s = np.stack(lst)
            m = np.mean(s, axis=0)  # If dtype is changed already here, we get overflows
            m = m.astype('int16')
            tif.imwrite(os.path.join(dirpath, "AVG_" + filename.replace("_00001", "")), m)
            # Delete original file after saving the average
            os.remove(os.path.join(dirpath, filename))


def get_next_filename(fname, target=None, suffix='_copy', start_with_zero=False):
    """
    Looks for and returns the next available filename to avoid overwriting.
    :param fname: str, filename of original file.
    :param target: str, optional. If given, filename will be in the target rather than the original directory.
    :param suffix: str, suffix to append to the original filename. Default is '_copy'.
    :param start_with_zero: bool flag whether to start the numbering with 0 or 1.
    :return:
    """
    if target is not None:
        mod_fname = os.path.join(target, os.path.splitext(fname)[0].split(os.path.sep)[-1]) + suffix + '%s' + \
                    os.path.splitext(fname)[1]
    else:
        mod_fname = os.path.splitext(fname)[0] + suffix + '%s' + os.path.splitext(fname)[1]
    if start_with_zero:
        i = 0
    else:
        i = 1
    while os.path.exists(mod_fname % i):
        i += 1
    return os.path.splitext(mod_fname)[0][:-2] + str(i) + os.path.splitext(fname)[1]
