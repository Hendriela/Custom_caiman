import os
from glob import glob
import shutil
from pathlib import Path
from standard_pipeline.place_cell_pipeline import progress


def transfer_raw_movies(source, target, basename='file', restore=False):
    """
    Transfers all raw TIFF movies in the source directory and its subdirectories to the target directory while
    maintaining the source directory structure and creating new directories in the target if necessary.
    :param source: str, parent directory from where TIFF files should be taken
    :param target: str, target directory where TIFF files should be moved to
    :param basename: str, basename (before the underscore) of the TIFF files that should be transferred; default 'file'
    :param restore: bool flag whether the function is used to recover raw files back to the working directory. If True,
                        no PCF object has to be in the parent directory of transfer TIFF files. If False, raw files in
                        the source directory are deleted after successful transfer to make room on the server.
    :return:
    """
    tif_list = []
    # Find movie files that are part of a completely analysed session (PCF file exists)
    for step in os.walk(source):
        if len(glob(step[0] + f'\\{basename}_00???.tif')) == 1:
            if len(glob(step[0].rsplit(os.sep, 1)[0] + r'\\pcf*')) > 0 or restore:
                tif_list.append(glob(step[0] + f'\\{basename}_00???.tif')[0][len(source):])

    for idx, file in enumerate(tif_list):
        # Check the path to the current TIFF file and create any non-existing directories
        progress(idx, len(tif_list)-1, status=f'Moving files to target directory...({idx+1}/{len(tif_list)})')
        targ = Path(target+file)
        for parent in targ.parents.__reversed__():
            if not os.path.isdir(parent):
                os.mkdir(parent)
        # Transfer the file to the new location
        shutil.copy(src=source+file, dst=target+file)
        # If the files were transferred from the server to the backup hard drive, delete the files from the server after
        # successful transfer.
    if not restore:
        [os.remove(source+fname) for fname in tif_list]
    print('\nDone!')


def remove_mmap_after_analysis(root):
    """
    Removes mmap files in all subdirectories that are completely analysed (PCF file exists in the same directory).
    :param root: parent directory from whose subdirectories mmap files should be removed
    :return:
    """
    free_mem = 0
    n_files = 0

    del_files = []
    for step in os.walk(root):
        mmap_file = glob(step[0]+'\\memmap__*')
        ana_file = glob(step[0] + '\\pcf_result*')
        if len(mmap_file) > 0 and len(ana_file) > 0:
            for file in mmap_file:
                free_mem += os.path.getsize(file)/1073741824    # Remember size of deleted file for later report
                n_files += 1
                del_files.append(file)

    print(f'Found {len(del_files)} files to be deleted:')
    print(*del_files, sep='\n')
    answer = None
    while answer not in ("y", "n", 'yes', 'no'):
        answer = input(f'These files would free up {int(free_mem*100)/100} GB. Do you want to delete them? [y/n]')
        if answer == "yes" or answer == 'y':
            print('Deleting...')
            for file in del_files:
                os.remove(file)
            print('Done!')
        elif answer == "no" or answer == 'n':
            print('Deleting cancelled.')
        else:
            print("Please enter yes or no.")
