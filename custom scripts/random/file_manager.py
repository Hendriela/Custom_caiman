import os
from glob import glob

def backup_raw_movies(source, target):

    # Find
    for step in os.walk(source):
        if len(glob(step[0] + r'\\file_00???.tif')) == 1 and len(glob(step[0].rsplit(os.sep, 1)[0] + r'\\pcf*')) > 0:
            print(glob(step[0] + r'\\file_00???.tif'))
