from pandas import DataFrame
import pandas as pd
import csv

fname = r"W:\Neurophysiology-Storage1\Wahl\Jithin\6_20200305T130305_StageP1b_pull_NoStim.tsv"

file = []
with open(fname) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        file.append(line)

df = pd.read_csv(r"W:\Neurophysiology-Storage1\Wahl\Jithin\6_20200305T130305_StageP1b_pull_NoStim.tsv", sep="\t")

df = pd.DataFrame(data=file[16:], columns=file[15])
