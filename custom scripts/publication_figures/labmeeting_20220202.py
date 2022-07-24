#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 21/01/2022 17:35
@author: hheise

"""

import standard_pipeline.place_cell_pipeline as pipe
import matplotlib.pyplot as plt
import numpy as np

# Place cell cross-session correlation examples
pcf = pipe.load_pcf(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200818")

pc1 = 867   # pre-stroke place cell
pc2 = 872   # high correlation
pc3 = 555   # still place cell, but low correlation
pc4 = 120   # no spatial preference
x = np.linspace(0, 400, 80)

plt.figure()
plt.plot(x, pcf.bin_avg_activity[pc1])
plt.xlabel("VR position", fontsize=20)
plt.ylabel("dF/F", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

fix, ax = plt.subplots(3,1, sharey="all", sharex="all")
ax[0].plot(x, pcf.bin_avg_activity[pc2])
ax[1].plot(x, pcf.bin_avg_activity[pc3])
ax[2].plot(x, pcf.bin_avg_activity[pc4])

for i in range(3):
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[i].set_ylabel("dF/F", fontsize=20)

plt.xlabel("VR position", fontsize=20)
plt.tight_layout()

