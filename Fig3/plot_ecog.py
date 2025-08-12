#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:46:42 2023

@author: sebastian
"""

import scipy as sp
import matplotlib
from nilearn import plotting
import numpy as np
import os
infilePath = '/CRM_project/Podcast/ds005574/derivatives/crm_canon_prev_word_10fold/'



pos = sp.io.loadmat(infilePath+ '/all_epos.mat')['all_epos']
vals1 =  sp.io.loadmat(infilePath+ '/all_topos_cca50Fin.mat')['all_topos_cca']
vals2 =  sp.io.loadmat(infilePath+ '/all_topos_crm50Fin.mat')['all_topos_crm']


# Filter values and positions for vals1
mask1 = vals1 > 0.1
indices = np.where(mask1)[0]

filtered_vals1 = vals1[mask1]
filtered_pos1 = pos[indices, :]


# Filter values and positions for vals2
mask2 = vals2 > 0.1
filtered_vals2 = vals2[mask2]
indices = np.where(mask2)[0]
filtered_pos2 = pos[indices, :]
cmap = matplotlib.pyplot.get_cmap('YlOrRd')

# Plot markers for filtered vals1 and vals2
plotting.plot_markers(filtered_vals1, filtered_pos1, node_size=12, node_cmap=cmap, node_vmin=0.1, node_vmax=1, alpha=1);
matplotlib.pyplot.savefig(os.path.join(infilePath, 'cca.png'), dpi=600)

plotting.plot_markers(filtered_vals2, filtered_pos2, node_size=12, node_cmap=cmap, node_vmin=0.1, node_vmax=1, alpha=1)
matplotlib.pyplot.savefig(os.path.join(infilePath, 'crm.png'), dpi=600)

# Define blue-white-red colormap
cmap = matplotlib.pyplot.get_cmap('bwr')

# Filter values and positions for the difference between vals2 and vals1
vals_diff = vals2 - vals1
mask_diff = (vals_diff > 0.05) | (vals_diff < -0.05)
filtered_vals_diff = vals_diff[mask_diff]
indices = np.where(mask_diff)[0]

filtered_pos_diff = pos[indices, :]

# Plot markers for the filtered difference between vals2 and vals1
plotting.plot_markers(filtered_vals_diff, filtered_pos_diff, node_size=12, node_cmap=cmap, node_vmin=-0.4, node_vmax=0.4, alpha=1)
matplotlib.pyplot.savefig(os.path.join(infilePath, 'crm_min_cca.png'), dpi=600)

