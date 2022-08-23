# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tools to analyse cavity in the notch geometry.

"""
from typing import Optional

import os
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.plotting import jy_pink
jy_pink.register()

def histogram_iq(
    i:np.ndarray,
    q:np.ndarray,
    bins:np.ndarray,
    counts_lims: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
)-> None:

    fig, ax = plt.subplots(figsize = (20,10),constrained_layout=True, dpi = 50)
    hist = ax.hist2d(i,q, bins = bins, 
                vmin = counts_lims[0] if counts_lims else 0,
                vmax = counts_lims[1] if counts_lims else None)
    ax.set_xlabel('I (mV)')
    ax.set_ylabel('Q (mV)')
    cb = fig.colorbar(hist[3], ax=ax)