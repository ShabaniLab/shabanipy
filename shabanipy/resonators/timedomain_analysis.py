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
from scipy.signal import windows, oaconvolve
import matplotlib.pyplot as plt
from shabanipy.plotting import jy_pink
jy_pink.register()

def plot_histogram_iq(
    i:np.ndarray,
    q:np.ndarray,
    bins:np.ndarray,
    counts_lims: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
)-> None:

    fig, ax = plt.subplots(figsize = (14,10),constrained_layout=True, dpi = 50)
    hist = ax.hist2d(i,q, bins = bins, 
                vmin = counts_lims[0] if counts_lims else 0,
                vmax = counts_lims[1] if counts_lims else None)
    ax.set_xlabel('I (mV)')
    ax.set_ylabel('Q (mV)')
    cb = fig.colorbar(hist[3], ax=ax)

def gaussian_convolution(data,avgTime,sampleRate):

    '''Smooths data by convolving with a gaussian window of duration avgTime

        and standard deviation avgTime/7.

        returns smoothed data with original shape.

        -------------------------------------

        data:       single or dual channel data. dual channel should have shape (nSamples,2)

        avgTime:    duration of Hann window in seconds

        sampleRate: sample rate of data in Hz.

    '''

    nAvg = int(max(avgTime*sampleRate,1))

    window = windows.gaussian(nAvg,nAvg/7)

    norm = sum(window)

    if len(data.shape) == 2:

        mean = np.mean(data,axis=1)

        window = np.vstack((window,window))

        return (oaconvolve((data.T-mean).T,window,mode='same',axes=1).T/norm + mean).T

    else:

        mean = np.mean(data)

        return oaconvolve(data-mean,window,mode='same')/norm + mean