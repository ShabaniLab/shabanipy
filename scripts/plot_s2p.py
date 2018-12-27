# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018 by Shabanipy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Simple script to plot the S parameters from a s2p file.

We only plot the amplitudes in dB.

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

#: Directory in which the data are located.
DIRECTORY = '/Users/mdartiailh/Documents/PostDocNYU/Fridge/EcosorbRFFilters'

#: Touchstone file to plot
FILENAME = 'ECOSORB_8.S2P'

#: Should the plot be saved (same directory and name)
SAVE_PLOT = True

#: Format under which to save the plot.
SAVE_PLOT_FORMAT = 'png'


if __name__ == '__main__':

    data = pd.read_csv(os.path.join(DIRECTORY,  FILENAME),
                       skiprows=5,
                       sep='\t',
                       names=['Freq',
                              'S11_dB', 'S11_deg',
                              'S21_dB', 'S21_deg',
                              'S12_dB', 'S12_deg',
                              'S22_dB', 'S22_deg'])

    f, axarr = plt.subplots(1, 2)
    f.suptitle(FILENAME)
    axarr[0].plot(data['Freq']/1e9, data['S11_dB'], label='S11')
    axarr[0].plot(data['Freq']/1e9, data['S22_dB'], label='S22')
    axarr[0].set_xlabel('Frequency (GHz)')
    axarr[0].set_ylabel('Reflection (dB)')
    axarr[0].legend()
    axarr[1].plot(data['Freq']/1e9, data['S21_dB'], label='S21')
    axarr[1].plot(data['Freq']/1e9, data['S12_dB'], label='S12')
    axarr[1].set_xlabel('Frequency (GHz)')
    axarr[1].set_ylabel('Transmission (dB)')
    axarr[1].legend()

    if SAVE_PLOT:
        plt.savefig(os.path.join(DIRECTORY,
                                 FILENAME[:-3] + SAVE_PLOT_FORMAT))

    plt.show()
