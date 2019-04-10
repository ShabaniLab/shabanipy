# -*- coding: utf-8 -*-
"""Generate a color plot of a shapiro step experiment through binning.

The plot use the folling axes:
- x axis: Power
- y axis: Normalized voltage
- color axis: bin count in current

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/02/Data_0205/JS124S_BM002_171.hdf5'

#: Name or index of the column containing the frequency data if applicable.
#: Leave blanck if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = 2

#: Frequencies of the applied microwave in Hz (one graph will be generated for
#: each frequecy).
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [15e9]

#: Name or index of the column containing the power data
POWER_NAME = 1

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = -1

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CURRENT_NAME = 0

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.1

#: Critical power at which we close the gap.
CRITICAL_POWER = 15.8

#: Label of the x axis, if left blanck the power column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Power (dB)'

#: Label of the y axis.
Y_AXIS_LABEL = 'Junction voltage (hf/2e)'

#: Label of the colorbar.
C_AXIS_LABEL = 'Counts (Ic)'

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1

#: Number of points of the lowest available power to use to correct the
#: voltage offset.
Y_OFFSET_CORRECTION = 50

#: Scaling factor for the c axis (used to convert between units)
C_SCALING = 1e6/4.3

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [0, 6]

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [0, 0.05]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = [1, 2, 3, 4, 5]

#: Power range in which to plot the dashed lines for shapiro steps. Use None
#: to indicate that the line should start/stop at the edge.
SHAPIRO_STEPS_POWERS = [0, 0.1]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.shapiro.binning import bin_power_shapiro_steps
from shabanipy.utils.labber_io import LabberData

for frequency in FREQUENCIES:
    print(f'Treating data for frequency {frequency/1e9} GHz')
    with LabberData(PATH) as data:

        filters = {}
        if FREQUENCY_NAME is not None:
            filters[FREQUENCY_NAME] = frequency

        shape = data.compute_shape((POWER_NAME, CURRENT_NAME))

        power = data.get_data(POWER_NAME, filters=filters)
        volt = data.get_data(VOLTAGE_NAME, filters=filters)
        curr = data.get_data(CURRENT_NAME, filters=filters)

        # Handle interruptions in the last scan.
        while len(power) < shape[0]*shape[1]:
            shape[1] -= 1

        length = shape[0]*shape[1]
        power = power[:length].reshape(shape)
        volt = volt[:length].reshape(shape)
        curr = curr[:length].reshape(shape)

    # Convert the current data if requested
    if CURRENT_CONVERSION is not None:
        curr *= CURRENT_CONVERSION

    if Y_OFFSET_CORRECTION:
        offset = np.average(volt[0, :Y_OFFSET_CORRECTION])
        volt -= offset

    # Bin the data
    power, voltage, histo = bin_power_shapiro_steps(power, curr, volt,
                                                    frequency, STEP_FRACTION)

    power -= CRITICAL_POWER

    # Plot the data
    plt.figure()
    plt.imshow(C_SCALING * histo.T,
               extent=(power[0], power[-1], voltage[0], voltage[-1]),
               origin='lower',
               aspect='auto',
               vmin=C_LIMITS[0],
               vmax=C_LIMITS[1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(C_AXIS_LABEL)

    if SHOW_SHAPIRO_STEP:
        steps = [n * Y_SCALING
                 for n in SHOW_SHAPIRO_STEP]
        if not X_LIMITS:
            X_LIMITS = [None, None]
        lims = [x or X_LIMITS[i] or power[-i]
                for i, x in enumerate(SHAPIRO_STEPS_POWERS)]
        plt.hlines(steps, *lims, linestyles='dashed')

    sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
    plt.title(f'Frequency {frequency/1e9} GHz')
    plt.xlabel(X_AXIS_LABEL or POWER_NAME)
    plt.ylabel(Y_AXIS_LABEL)
    if X_LIMITS:
        plt.xlim(X_LIMITS)
    if Y_LIMITS:
        plt.ylim(Y_LIMITS)
    plt.tight_layout()

plt.show()
