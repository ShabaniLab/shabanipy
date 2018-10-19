# -*- coding: utf-8 -*-
"""Generate a histogram of the voltage for a given frequency and power.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r'/Users/mdartiailh/Labber/Data/2018/09/Data_0913/JS124L_CD004_009.hdf5'

#: Name or index of the column containing the frequency data if applicable.
#: Set to None if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = 2

#: Frequency of the applied microwave in Hz.
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [6e9, 12e9]

#: Name or index of the column containing the power data
POWER_NAME = 1

#: Powers in dBm at which to plot the V-I characteristic
POWERS = [3]

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 3

#: Name or index of the column containing the current data
CURRENT_NAME = 0

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.25

#: Label of the y axis.
X_AXIS_LABEL = 'Counts (µA)'

#: Label of the colorbar.
Y_AXIS_LABEL = 'Junction voltage (hf/2e)'

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1e6

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1

#: Limits to use for the x axis (after scaling)
X_LIMITS = [0, 1]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [-6, 6]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import matplotlib.pyplot as plt

from shabanipy.shapiro import shapiro_step
from shabanipy.shapiro.binning import bin_shapiro_steps, center_bin
from shabanipy.utils.labber_io import LabberData

for frequency in FREQUENCIES:
    print(f'Treating data for frequency {frequency/1e9} GHz')

    with LabberData(PATH) as data:

        filters = {POWER_NAME: 0}
        if FREQUENCY_NAME is not None:
            filters[FREQUENCY_NAME] = frequency

        for p in POWERS:
            plt.figure()
            filters[POWER_NAME] = p
            volt = data.get_data(VOLTAGE_NAME,
                                 filters=filters)
            curr = data.get_data(CURRENT_NAME,
                                 filters=filters)

            # Convert the current data if requested
            if CURRENT_CONVERSION is not None:
                curr *= CURRENT_CONVERSION

            c_step = abs(curr[1] - curr[0])
            counts, bins = bin_shapiro_steps(volt, frequency, STEP_FRACTION)

            # Scales counts in current and center bins
            bins = center_bin(bins)
            result = c_step*counts

            # Apply scaling factor before plotting
            bins = Y_SCALING*bins/shapiro_step(frequency)
            result *= X_SCALING

            # Plot the data
            plt.barh(bins, result, height=STEP_FRACTION)

            if SHOW_SHAPIRO_STEP:
                steps = [n for n in SHOW_SHAPIRO_STEP]
                lims = X_LIMITS or (bins[0], bins[-1])
                plt.hlines(steps, *lims, linestyles='dashed')

            sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
            plt.title(f'Sample {sample}:\n'
                      f'Frequency {frequency/1e9} GHz Power {p} dBm')
            plt.xlabel(X_AXIS_LABEL or CURRENT_NAME)
            plt.ylabel(Y_AXIS_LABEL or VOLTAGE_NAME)
            if X_LIMITS:
                plt.xlim(X_LIMITS)
            if Y_LIMITS:
                plt.ylim(Y_LIMITS)
            plt.tight_layout()

plt.show()
