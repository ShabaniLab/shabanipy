# -*- coding: utf-8 -*-
"""Generate a set linear plot (V-I) for different to powers

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/02/Data_0202/JS124S_BM002_150.hdf5'

#: Directory in which to save the figure.
FIG_DIRECTORY = '/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/Shapiro/2019-01/FrequencyDependence'

#: Name or index of the column containing the frequency data if applicable.
#: Set to None if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = None

#: Frequency of the applied microwave in Hz.
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [6e9]

#: Name or index of the column containing the power data
POWER_NAME = 1

#: Powers in dBm at which to plot the V-I characteristic
POWERS = [-8, -3, 1, 4, 5.6, 7.6]

#: Power at which we observe the gap closing
CRITICAL_POWER = 5.6

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = -1

#: Name or index of the column containing the current data
CURRENT_NAME = 0

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Number of points to use to correct for the offset of the voltage using the
#: lowest power scan.
CORRECT_Y_OFFSET = 50

#: Should the Y axis be normalised in unit of the Shapiro step size hf/2e
NORMALIZE_Y = True

#: Label of the x axis, if left blanck the current column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Bias Current (µA)'

#: Label of the y axis, if left blanck the voltage column name will be used
#: If an index was passed the name found in the labber file is used.
Y_AXIS_LABEL = 'Voltage drop (hf/2e)'

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1e6

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1

#: Limits to use for the x axis (after scaling)
X_LIMITS = [0, 5]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [-0.1, 8]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = [1, 2, 3]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.shapiro import shapiro_step
from shabanipy.utils.labber_io import LabberData

for frequency in FREQUENCIES:
    print(f'Treating data for frequency {frequency/1e9} GHz')
    plt.figure()
    with LabberData(PATH) as data:

        filters = {POWER_NAME: 0}
        if FREQUENCY_NAME is not None:
            filters[FREQUENCY_NAME] = frequency

        # Get the offset on the voltage axis
        if CORRECT_Y_OFFSET:
            min_p = min(POWERS)
            filters[POWER_NAME] = min_p
            offset = np.average(data.get_data(VOLTAGE_NAME,
                                        filters=filters)[:CORRECT_Y_OFFSET])

        for p in POWERS:
            filters[POWER_NAME] = p
            volt = data.get_data(VOLTAGE_NAME,
                                 filters=filters)
            curr = data.get_data(CURRENT_NAME,
                                 filters=filters)

            # Convert the current data if requested
            if CURRENT_CONVERSION is not None:
                curr *= CURRENT_CONVERSION

            if CORRECT_Y_OFFSET:
                volt -= offset

            if NORMALIZE_Y:
                volt /= shapiro_step(frequency)

            # Apply scaling factor before plotting
            volt *= Y_SCALING
            curr *= X_SCALING

            # Plot the data
            plt.plot(curr, volt, label='Power %g dB' % (p - CRITICAL_POWER))

    if SHOW_SHAPIRO_STEP:
        if NORMALIZE_Y:
            steps = SHOW_SHAPIRO_STEP
        else:
            steps = [n * shapiro_step(frequency) * Y_SCALING
                    for n in SHOW_SHAPIRO_STEP]
        lims = X_LIMITS or (curr[0], curr[-1])
        plt.hlines(steps, *lims, linestyles='dashed')

    sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
    plt.title(f'Frequency {frequency/1e9} GHz')
    plt.xlabel(X_AXIS_LABEL or CURRENT_NAME)
    plt.ylabel(Y_AXIS_LABEL or VOLTAGE_NAME)
    if X_LIMITS:
        plt.xlim(X_LIMITS)
    if Y_LIMITS:
        plt.ylim(Y_LIMITS)
    plt.legend()
    plt.tight_layout()

    if FIG_DIRECTORY:
        plt.savefig(os.path.join(FIG_DIRECTORY,
                                f'{frequency/1e9}GHz_' +
                                os.path.split(PATH)[1].split('.')[0] +
                                '.pdf'))

plt.show()
