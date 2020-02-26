# -*- coding: utf-8 -*-
"""Generate a color plot of the weight of a shapiro steps.

The plot is done as a function of power and frequency and the power is
normalized by the power at which the step 0 disappear.

The plot use the folling axes:
- x axis: Normalized power
- y axis: Frequency
- color axis: bin count in current

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r'/Users/mdartiailh/Labber/Data/2018/09/Data_0913/JS124L_CD004_009.hdf5'

#: Index of the setps for which to generate a plot.
STEP_INDEXES = [0, 1, 2, 3, 4]

#: Name or index of the column containing the frequency data if applicable.
#: Leave blanck if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = 2

#: Frequencies of the applied microwave in Hz. Use an empty list to use all
#: the frequencies found in the meaurement.
FREQUENCIES = [4e9, 6e9, 8e9, 12e9, 14e9]

#: Name or index of the column containing the power data
POWER_NAME = 1

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 3

#: Should we correct the offset in voltage.
CORRECT_VOLTAGE_OFFSET = True

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CURRENT_NAME = 0

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.2

#: Threshold as fraction of the low power step used to identify the normalizing
#: power, defined at the first power for which the count of the step 0 is below
#: the threshold.
NORMALIZING_THRESHOLD = 0.1

#: Label of the x axis, if left blanck the power column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Normalized power (a.u.)'

#: Label of the y axis.
Y_AXIS_LABEL = 'Frequency (GHz)'

#: Label of the colorbar.
C_AXIS_LABEL = 'Counts (µA)'

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1e-9

#: Scaling factor for the c axis (used to convert between units)
C_SCALING = 1e6

#: Limits to use for the x axis (after scaling)
X_LIMITS = []

#: Limits to use for the y axis (after scaling)
Y_LIMITS = []

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [None, 1]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings
import math

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.jj.shapiro import normalize_db_power
from shabanipy.jj.shapiro.binning import (bin_power_shapiro_steps,
                                       extract_step_weight)
from shabanipy.utils.labber_io import LabberData

with LabberData(PATH) as data:

    frequencies = FREQUENCIES or np.unique(data.get_data(FREQUENCY_NAME))
    grid_f = []
    grid_p = []
    step_counts = {s_i: [] for s_i in STEP_INDEXES}

    for frequency in frequencies:
        filters = {}
        filters[FREQUENCY_NAME] = frequency

        shape = data.compute_shape((POWER_NAME, CURRENT_NAME))

        power = data.get_data(POWER_NAME, filters=filters)
        volt = data.get_data(VOLTAGE_NAME, filters=filters)
        curr = data.get_data(CURRENT_NAME, filters=filters)

        if CORRECT_VOLTAGE_OFFSET:
            volt -= volt[np.argmin(np.abs(curr))]

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

        # Bin the data
        power, voltage, histo = bin_power_shapiro_steps(power, curr, volt,
                                                        frequency,
                                                        STEP_FRACTION)

        # Find the normalizing power
        step_0 = extract_step_weight(voltage, histo, 0)
        indexes = np.where(np.less(step_0, step_0[0]*NORMALIZING_THRESHOLD))[0]
        print(f'At f={frequency/1e9} above threshold power: {power[indexes]}')
        if len(indexes):
            norm_p = power[np.min(indexes)]
        else:
            msg = ('Power was always lower than threshold for '
                   f'f={frequency/1e9:.1f} GHz')
            warnings.warn(msg)
            norm_p = power[-1]

        # Fill the results
        grid_f.append(frequency*np.ones(len(power)))
        grid_p.append(normalize_db_power(power, norm_p))
        for i, res in step_counts.items():
            res.append(extract_step_weight(voltage, histo, i))

# Create the base for the mesh by joining the frequency and power arrays
tri_f = np.concatenate(grid_f)*Y_SCALING
tri_p = np.concatenate(grid_p)*X_SCALING

# Plot the data
for step_index in step_counts:

    plt.figure()
    weights = np.concatenate(step_counts[step_index])
    plt.tricontourf(tri_p, tri_f,
                    weights*C_SCALING,
                    vmin=C_LIMITS[0],
                    vmax=C_LIMITS[1])
    m = plt.cm.ScalarMappable()
    m.set_array(weights*C_SCALING)
    m.set_clim(*C_LIMITS)
    cbar = plt.colorbar(m)
    cbar.ax.set_ylabel(C_AXIS_LABEL)
    plt.scatter(tri_p, tri_f, marker='1', color='k')

    sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
    plt.title(f'Sample {sample}: Step {step_index}')
    plt.xlabel(X_AXIS_LABEL)
    plt.ylabel(Y_AXIS_LABEL)
    if X_LIMITS:
        plt.xlim(X_LIMITS)
    if Y_LIMITS:
        plt.ylim(Y_LIMITS)
    plt.tight_layout()

    # Get the index of the associated Shapiro step that needs to be used when
    # plotting Q
    associated_index = step_index + math.copysign(1, step_index)
    if (step_index % 2 == 1 and associated_index in step_counts):
        plt.figure()
        weights_next = np.concatenate(step_counts[associated_index])
        # Get the indexes at which the denominator does not vanish
        indexes = np.nonzero(weights_next)
        # Compute the ratio
        ratio = weights[indexes]/weights_next[indexes]
        plt.tricontourf(tri_p[indexes], tri_f[indexes],
                        ratio, levels=np.arange(0, 2.0, 0.25),
                        vmin=0,
                        vmax=2)
        m = plt.cm.ScalarMappable()
        m.set_array(weights[indexes]/weights_next[indexes])
        m.set_clim(0, 2)
        cbar = plt.colorbar(m)
        cbar.ax.set_ylabel('$Q_{%d, %d}$' % (step_index, associated_index))
        plt.scatter(tri_p[indexes], tri_f[indexes], marker='1', color='k')

        sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
        plt.title(f'Sample {sample}: $Q_{{%d, %d}}$' %
                  (step_index, associated_index))
        plt.xlabel(X_AXIS_LABEL)
        plt.ylabel(Y_AXIS_LABEL)
        if X_LIMITS:
            plt.xlim(X_LIMITS)
        if Y_LIMITS:
            plt.ylim(Y_LIMITS)
        plt.tight_layout()

plt.show()
