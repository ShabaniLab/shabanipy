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
PATH = '/Users/kasrasardashti/Labber/Data/2019/02/Data_0209/KS7_KS8_JS221_JY001_005.hdf5'

#: Name or index of the column containing the temperature data
TEMPERATURE_NAME = 'Oxford Mercury iTC - Sensor2.TargetTemperature'

#: Name or index of the column containing the field data
MAGFIELD_NAME = 'Magnet - Magnetic Field'

#: Name or index of the column containing the resistance data
#: This should be a stepped channel ! use the measured voltage from lock-in
RESISTANCE_NAME_1 = 'Lockin-1 - Value'
RESISTANCE_NAME_2 = 'Lockin-2 - Value'
RESISTANCE_NAME_3 = 'Lockin-3 - Value'

#: Label of the x axis, if left blanck the field column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Magnetic Field (B)'

#: Label of the y axis.
Y_AXIS_LABEL = 'Temperature (K)'

#: Label of the colorbar.
C_AXIS_LABEL = r'Resistance ($\Omega$)'

#: Scaling factor for the c axis (used to convert between units)
C_SCALING = 1e6

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [1.4, 8]

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [None, None]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.utils.labber_io import LabberData

with LabberData(PATH) as data:

    shape = data.compute_shape((MAGFIELD_NAME, TEMPERATURE_NAME))

    field = data.get_data(MAGFIELD_NAME)
    temp = data.get_data(TEMPERATURE_NAME)
    resist_1 = np.abs(data.get_data(RESISTANCE_NAME_1))
    resist_2 = np.abs(data.get_data(RESISTANCE_NAME_2))

    # Handle interruptions in the last scan.
    while len(field) < shape[0]*shape[1]:
        shape[1] -= 1

    length = shape[0]*shape[1]
    field = field[:length].reshape(shape)
    temp = temp[:length].reshape(shape)
    resist_1 = resist_1[:length].reshape(shape).T
    resist_2 = resist_2[:length].reshape(shape).T

    for i in range(1, resist_1.shape[0], 2):
        resist_1[i, :] = resist_1[i, ::-1]
        resist_2[i, :] = resist_2[i, ::-1]


# Plot the data
plt.figure(1)
plt.imshow(C_SCALING * resist_1,
           extent=(field[0, 0], field[-1, 0], temp[0, 0], temp[0, -1]),
           origin='lower',
           aspect='auto',
           vmin=C_LIMITS[0],
           vmax=C_LIMITS[1])
plt.xlabel(X_AXIS_LABEL)
plt.xlim(X_LIMITS)
plt.ylabel(Y_AXIS_LABEL)
plt.ylim(Y_LIMITS)
cbar = plt.colorbar()
cbar.ax.set_ylabel(C_AXIS_LABEL)

plt.figure(2)
plt.imshow(C_SCALING * resist_2,
           extent=(field[0, 0], field[-1, 0], temp[0, 0], temp[0, -1]),
           origin='lower',
           aspect='auto',
           vmin=C_LIMITS[0],
           vmax=C_LIMITS[1])
plt.xlabel(X_AXIS_LABEL)
plt.xlim(X_LIMITS)
plt.ylabel(Y_AXIS_LABEL)
plt.ylim(Y_LIMITS)
cbar = plt.colorbar()
cbar.ax.set_ylabel(C_AXIS_LABEL)

plt.show()
