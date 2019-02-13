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

#: Name or index of the column containing the power data
TEMPERATURE_NAME = 1

#: Name or index of the column containing the voltage data
FIELD_NAME = 0

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CONDUCTANCE_NAME = 2

#: Label of the x axis, if left blanck the power column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Power (dBm)'

#: Label of the y axis.
Y_AXIS_LABEL = 'Junction voltage (hf/2e)'

#: Label of the colorbar.
C_AXIS_LABEL = 'Counts (µA)'

#: Scaling factor for the c axis (used to convert between units)
C_SCALING = 1e6

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [1.5, 8]

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [None, None]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.shapiro.binning import bin_power_shapiro_steps
from shabanipy.utils.labber_io import LabberData

with LabberData(PATH) as data:

    shape = data.compute_shape((FIELD_NAME, TEMPERATURE_NAME))

    field = data.get_data(FIELD_NAME)
    temp = data.get_data(TEMPERATURE_NAME)
    cond = np.abs(data.get_data(CONDUCTANCE_NAME))

    # Handle interruptions in the last scan.
    while len(field) < shape[0]*shape[1]:
        shape[1] -= 1

    length = shape[0]*shape[1]
    field = field[:length].reshape(shape)
    temp = temp[:length].reshape(shape)
    cond = cond[:length].reshape(shape).T
    for i in range(1, cond.shape[0], 2):
        cond[i, :] = cond[i, ::-1]


# Plot the data
plt.imshow(C_SCALING * cond,
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
