"""Plot the VI-characteristic of the JS129 JJ with overlaid fits and an inset.

"""

# ======================================================================================
# --- Parameters -----------------------------------------------------------------------
# ======================================================================================

#: Path to the file containing the data.
PATH = "/Users/mdartiailh/Labber/Data/2020/02/Data_0228/JJ307_7JJ-hBN-1_KSMD001_JJ5_012.hdf5"

#: For all the following parameters one can use a dictionary with sample names
#: as keys can be used.

#: Name or index of the column containing the voltage bias data.
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
BIAS_NAME = 0

#: Name or index of the column containing the current sent to the magnet.
MAGNET_NAME = 1

#: Name or index of the column containing an extra parameter (gate) to use to
#: filter the date. Use None if no such parameters.
FILTER_NAME = 2

#: Filter value to use in the plot
FILTER_VALUE = 2.0

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 3

#: Voltage threshold used to extract the critical current
VOLTAGE_THRESHOLD = 0.1

#: Voltage to resistance conversion factor
VOLTAGE_TO_RESISTANCE = 1e8

#: Low and high limits of the magnet current values to include
#: Use None to not limit the range
CURRENT_LIMITS = [None, None]

#: Magnet current values to skip
CURRENT_TO_SKIP = []

#: Current at which the center of the Fraunhofer pattern is localized.
CURRENT_OFFSET = -1e-3

#: Current to field conversion factor (A to T)
CURRENT_FIELD_CONVERSION = 18.2

#: Savgol filter parameters to use before fitting (points, order)
SAVGOL_FILTERING = (7, 3)

#: Gaussian width to use to weight the data during fitting (useful when the
#: Fraunhofer is not periodic). The width is expressed in unit of the central
#: lobe width. Use None to apply no weight when fitting.
GAUSSIAN_WIDTH = None


# ======================================================================================
# --- Execution ------------------------------------------------------------------------
# ======================================================================================

import os
import warnings
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.signal import savgol_filter

from lmfit import minimize

from shabanipy.labber import LabberData

plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42

get_value = lambda s, p: p[s] if isinstance(p, dict) else p

sample = None
plt.figure(0, figsize=(3.4, 3.4))

with LabberData(PATH) as data:

    field = np.unique(data.get_data(MAGNET_NAME))
    bias = np.unique(data.get_data(BIAS_NAME))

    resistance = data.get_data(VOLTAGE_NAME, filters={GATE_NAME: GATE_VALUE})

    # Line index match field, column index match bias
    resistance = data.reshape_data((BIAS_NAME, MAGNET_NAME), resistance).T

# Remove the bad field values
masks = [
    np.greater(np.abs(field - black_value), 1e-12) for black_value in CURRENT_TO_SKIP
]
if CURRENT_LIMITS[0] is not None:
    masks.append(np.greater_equal(field, CURRENT_LIMITS[0]))
if CURRENT_LIMITS[1] is not None:
    masks.append(np.less_equal(field, CURRENT_LIMITS[1]))

if masks:
    mask = masks[0]
    for m in masks:
        np.logical_and(mask, m, mask)

    resistance = resistance[mask]
    field = field[mask] - CURRENT_OFFSET

field -= CURRENT_OFFSET
field *= 1e3 / 18.2

plt.figure(constrained_layout=True)
plt.plot(field, 2.9 * np.abs(np.sinc(field / 0.45)), color="C1")
plt.imshow(
    resistance.T,
    origin="lower",
    aspect="auto",
    extent=(field[0], field[-1], bias[0], bias[-1]),
    vmax=150,
)
plt.xlabel("Out of plane field (mT)")
plt.ylabel("Bias current (µA)")
cbar = plt.colorbar()
cbar.ax.set_ylabel("Differential resistance (Ω)")

plt.show()
