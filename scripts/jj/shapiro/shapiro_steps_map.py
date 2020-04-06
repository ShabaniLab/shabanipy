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

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'JS129D/shapiro_step_map+histo.py'

#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/11/Data_1114/JS129D_BM001_038.hdf5'

#: Directory in which to save the figure.
FIG_DIRECTORY = '/Users/mdartiailh/Documents/PostDocNYU/Papers/Dartiailh-shapiro-steps/raw-figures/JS129D'

#: Name or index of the column containing the frequency data if applicable.
#: Leave blanck if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = 2

#: Frequencies of the applied microwave in Hz (one graph will be generated for
#: each frequecy).
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [7e9, 9e9, 11e9, 13e9]

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

#: Conversion factor to apply to the voltage data (take into account possible
#: amplification)
VOLTAGE_CONVERSION = 1e-2

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.05

#: Critical power at which we close the gap. One can use a dictionary with
#: frequencies as key.
CRITICAL_POWER = {7e9: -6.3, 9e9: 2.4, 11e9: 8.8, 13e9: 7}

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
Y_OFFSET_CORRECTION = 20

#: Scaling factor for the c axis (used to convert between units)
C_SCALING = 1e6/1.8

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [None, None]

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [0, 0.1]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = {7e9: [1, 2, 3, 4],
                     9e9: [1, 2, 3],
                     11e9: [1, 2],
                     13e9: [0.5, 1, 1.5, 2]}

#: Power range in which to plot the dashed lines for shapiro steps. Use None
#: to indicate that the line should start/stop at the edge.
SHAPIRO_STEPS_POWERS = [None, -9]

#: Display an histogram of the counts at a given power next to the 2D map. Use
#: None to not display an histogram. A dict with per frequency can be used.
HISTOGRAM_AT_POWER = {7e9: -11, 9e9: -5, 11e9: 2.5, 13e9: -1}

#: Index of the Shapiro step whose weight to plot as a function of power.
SHAPIRO_WEIGTHS = [0, -1, -2]

#: Number of histogram to average together to obtain the weight plot. Should be odd
SHAPIRO_WEIGTHS_AVG = 3

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from shabanipy.jj.iv_analysis import compute_voltage_offset
from shabanipy.jj.shapiro.binning import bin_power_shapiro_steps
from shabanipy.utils.labber_io import LabberData

plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42

if CONFIG_NAME:
    print(
        f"Using configuration {CONFIG_NAME}, all scripts constants will be"
        " overwritten."
    )
    path = os.path.join(os.path.dirname(__file__), "configs", CONFIG_NAME)
    with open(path) as f:
        exec(f.read())

if not isinstance(CRITICAL_POWER, dict):
    CRITICAL_POWER = dict.fromkeys(FREQUENCIES, CRITICAL_POWER)

for frequency in FREQUENCIES:
    print(f"Treating data for frequency {frequency/1e9} GHz")
    with LabberData(PATH) as data:

        filters = {}
        if FREQUENCY_NAME is not None:
            filters[FREQUENCY_NAME] = frequency

        shape = data.compute_shape((POWER_NAME, CURRENT_NAME))

        power = data.get_data(POWER_NAME, filters=filters)
        volt = data.get_data(VOLTAGE_NAME, filters=filters)
        curr = data.get_data(CURRENT_NAME, filters=filters)

        # Handle interruptions in the last scan.
        while len(power) < shape[0] * shape[1]:
            shape[1] -= 1

        length = shape[0] * shape[1]
        power = power[:length].reshape(shape)
        volt = volt[:length].reshape(shape)
        curr = curr[:length].reshape(shape)

    # Convert the current data if requested
    if CURRENT_CONVERSION is not None:
        curr *= CURRENT_CONVERSION

    if Y_OFFSET_CORRECTION:
        offset = compute_voltage_offset(curr[:, 0], volt[:, 0], Y_OFFSET_CORRECTION)
        volt -= offset

    if VOLTAGE_CONVERSION is not None:
        volt *= VOLTAGE_CONVERSION

    # Bin the data
    power, voltage, histo = bin_power_shapiro_steps(
        power, curr, volt, frequency, STEP_FRACTION
    )

    cp = (
        CRITICAL_POWER.get(frequency)
        if isinstance(CRITICAL_POWER, dict)
        else CRITICAL_POWER
    )
    if cp is not None:
        power -= cp

    # Determine x limits
    x_lims = [x or power[-i + (-1) ** i] * X_SCALING for i, x in enumerate(X_LIMITS)]

    # Plot the data
    h_power = HISTOGRAM_AT_POWER.get(frequency)
    indexes = (
        SHAPIRO_WEIGTHS.get(frequency)
        if isinstance(SHAPIRO_WEIGTHS, dict)
        else SHAPIRO_WEIGTHS
    )

    if h_power is not None and indexes is not None:
        f = plt.figure(constrained_layout=True, figsize=(9, 6))
        spec = f.add_gridspec(
            ncols=2, nrows=2, width_ratios=(1, 3), height_ratios=(1, 3)
        )
        w_ax = f.add_subplot(spec[0, 1])
        h_ax = f.add_subplot(spec[1, 0])
        m_ax = f.add_subplot(spec[1, 1])
        m_ax.tick_params(labelleft=False)
        w_ax.tick_params(labelbottom=False)
        index = np.argmin(np.abs(power + cp - h_power))
        h_ax.barh(voltage, C_SCALING * histo[index], height=STEP_FRACTION)
        h_ax.set_xlabel(C_AXIS_LABEL)
        h_ax.set_ylabel(Y_AXIS_LABEL)
        h_ax.set_xlim(C_LIMITS)
        h_ax.set_ylim(voltage[0], voltage[-1])
    else:
        f = plt.figure(constrained_layout=True)
        m_ax = f.gca()

    im = m_ax.imshow(
        C_SCALING * histo.T,
        extent=(power[0], power[-1], voltage[0], voltage[-1]),
        origin="lower",
        aspect="auto",
        vmin=C_LIMITS[0],
        vmax=C_LIMITS[1],
    )
    cbar = f.colorbar(im, ax=m_ax, aspect=50)
    cbar.ax.set_ylabel(C_AXIS_LABEL)

    ssstep = (
        SHOW_SHAPIRO_STEP.get(frequency, None)
        if isinstance(SHOW_SHAPIRO_STEP, dict)
        else SHOW_SHAPIRO_STEP
    )
    if ssstep:
        steps = [n * Y_SCALING for n in ssstep]
        if not X_LIMITS:
            X_LIMITS = [None, None]
        lims = [
            x or X_LIMITS[i] or power[-i] for i, x in enumerate(SHAPIRO_STEPS_POWERS)
        ]
        m_ax.hlines(steps, *lims, linestyles="dashed")

    if h_power is not None:
        lims = [y or voltage[-i] for i, y in enumerate(Y_LIMITS)]
        m_ax.vlines([h_power - cp], *lims, linestyle="dotted")

    sample = (PATH.rsplit(os.sep, 1)[1]).split("_")[0]
    f.suptitle(f"Frequency {frequency/1e9} GHz")
    m_ax.set_xlabel(X_AXIS_LABEL or POWER_NAME)
    if not (h_power is not None and indexes is not None):
        m_ax.set_ylabel(Y_AXIS_LABEL)

    m_ax.set_xlim(x_lims)
    w_ax.set_xlim(x_lims)
    if Y_LIMITS:
        h_ax.set_ylim(Y_LIMITS)
        m_ax.set_ylim(Y_LIMITS)

    # Plot the evolution of the weight of the shapiro steps as a function of power.
    indexes = (
        SHAPIRO_WEIGTHS.get(frequency)
        if isinstance(SHAPIRO_WEIGTHS, dict)
        else SHAPIRO_WEIGTHS
    )
    avg = (
        SHAPIRO_WEIGTHS_AVG.get(frequency)
        if isinstance(SHAPIRO_WEIGTHS_AVG, dict)
        else SHAPIRO_WEIGTHS_AVG
    )
    if avg % 2 != 1:
        raise ValueError(
            "Need an odd number for SHAPIRO_WEIGTHS_AVG " f"at frequency {frequency}"
        )
    avg_i = (avg - 1) // 2
    ylim = 0
    for i in indexes:
        index = np.argmin(np.abs(voltage - i))
        data = C_SCALING * np.average(
            histo[:, index - avg_i : index + avg_i + 1], axis=-1
        )
        data = savgol_filter(data, 7, 2)
        if i != 0:
            ylim = max(ylim, np.max(data))
        w_ax.plot(power, data, label=f"Step {i}")
    w_ax.set_ylim((0, 1.2 * ylim))
    w_ax.set_ylabel(C_AXIS_LABEL)
    w_ax.legend()

    if FIG_DIRECTORY:
        f.savefig(
            os.path.join(
                FIG_DIRECTORY,
                f"histo_{frequency/1e9}GHz_"
                + os.path.split(PATH)[1].split(".")[0]
                + ".pdf",
            )
        )

plt.show()
