# -*- coding: utf-8 -*-
"""Generate a set linear plot (V-I) for different to powers and a map of the
differential conductance

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'JS131A/shapiro_steps_cuts+dvdi.py'

#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/11/Data_1114/JS129D_BM001_038.hdf5'

#: Directory in which to save the figure.
FIG_DIRECTORY = '/Users/mdartiailh/Documents/PostDocNYU/Papers/Dartiailh-shapiro-steps/raw-figures/JS131A'

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

#: Powers in dBm at which to plot the V-I characteristic. To use different
#: powers per frequency use a dictionary.
POWERS = {7e9: [-19.5, -15, -12, -7],
          9e9: [-9.5, -7, -3, -1],
          11e9: [-4.5, -2, 1, 8],
          13e9: [-9.5, -4, -2, 1],
          }

#: Power at which we observe the gap closing. To use different values per
#: frequency use a dictionary.
CRITICAL_POWER = {7e9: -6.3, 9e9: 2.4, 11e9: 8.8, 13e9: 7}

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Conversion factor to apply to the voltage data (take into account possible
#: amplification)
VOLTAGE_CONVERSION = 1e-2

#: Number of points to use to correct for the offset of the voltage using the
#: lowest power scan.
Y_OFFSET_CORRECTION = 20

#: Should the Y axis be normalized in unit of the Shapiro step size hf/2e
NORMALIZE_Y = True

#: Label of the x axis, if left blank the current column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Bias Current (µA)'

#: Label of the y axis, if left blank the voltage column name will be used
#: If an index was passed the name found in the labber file is used.
Y_AXIS_LABEL = 'Voltage drop (hf/2e)'

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1e6

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [None, None]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = [-3, -2, -1, 1, 2, 3]

#: Label of the y axis for the differential resistance map.
DIFF_Y_AXIS_LABEL = "Microwave power (dB)"

#: Label of the colorbar in the differential resistance map
DIFF_C_AXIS_LABEL = "Differential resistance (Ω)"

#: Limits for the colorbar in the differential resistance map
DIFF_C_AXIS_LIMITS = (0, 300)

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.jj.iv_analysis import compute_voltage_offset
from shabanipy.jj.shapiro import shapiro_step
from shabanipy.utils.labber_io import LabberData

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['font.size'] = 13
plt.rcParams['pdf.fonttype'] = 42

if CONFIG_NAME:
    print(f"Using configuration {CONFIG_NAME}, all scripts constants will be"
          " overwritten.")
    path = os.path.join(os.path.dirname(__file__), 'configs', CONFIG_NAME)
    with open(path) as f:
        exec(f.read())

for frequency in FREQUENCIES:
    print(f'Treating data for frequency {frequency/1e9} GHz')
    f, (vi_ax, dr_ax) = plt.subplots(2, 1, figsize=(7, 9),
                                     gridspec_kw=dict(height_ratios=(1, 2)),
                                     sharex=True, constrained_layout=True)
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

    if VOLTAGE_CONVERSION is not None:
        volt *= VOLTAGE_CONVERSION

    # Get the offset on the voltage axis
    if Y_OFFSET_CORRECTION:
        offset = compute_voltage_offset(curr[:, 0], volt[:, 0], Y_OFFSET_CORRECTION)
        volt -= offset

    # Get the critical power at the considered frequency
    cp = (CRITICAL_POWER.get(frequency)
            if isinstance(CRITICAL_POWER, dict) else CRITICAL_POWER)
    if cp is not None:
        power -= cp

    # Determine x limits
    x_lims = [x or curr[-i + (-1)**i, 0]*X_SCALING
              for i, x in enumerate(X_LIMITS)]


    # Plot the differential conductance map
    vm, vp = volt[:-2], volt[2:]
    diff_r = np.abs((vp - vm)/(curr[2, 0] - curr[0, 0]))  # Ugly hack for alternating data
    im = dr_ax.imshow(diff_r.T,
                      extent=(curr[1, 0]*X_SCALING, curr[-2, 0]*X_SCALING,
                              power[0, 0], power[0, -1]),
                      origin='lower',
                      aspect='auto',
                      vmin=DIFF_C_AXIS_LIMITS[0],
                      vmax=DIFF_C_AXIS_LIMITS[1])
    cbar = f.colorbar(im, ax=dr_ax, aspect=50)
    cbar.ax.set_ylabel(DIFF_C_AXIS_LABEL)

    # Plot the VI at different powers
    powers = POWERS[frequency] if isinstance(POWERS, dict) else POWERS
    for i, p in enumerate(reversed(sorted(powers))):
        index = np.argmin(np.abs(power[0] + cp - p))

        dr_ax.hlines([p - cp], *x_lims, linestyles='dashed', color=f"C{i}")

        v = volt[:, index]
        if NORMALIZE_Y:
            v /= shapiro_step(frequency)

        # Apply scaling factor before plotting
        v *= Y_SCALING

        # Plot the data
        vi_ax.plot(curr[:, index]*X_SCALING, v, label='Power %g dB' % (p - cp))

    # --- Generate the differential resistance plot
    if SHOW_SHAPIRO_STEP:
        if NORMALIZE_Y:
            steps = SHOW_SHAPIRO_STEP
        else:
            steps = [n * shapiro_step(frequency) * Y_SCALING
                    for n in SHOW_SHAPIRO_STEP]
        vi_ax.hlines(steps, *x_lims, linestyles='dashed')

    sample = (PATH.rsplit(os.sep, 1)[1]).split('_')[0]
    f.suptitle(f'Frequency {frequency/1e9} GHz')
    vi_ax.set_ylabel(Y_AXIS_LABEL or VOLTAGE_NAME)
    vi_ax.set_xlim(x_lims)
    if Y_LIMITS:
        vi_ax.set_ylim(Y_LIMITS)
    vi_ax.legend()

    dr_ax.set_xlabel(X_AXIS_LABEL or CURRENT_NAME)
    dr_ax.set_ylabel(DIFF_Y_AXIS_LABEL or POWER_NAME)

    if FIG_DIRECTORY:
        f.savefig(os.path.join(FIG_DIRECTORY,
                                f'VI_{frequency/1e9}GHz_' +
                                os.path.split(PATH)[1].split('.')[0] +
                                '.pdf'))

plt.show()
