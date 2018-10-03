# -*- coding: utf-8 -*-
"""Generate a color plot of a shapiro step experiment through binning.

The plot use the folling axes:
- x axis: Power
- y axis: Normalized voltage
- color axis: bin count

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r''

#: Frequency of the applied microwave to consider in Hz.
FREQUENCY = None

#: Name of the column containing the power data
POWER_NAME = ''

#: Name of the column containing the voltage data
VOLTAGE_NAME = ''

#: Name of the column containing the current data
CURRENT_NAME = ''

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.1

#: Label of the x axis, if left blanck the current column name will be used
X_AXIS_LABEL = ''

#: Label of the y axis, if left blanck the voltage column name will be used
Y_AXIS_LABEL = ''

#: Limits to use for the x axis (after scaling)
X_LIMITS = []

#: Limits to use for the y axis (after scaling)
Y_LIMITS = []

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================