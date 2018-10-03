# -*- coding: utf-8 -*-
"""Generate a set linear plot (V-I) for different to powers

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r''

#: Frequency of the applied microwave to consider if teh file contains multiple
#: frequencies. Use None if the datafile does not contain a frequency sweep.
FREQUENCY = None

#: Powers in dBm at which to plot the V-I characteristic
POWERS = []

#: Name of the column containing the voltage data
VOLTAGE_NAME = ''

#: Name of the column containing the current data
CURRENT_NAME = ''

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = ''

#: Label of the x axis, if left blanck the current column name will be used
X_AXIS_LABEL = ''

#: Label of the y axis, if left blanck the voltage column name will be used
Y_AXIS_LABEL = ''

#: Scaling factor for the x axis (used to convert between units)
X_SCALING = 1

#: Scaling factor for the y axis (used to convert between units)
Y_SCALING = 1

#: Limits to use for the x axis (after scaling)
X_LIMITS = []

#: Limits to use for the y axis (after scaling)
Y_LIMITS = []

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================


