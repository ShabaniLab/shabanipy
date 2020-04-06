#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/02/Data_0205/JS124S_BM002_169.hdf5'

#: Directory in which to save the figure.
FIG_DIRECTORY = ''

#: Name or index of the column containing the frequency data if applicable.
#: Leave blanck if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = None

#: Frequencies of the applied microwave in Hz (one graph will be generated for
#: each frequecy).
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [4e9, 6e9]

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
POWERS = {4e9: [-18, -7, -5, -3], 6e9: [-14, -7, -3, 0],\
          }

#: Power at which we observe the gap closing. To use different values per
#: frequency use a dictionary.
CRITICAL_POWER = {4e9: -1.8, 6e9: 3.2}

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Conversion factor to apply to the voltage data (take into account possible
#: amplification)
VOLTAGE_CONVERSION = 1

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