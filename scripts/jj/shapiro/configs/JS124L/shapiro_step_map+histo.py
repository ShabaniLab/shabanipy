#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2018/09/Data_0913/JS124L_CD004_009.hdf5'

#: Name or index of the column containing the frequency data if applicable.
#: Leave blanck if the datafile does not contain a frequency sweep.
FREQUENCY_NAME = 2

#: Frequencies of the applied microwave in Hz (one graph will be generated for
#: each frequecy).
#: If a FREQUENCY_NAME is supplied data are filtered.
FREQUENCIES = [6e9, 12e9]

#: Name or index of the column containing the power data
POWER_NAME = 1

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 3

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CURRENT_NAME = 0

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Conversion factor to apply to the voltage data (take into account possible
#: amplification)
VOLTAGE_CONVERSION = 1

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.1

#: Critical power at which we close the gap. One can use a dictionary with
#: frequencies as key.
CRITICAL_POWER = {6e9: 11, 12e9: 9}

#: Label of the x axis, if left blanck the power column name will be used
#: If an index was passed the name found in the labber file is used.
X_AXIS_LABEL = 'Power (dBm)'

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
C_SCALING = 1e6/5.2

#: Limits to use for the x axis (after scaling)
X_LIMITS = [None, None]

#: Limits to use for the y axis (after scaling)
Y_LIMITS = [-5, 5]

#: Limits to use on the colorscale (after scaling). Use None for autoscaling.
C_LIMITS = [0, 0.1]

#: Plot dashed lines for the specified Shapiro steps
SHOW_SHAPIRO_STEP = {6e9: [-1, -2, -3, -4],
                     12e9: [-1, -2, -3, -4],
                     }

#: Power range in which to plot the dashed lines for shapiro steps. Use None
#: to indicate that the line should start/stop at the edge.
SHAPIRO_STEPS_POWERS = [None, -12]

#: Display an histogram of the counts at a given power next to the 2D map. Use
#: None to not display an histogram. A dict with per frequency can be used.
HISTOGRAM_AT_POWER = {6e9: 5, 12e9: 2}

#: Index of the Shapiro step whose weight to plot as a function of power.
SHAPIRO_WEIGTHS = [0, -1, -2, -3, -4]

#: Number of histogram to average together to obtain the weight plot. Should be odd
SHAPIRO_WEIGTHS_AVG = 1
