#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field, file path.
DATA_PATHS = {300: '04/Data_0412/JS124S_BM002_487.hdf5',
              250: '03/Data_0330/JS124S_BM002_447.hdf5',
              150: '04/Data_0418/JS124S_BM002_504.hdf5',}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {300: (-10e-3, -7.66e-3),
                250: (-5.76e-3, 0.8e-3),
                150: (-7.6e-3, None),}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 2

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 3

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = False

#: Should we enforce the equality of the transparencies.
EQUAL_TRANSPARENCIES = True

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = -1

#: Handedness of the system.
HANDEDNESS = 1

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = True

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 1.0

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_low_field/j2/Bx/equal_t')
