#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field: pair of file path. The first path should refer
#: to the dataset in which the j1 junction is gated, the second to the dataset
#: in which the j2 junction is gated.
DATA_PATHS = {50: ['04/Data_0409/JS124S_BM002_476.hdf5',
                   '03/Data_0320/JS124S_BM002_402.hdf5']}

#: Perpendicular field range to fit for each parallel field.
FIELD_RANGES = {50: [(), (-2.15e-3, None)]}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {50: 0.95}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5, 2, 3]

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 2

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 3

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7 # 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = False

#: Sign of the phase difference created by the perpendicular field. The
#: phase difference is applied on the junction j1.
PHASE_SIGN = (1, -1)

#: Handedness of the system ie does a positive field translate in a negative
#: or positive phase difference.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = (0.98, 1.0)

#: Fix the anomalous phase to 0.
FIX_PHI_ZERO = False

#: Enforce equality of the transparencies
EQUAL_TRANSPARENCIES = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
#: Recognized values are False, True, 'color' (to plot over the colormap)
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_low_field/combined/By/50mT')