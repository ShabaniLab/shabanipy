#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
            #   400: 'Data_0405/JS124S_BM002_465.hdf5',
              350: 'Data_0406/JS124S_BM002_466.hdf5',
            #   300: 'Data_0406/JS124S_BM002_467.hdf5',
            #   250: 'Data_0406/JS124S_BM002_468.hdf5',
              200: 'Data_0407/JS124S_BM002_470.hdf5',
            #   150: 'Data_0407/JS124S_BM002_471.hdf5',
            #   100: 'Data_0409/JS124S_BM002_474.hdf5',
              50:  'Data_0409/JS124S_BM002_476.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (None, -1.3e-3),
                350: (None, 0.5e-3),
                300: (),
                250: (),
                200: (0.5e-3, None),
                150: (),
                100: (None, 0),
                50: ()}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,
                      350: 0.1,
                      300: 0.2,
                      250: 0.3,
                      200: 0.8,
                      150: 0.6,
                      100: 0.8,
                      50: 0.95,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, 1.0, 2.0]

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
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = 1

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = True

#: Should the idler jj current be fixed accross gates
FIX_IDLER_CURRENT = True

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = {
                                350: 1.15,
                                200: 1,
                                50: 0.95,
                                }

MULTIPLE_TRANSPARENCIES = [0.01, 0.6, 0.95]

#: Path to which save the graphs and fitted parameters.
# ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
#                  'SQUID/By/active_t_fixed')
