#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/05'

#: Dictionary of parallel field, file path.
DATA_PATHS = {-800: 'Data_0523/JS124S_BM002_605.hdf5',
              -700: 'Data_0520/JS124S_BM002_598.hdf5',
              -600: 'Data_0521/JS124S_BM002_600.hdf5',
              -500: 'Data_0521/JS124S_BM002_601.hdf5',
              -400: 'Data_0522/JS124S_BM002_602.hdf5',
              -200: 'Data_0522/JS124S_BM002_603.hdf5',
              -100: 'Data_0522/JS124S_BM002_604.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {
                -800: (),
                -700: (None,  -6.9e-3),
                -600: (),
                -500: (),
                -400: (),
                -200: (),
                -100: (),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {
                      -800: 0.1,
                      -700: 0.1,
                      -600: 0.1,
                      -500: 0.1,
                      -400: 0.2,
                      -200: 0.6,
                      -100: 0.8,
                     }

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

#: Enforce equality of the transparencies
EQUAL_TRANSPARENCIES = True

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 1.0

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_all_field/By_neg')
