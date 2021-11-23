#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/05'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
                50: 'Data_0514/JS124S_BM002_580.hdf5',
               200: 'Data_0514/JS124S_BM002_578.hdf5',
               350: 'Data_0514/JS124S_BM002_581.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {50: (-3.7e-3, -1.4e-3),
                200: (-1.3e-3, 0.9e-3),
                350: (-0.3e-3, 2e-3),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {50: 0.9,
                      200: 0.5,
                      350: 0.1,
                      }

#: Name/index of the gate column.
GATE_COLUMN = None

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 1

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 2

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

#: Correction factor to apply on the estimated pulsation 0.95
CONVERSION_FACTOR_CORRECTION = {50: 0.95, 200: 1.0, 350: 0.98}

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Path to which save the graphs and fitted parameters.
# ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
#                  'SQUID/phaseshift_low_field/j2/By/active_t_fixed')
