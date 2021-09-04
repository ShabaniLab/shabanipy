#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field: pair of file path. The first path should refer
#: to the dataset in which the j1 junction is gated, the second to the dataset
#: in which the j2 junction is gated.
DATA_PATHS = {250: 'Data_0428/JS124S_BM002_524.hdf5',
            #   150: [04/Data_0429/JS124S_BM002_526.hdf5',
              -250: 'Data_0429/JS124S_BM002_529.hdf5',
              }

#: Perpendicular field range to fit for each parallel field.
FIELD_RANGES = {250: (-8.2e-3, -6e-3),
                150: (),
                -250: (5.6e-3, None),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {
                      250: 0.3,
                      150: 0.6,
                      -250: 0.3,
                      }

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

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 0.98#{250: 0.98, -250: 0.95}

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
# ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
#                  'SQUID/By/active_t_fixed')
