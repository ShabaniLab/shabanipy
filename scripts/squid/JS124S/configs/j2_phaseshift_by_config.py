#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/'

#: Dictionary of parallel field, file path.
DATA_PATHS = {400: '03/Data_0316/JS124S_BM002_390.hdf5',
            #   350: '03/Data_0317/JS124S_BM002_392.hdf5',
            #   300: '03/Data_0318/JS124S_BM002_394.hdf5',
               250: '03/Data_0318/JS124S_BM002_395.hdf5',
            #   200: '03/Data_0318/JS124S_BM002_396.hdf5',
            #   150: '03/Data_0319/JS124S_BM002_397.hdf5',
              100: '03/Data_0321/JS124S_BM002_405.hdf5',
            #   50:  '03/Data_0320/JS124S_BM002_402.hdf5',
              -300: '04/Data_0430/JS124S_BM002_532.hdf5',
               }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (-8e-3, -5.5e-3),
                350: (None, -6e-3),
                300: (-6.59e-3, -4.75e-3),
                250: (),
                200: (),
                150: (-5.05e-3, None),
                100: (-3.9e-3, -1.1e-3),
                50: (-2.2e-3, None),
                -300: (-1e-3, 1.2e-3),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,
                      350: 0.1,
                      300: 0.2,
                      250: 0.3,
                      200: 0.4,
                      150: 0.6,
                      100: 0.8,
                      -300: 0.2,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5, -2.0, -1.0, 1.0, 2.0, 3.0]

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
CONVERSION_FACTOR_CORRECTION = 1.07

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
# ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
#                  'SQUID/phaseshift_low_field/j2/By/active_t_fixed')
