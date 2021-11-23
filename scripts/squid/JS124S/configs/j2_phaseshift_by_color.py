#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
                # 50: '04/Data_0401/JS124S_BM002_450.hdf5',
            #    200: '04/Data_0424/JS124S_BM002_515.hdf5',
            #    201: '03/Data_0318/JS124S_BM002_396.hdf5',
               350: '03/Data_0329/JS124S_BM002_445.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {50: (None, 1.3e-3),
                200: (None, -0.5e-3),
                201: (),
                350: (-.5e-3, None),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {50: 0.9,
                      200: 0.4,
                      201: 0.4,
                      350: 0.01,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5, 2.0, 3.0]

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
CONVERSION_FACTOR_CORRECTION = 1.05
# 1.1 for 50, 1.0 for 200, 1.05 for 350

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Path to which save the graphs and fitted parameters.
# ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
#                  'SQUID/phaseshift_low_field/j2/By/active_t_fixed')
