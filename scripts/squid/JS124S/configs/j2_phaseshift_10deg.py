#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
               200: '04/Data_0424/JS124S_BM002_513.hdf5',
            #    300: 'Data_0424/JS124S_BM002_512.hdf5',  # too small current range
               400: '04/Data_0423/JS124S_BM002_511.hdf5',
               500: '04/Data_0421/JS124S_BM002_509.hdf5',
               600: '04/Data_0419/JS124S_BM002_506.hdf5',
               700: '05/Data_0519/JS124S_BM002_596.hdf5',
               800: '05/Data_0520/JS124S_BM002_597.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {
                200: (None, -1.5e-3),
                300: (),
                400: (None, -0.002),
                500: (None, None),
                600: (None, None),
                700: (None, 9.2e-3),
                800: (None, 10.6e-3),
                }

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {
                      200: 0.7,
                      300: 0.4,
                      400: 0.1,
                      500: 0.1,
                      600: 0.1,
                      700: 0.1,
                      800: 0.1,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5, 3.0]

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

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Should the idler jj current be fixed accross gates
FIX_IDLER_CURRENT = True

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 0.98#{400: 0.9,
                                # 200: 1.05,
                                # 500: 0.9,
                                # 600: 1.0,
                                # 700: 1.0,
                                # 800: 1.05,
                                # }

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_all_field/10deg')
