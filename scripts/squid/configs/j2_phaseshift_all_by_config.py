#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/03'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
              400: 'Data_0316/JS124S_BM002_390.hdf5',
              350: 'Data_0317/JS124S_BM002_392.hdf5',
              300: 'Data_0318/JS124S_BM002_394.hdf5',
              250: 'Data_0318/JS124S_BM002_395.hdf5',
              200: 'Data_0318/JS124S_BM002_396.hdf5',
              150: 'Data_0319/JS124S_BM002_397.hdf5',
              100: 'Data_0321/JS124S_BM002_405.hdf5',
            # #   50:  'Data_0320/JS124S_BM002_402.hdf5',
              450: 'Data_0316/JS124S_BM002_389.hdf5',
              475: 'Data_0317/JS124S_BM002_393.hdf5',
              500: 'Data_0316/JS124S_BM002_388.hdf5',
              525: 'Data_0317/JS124S_BM002_391.hdf5',
              550: 'Data_0315/JS124S_BM002_386.hdf5',
              600: 'Data_0315/JS124S_BM002_387.hdf5',
              650: 'Data_0315/JS124S_BM002_385.hdf5',
            # #   700: 'Data_0322/JS124S_BM002_413.hdf5',  # wrong period
              750: 'Data_0314/JS124S_BM002_384.hdf5',
              800: 'Data_0323/JS124S_BM002_414.hdf5',
              850: 'Data_0314/JS124S_BM002_383.hdf5',
              900: 'Data_0323/JS124S_BM002_415.hdf5',
            # #   950: 'Data_0323/JS124S_BM002_416.hdf5',  # really ugly data
            #   1000: 'Data_0324/JS124S_BM002_422.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (-7.8e-3, -5.5e-3),
                350: (None, -6e-3),
                300: (-6.59e-3, -4.75e-3),
                250: (),
                200: (-5.7e-3, None),
                150: (-5.05e-3, None),
                100: (-3.9e-3, -1.1e-3),
                50: (-2.2e-3, None),
                450: (-7e-3, -5.5e-3),
                475: (-8.7e-3, -6.3e-3),
                500: (-8.9e-3, -7e-3),
                525: (None, -6.3e-3),
                550: (-9e-3, None),
                600: (-10e-3, -7.5e-3),
                650: (-10.1e-3, None),
                700: (),
                750: (-9.8e-3, -7.8e-3),
                800: (None, -1.3e-3),
                850: (None, -9e-3),
                900: (-0.3e-3, None),
                950: (-0.5e-3, None),
                1000: ()}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,
                      350: 0.1,
                      300: 0.2,
                      250: 0.3,
                      200: 0.4,
                      150: 0.6,
                      100: 0.8,
                      50: 0.85,
                      450: 0.01,
                      475: 0.01,
                      500: 0.01,
                      525: 0.01,
                      550: 0.01,
                      600: 0.01,
                      650: 0.01,
                      700: 0.01,
                      750: 0.01,
                      800: 0.01,
                      850: 0.01,
                      900: 0.01,
                      950: 0.01,
                      1000: 0.01,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5]

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
HANDEDNESS = -1

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = True

#: Should the idler jj current be fixed accross gates
FIX_IDLER_CURRENT = False

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = {400: 1.0,
                                350: 1.07,
                                300: 1.08,
                                250: 1.0,
                                200: 1.07,
                                150: 1.12,
                                100: 1.17,
                                50: 1.0,
                                450: 0.95,
                                475: 1.05,
                                500: 1.18,
                                525: 0.90,
                                550: 0.91,
                                600: 1.12,
                                650: 1.07,
                                700: 1.07,
                                750: 1.12,
                                800: 0.94,
                                850: 1.08,
                                900: 1.15,
                                950: 1.10,
                                1000: 1.05,
                                }

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_all_field/By_attempt_30-05-2019')
