#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
              90: '04/Data_0412/JS124S_BM002_487.hdf5',
              60: '04/Data_0413/JS124S_BM002_491.hdf5',
              30: '04/Data_0414/JS124S_BM002_493.hdf5',
            #    52.5: '04/Data_0415/JS124S_BM002_495.hdf5',
            #   45: '04/Data_0414/JS124S_BM002_494.hdf5',
               37.5: '04/Data_0416/JS124S_BM002_499.hdf5',
            #    15: '04/Data_0417/JS124S_BM002_500.hdf5',
               75: '04/Data_0417/JS124S_BM002_503.hdf5',
            #   0: '03/Data_0318/JS124S_BM002_394.hdf5',
              }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {90: (-9.5e-3, -8e-3),
                60: (-7e-3, -4.5e-3),
                30: (-3e-3, -1.3e-3),
                52.5: (-9.5e-3, -7.2e-3),
                45: (None, -5.7e-3),
                37.5: (-6e-3, -3.5e-3),
                15: (-4.5e-3, -2.5e-3),
                75: (-6e-3, -4e-3),
                0: (-6.59e-3, -4.75e-3)}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {90: 0.2,
                      60: 0.2,
                      30: 0.2,
                      52.5: 0.2,
                      45: 0.2,
                      37.5: 0.2,
                      15: 0.2,
                      75: 0.2,
                      0: 0.2,
                      }

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-2.0, -1.0, 1.0, 2.0, 3.0]

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

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 1.05#{90: 1.02,
                                # 60: 1.15,
                                # 30: 1.17,
                                # 52.5: 0.98,
                                # 45: 1.03,
                                # 37.5: 1.13,
                                # 15: 0.88,
                                # 75: 0.92,
                                # 0: 1.12,
                                # }

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Should the idler jj current be fixed across gates
FIX_IDLER_CURRENT = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_low_field/j2/angle_set2')
