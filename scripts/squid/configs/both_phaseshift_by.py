#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019'

#: Dictionary of parallel field: pair of file path. The first path should refer
#: to the dataset in which the j1 junction is gated, the second to the dataset
#: in which the j2 junction is gated.
DATA_PATHS = {400: ['04/Data_0405/JS124S_BM002_465.hdf5',
                    '03/Data_0316/JS124S_BM002_390.hdf5'],
            #   350: ['04/Data_0406/JS124S_BM002_466.hdf5',
            #         '03/Data_0317/JS124S_BM002_392.hdf5'],
            #   300: ['04/Data_0406/JS124S_BM002_467.hdf5',
            #         '03/Data_0318/JS124S_BM002_394.hdf5'],
            #   250: ['04/Data_0406/JS124S_BM002_468.hdf5',
            #         '03/Data_0318/JS124S_BM002_395.hdf5'],
            #   200: ['04/Data_0407/JS124S_BM002_470.hdf5',
            #         '03/Data_0318/JS124S_BM002_396.hdf5'],
            #   150: ['04/Data_0407/JS124S_BM002_471.hdf5',
            #         '03/Data_0319/JS124S_BM002_397.hdf5'],
              100: ['04/Data_0409/JS124S_BM002_474.hdf5',
                    '03/Data_0321/JS124S_BM002_405.hdf5']}

#: Perpendicular field range to fit for each parallel field.
FIELD_RANGES = {400: [(), (-8e-3, -5.5e-3)],
                350: [(None, 0.2e-3), (None, -6e-3)],
                300: [(), (-6.59e-3, -4.75e-3)],
                250: [(None, 0.2e-3), ()],
                200: [(), ()],
                150: [(), (-4.7e-3, None)],
                100: [(None, 0.2e-3), (-3.9e-3, -1.1e-3)]}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,
                      350: 0.1,
                      300: 0.2,
                      250: 0.3,
                      200: 0.4,
                      150: 0.6,
                      100: 0.8}

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
CONVERSION_FACTOR_CORRECTION = (1.03, 1.07)

#: Fix the anomalous phase to 0.
FIX_PHI_ZERO = False

#: Enforce equality of the transparencies
EQUAL_TRANSPARENCIES = False

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
#: Recognized values are False, True, 'color' (to plot over the colormap)
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ''#('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 #'SQUID/phaseshift_low_field/combined/By/equal_transparencies')