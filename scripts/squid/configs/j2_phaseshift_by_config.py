#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/03'

#: Dictionary of parallel field, file path.
DATA_PATHS = {400: 'Data_0316/JS124S_BM002_390.hdf5',
              350: 'Data_0317/JS124S_BM002_392.hdf5',
              300: 'Data_0318/JS124S_BM002_394.hdf5',
              250: 'Data_0318/JS124S_BM002_395.hdf5',
              200: 'Data_0318/JS124S_BM002_396.hdf5',
              150: 'Data_0319/JS124S_BM002_397.hdf5',
              100: 'Data_0321/JS124S_BM002_405.hdf5',
              50:  'Data_0320/JS124S_BM002_402.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (None, -5.5e-3),
                350: (None, -6e-3),
                300: (-6.59e-3, -4.75e-3),
                250: (),
                200: (),
                150: (-5.05e-3, None),
                100: (-3.9e-3, -1.1e-3),
                50: (-2.2e-3, None)}

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
PLOT_EXTRACTED_SWITCHING_CURRENT = True

#: Should we fix the transparency of the idler as a function of field.
FIX_IDLER_TRANSPARENCY = False

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = -1

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/phaseshift_low_field/j2/By/active_t_fixed')
