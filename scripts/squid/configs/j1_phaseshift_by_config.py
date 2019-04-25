#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field, file path.
DATA_PATHS = {400: 'Data_0405/JS124S_BM002_465.hdf5',
              350: 'Data_0406/JS124S_BM002_466.hdf5',
              300: 'Data_0406/JS124S_BM002_467.hdf5',
              250: 'Data_0406/JS124S_BM002_468.hdf5',
              200: 'Data_0407/JS124S_BM002_470.hdf5',
              150: 'Data_0407/JS124S_BM002_471.hdf5',
              100: 'Data_0409/JS124S_BM002_474.hdf5',}
            #   50:  'Data_0409/JS124S_BM002_476.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (),
                350: (),
                300: (),
                250: (),
                200: (),
                150: (),
                100: (),
                50: ()}

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

#: Should we fix the transparency of the idler as a function of field.
FIX_IDLER_TRANSPARENCY = False

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/By/active_t_fixed')
