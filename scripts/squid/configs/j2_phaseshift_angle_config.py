#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field, file path.
DATA_PATHS = {90: 'Data_0412/JS124S_BM002_487.hdf5',
              60: 'Data_0413/JS124S_BM002_491.hdf5',
              30: 'Data_0414/JS124S_BM002_493.hdf5',
              52.5: 'Data_0415/JS124S_BM002_495.hdf5',
              45: 'Data_0414/JS124S_BM002_494.hdf5',
              37.5: 'Data_0416/JS124S_BM002_499.hdf5',
              15: 'Data_0417/JS124S_BM002_500.hdf5',
              75:  'Data_0417/JS124S_BM002_503.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {90: (-10e-3, -8e-3),
                60: (-7e-3, -4.5e-3),
                30: (-3e-3, -1e-3),
                52.5: (-9.5e-3, -6.5e-3),
                45: (None, -5e-3),
                37.5: (-6e-3, -3.5e-3),
                15: (-4.9e-3, -2.5e-3),
                75: (-6e-3, None)}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

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
                 'SQUID/phaseshift_low_field/j2/angle_300mT')
