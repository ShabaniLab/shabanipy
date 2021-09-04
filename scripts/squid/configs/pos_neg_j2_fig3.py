#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/05'

#: Dictionary of parallel field, file path.
DATA_PATHS = {
    # -300: 'Data_0513/JS124S_BM002_575.hdf5',
    50: 'Data_0516/JS124S_BM002_587.hdf5',
    200: 'Data_0515/JS124S_BM002_583.hdf5',
    # 350: 'Data_0514/JS124S_BM002_582.hdf5',
    }

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {-300: (-4.8e-3, None),
                50: (-1.97e-3, 0.0),
                200: (-0.8e-3, None),
                350: (0.55e-3, None)}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {
    -300: 0.2,
    50: 0.9,
    200: 0.7,
    350: 0.2,
    }

#: Name/index of the gate column.
GATE_COLUMN = 2

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 3

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/index of the counter current column used to determine the sweep
#: direction.
COUNTER_COLUMN = 1

#: Branch of the SQUID current measured for a given counter value
COUNTER_MEANING = {1.0: 'positive', -1.0: 'negative'}

#: Are data symmetric in bias or did we acquire only the interesting half
SYMMETRIC_DATA = {
    -300: True,
    50: False,
    200: False,
    350: False
    }

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 4

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = True

#: Should we enforce the equality of the transparencies.
EQUAL_TRANSPARENCIES = True

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 0.95

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Should the idler jj current be fixed across gates
FIX_IDLER_CURRENT = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = True

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = '/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/SQUID/phaseshift_low_field/symmetric_j2'