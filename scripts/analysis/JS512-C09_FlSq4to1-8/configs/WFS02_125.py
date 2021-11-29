from shabanipy.labber import get_data_dir
from shabanipy.constants import VECTOR9_AMPS_PER_TESLA_X, VECTOR10_AMPS_PER_TESLA_X

# sample identification
WAFER = "JS512"
PIECE = "C09"
LAYOUT = "FlSq4to1-8"
DEVICE = "FlSq4to2"

# data identification
COOLDOWN_SCAN = "WFS02_125"
DATAPATH = (
    get_data_dir()
    / "2021/08/Data_0821"
    / f"{WAFER}-{PIECE}_{LAYOUT}_{DEVICE}_{COOLDOWN_SCAN}.hdf5"
)

# make sure we use the right magnet conversion factors
FRIDGE = "vector10"
assert FRIDGE in str(
    DATAPATH
), f"I can't double check that {DATAPATH} is from {FRIDGE}"  # depends on my local file structure
AMPS_PER_T = vars()[f"{FRIDGE.upper()}_AMPS_PER_TESLA_X"]

# names or indices of the data channels
CHAN_LOCKIN = "vicurve - dR vs I curve"
CHAN_FIELD_PERP = "X magnet - Source current"
CHAN_FIELD_PRLL = "Y magnet - Field"
CHAN_GATE_4UM = "4um gate - Source voltage"
CHAN_GATE_2UM = "2um gate - Source voltage"
# TODO add check for gate leakage
CHAN_GATE_4UM_MEAS = "4um gate - Current"
CHAN_GATE_2UM_MEAS = "2um gate - Current"
# TODO add check for temperature variation
CHAN_TEMP = "fridge - Temperature closed loop - target temperature"
CHAN_TEMP_MEAS = "fridge - MC-RuOx-Temperature"

# threshold used to determine switching current
RESISTANCE_THRESHOLD = 400

# handedness of the system; i.e. sign(bfield) * sign(phase)
HANDEDNESS = -1

# plot the initial guess along with the best fit
PLOT_GUESS = False

# directory and path prefix to save plots and fit parameters
OUTDIR = f"{__file__.split('.py')[0]}_results/"
OUTPATH = Path(OUTDIR) / COOLDOWN_SCAN
