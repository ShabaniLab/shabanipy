"""Reconstruct current density distribution of Maryland multigate device.

Device ID: JS311_2HB-2JJ-5MGJJ-MD-001_MG2.
Scan ID: JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-060.
Fridge: vector9

This scan contains Fraunhofer data for a linear multigate -1-2-3-4-5-
Gates 1 and 5 are grounded; gates 2 and 4 are shorted.
Both Vg3 and Vg2(=Vg4) are swept independently.
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cs

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
)
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center, symmetrize_fraunhofer
from shabanipy.jj.utils import extract_switching_current
from shabanipy.utils.labber_io import LabberData

LABBER_DATA_DIR = os.environ["LABBER_DATA_DIR"]
DATA_FILE_PATH = (
    Path(LABBER_DATA_DIR)
    / "2020/12/Data_1202/JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-060.hdf5"
)

# channel names
CH_GATE_3 = "SM1 - Source voltage"
CH_GATE_2_4 = "SM2 - Source voltage"
CH_MAGNET = "Magnet Source - Source current"
CH_RESIST = "VITracer - VI curve"

# Coil current to B-field conversion factor.
# The new sample holder is perpendicular to the old one;
# the conversion factor along the new axis is 30mA to 1mT.
CURR_TO_FIELD = 1 / 30

# constants
PHI0 = cs.h / (2 * cs.e)  # magnetic flux quantum
JJ_WIDTH = 4e-6
# The effective junction length is largely unknown due to thin-film penetration depth
# and flux focusing effects; nominally 100nm.
JJ_LENGTH = 1200e-9
FIELD_TO_WAVENUM = 2 * np.pi * JJ_LENGTH / PHI0  # B-field to beta wavenumber
PERIOD = 2 * np.pi / (FIELD_TO_WAVENUM * JJ_WIDTH)

with LabberData(DATA_FILE_PATH) as f:
    # NOTE: The use of np.unique assumes the gate, field, and bias values are identical
    # for each sweep. This is true for the current datafile but may not hold in general.
    # NOTE: Also, in this case we have to manually correct some Labber shenanigans by
    # flipping some data.
    gate_3 = np.flip(np.unique(f.get_data(CH_GATE_3)))
    gate_2_4 = np.flip(np.unique(f.get_data(CH_GATE_2_4)))
    field = np.unique(f.get_data(CH_MAGNET)) * CURR_TO_FIELD

    # Bias current from the custom Labber driver VICurveTracer isn't available via
    # LabberData methods.
    bias = np.unique(f._file["/Traces/VITracer - VI curve"][:, 1, :])

    resist = f.get_data(CH_RESIST)

# extract_switching_current chokes on 1D arrays. Construct the ndarray of bias sweeps
# for each (gate, field) to match the shape of the resistance ndarray
ic = extract_switching_current(
    np.tile(bias, resist.shape[:-1] + (1,)), resist, threshold=2.96e-3,
)

# NOTE: Here, every other fraunhofer appears flipped horizontally (i.e.  field B -> -B)
# when compared to Labber's Log Viewer.  However, Labber's Log Viewer shows a field
# offset that systematically changes sign on every other fraunhofer. This suggests that
# the Log Viewer incorrectly flips every other fraunhofer.  To recover the data as
# viewed in Log Viewer, uncomment this line.  The fraunhofers are centered and
# symmetrized before current reconstruction, so it shouldn't matter.
# ic[:, 1::2, :] = np.flip(ic[:, 1::2, :], axis=-1)

# There are 11x10 fraunhofers, 1 for each of the 11 Vg3 and 10 Vg2(=Vg4) values.
# Make 21 plots by fixing Vg3 and sweeping over Vg2, and vice versa.
for i, g3 in enumerate(gate_3):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r"$V_\mathrm{g1,g5} = 0$, $V_\mathrm{g3} = $" + f"{g3} V")
    ax.set_xlabel(r"$B_\perp$ (mT)")
    ax.set_ylabel(r"$I_c$ (μA)")
    lines = ax.plot(field * 1e3, np.transpose(ic[i]) * 1e6)
    cmap = plt.get_cmap("inferno")
    for i, line in enumerate(lines):
        line.set_color(cmap(i / len(lines)))
    ax.legend(gate_2_4, title=r"$V_\mathrm{g2,g4}$ (V)")

sys.exit()

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title("JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-060")
ax.set_xlabel(r"$x$ (μm)")
ax.set_ylabel(r"$J(x)$ (μA/μm)")

# save current reconstructions for further analysis
# x = np.empty(shape=ic.shape)
# jx = np.empty(shape=ic.shape)
for idx, gate_ in enumerate(gate):
    ic_ = ic[idx]
    field_ = field - find_fraunhofer_center(field, ic_)
    field_, ic_ = symmetrize_fraunhofer(field_, ic_)
    x_, jx_ = extract_current_distribution(
        field_, ic_, FIELD_TO_WAVENUM, JJ_WIDTH, len(field_)
    )
    ax.plot(x_ * 1e6, jx_, color=cmap(idx / len(gate)))

lines = ax.get_lines()
lines[0].set_label(gate[0])
lines[-1].set_label(gate[-1])
ax.legend(title=r"$V_\mathrm{g2,g4}$ (V)")
plt.show()
