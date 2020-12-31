"""Reconstruct current density distribution of Maryland multigate device.

Device ID: JS311_2HB-2JJ-5MGJJ-MD-001_MG2.
Scan ID: JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-051.
Fridge: vector9

This scan contains Fraunhofer data for a linear multigate -1-2-3-4-5-
Gates 1, 3, 5 are grounded; gates 2 and 4 are shorted.
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
from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center, symmetrize_fraunhofer
from shabanipy.jj.utils import extract_switching_current
from shabanipy.utils.labber_io import LabberData

LABBER_DATA_DIR = os.environ["LABBER_DATA_DIR"]
DATA_FILE_PATH = (
    Path(LABBER_DATA_DIR)
    / "2020/12/Data_1202/JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-051.hdf5"
)

# channel names
CH_GATE = "SM2 - Source voltage"
CH_MAGNET = "Magnet Source - Source current"
CH_RESIST = "VITracer - VI curve"

# coil current to B-field conversion factor
# the new sample holder is perpendicular to the old one; the
# conversion factor along the new axis is 30mA to 1mT
CURR_TO_FIELD = 1 / 30

# constants
#####PHI0 = cs.h / (2 * cs.e)  # magnetic flux quantum
#####jj_width = 4e-6
#####jj_length = 1800e-9  # includes London penetration depths
#####b2beta = 2 * np.pi * jj_length / PHI0  # B-field to beta factor
#####fraunhofer_period = 2 * np.pi / (b2beta * jj_width)


with LabberData(DATA_FILE_PATH) as f:
    # NOTE: The use of np.unique assumes the gate, field, and
    # bias values are identical for each sweep. This is true
    # for the current datafile but may not hold in general.
    gate = np.unique(f.get_data(CH_GATE))
    field = np.unique(f.get_data(CH_MAGNET)) * CURR_TO_FIELD

    # bias current from the custom Labber driver VICurveTracer isn't available
    # via LabberData methods
    bias = np.unique(f._file["/Traces/VITracer - VI curve"][:, 1, :])

    # TODO convert DMM volts to ohms
    resist = f.get_data(CH_RESIST)

# extract_switching_current chokes on 1D arrays, construct the ndarray of bias
# sweeps for each (gate, field)
bias = np.tile(bias, resist.shape[:-1] + (1,))
ic = extract_switching_current(bias, resist, threshold=2.75e-3)

fig, ax = plt.subplots()
ax.set_xlabel(r"$B_\perp$ (mT)")
ax.set_ylabel(r"$I_c$ (μA)")
lines = ax.plot(field * 1e3, np.transpose(ic) * 1e6)
cmap = plt.get_cmap("inferno")
for i, line in enumerate(lines):
    line.set_color(cmap(i / len(lines)))
lines[0].set_label(gate[0])
lines[-1].set_label(gate[-1])
ax.legend(title="gate voltage (V)")
plt.show()

# TODO you are here
sys.exit()

# bias sweeps are the same for all gate values
bias_min, bias_max = np.min(bias), np.max(bias)

jss = [[0.05, 0.26, 0.1], [], [0.5, 0.26, 0.18], [], [0.9, 0.26, 0.4]]
gate_V_selection = [-1, 0, 1]
for gate_, xresistx_, ic_, js in zip(gate, xresistx, ic, jss):
    if gate_ not in gate_V_selection:
        continue

    field_ = field - find_fraunhofer_center(field, ic_)
    field_, ic_ = symmetrize_fraunhofer(field_, ic_)

    # zero-padded uniform current distributions under variable gate widths
    x = np.linspace(-4 * jj_width, 4 * jj_width, 513)
    jx = np.zeros_like(x)
    mask = np.abs(x) < jj_width / 2
    jx[mask] = js[0]
    mask = np.logical_and(mask, np.abs(x) < jj_width / 2.3)
    jx[mask] = js[1]
    mask = np.logical_and(mask, np.abs(x) < jj_width / 8)
    jx[mask] = js[2]

    gen_ic = produce_fraunhofer_fast(field_ / 1e3, b2beta, jx, x)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r"$V_{g,odd}$ = " + f"{gate_}")
    ax.set_xlabel("Magnetic field (mT)")
    ax.set_ylabel("Bias current (µA)")
    ax.plot(field_, gen_ic * 1e6, label="generated")
    ax.plot(field_, ic_, label="data")
    ax.set_ylim(0, 1.6)
    ax.legend()

    x_data, jx_data = extract_current_distribution(
        field_ / 1e3, ic_, b2beta, jj_width, 200
    )
    x_gen, jx_gen = extract_current_distribution(
        field_ / 1e3, gen_ic, b2beta, jj_width, 200
    )

    fig2, ax2 = plt.subplots(constrained_layout=True)
    ax2.set_title(r"$V_{g,odd}$ = " + f"{gate_}")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("Current density (µA/µm)")
    ax2.plot(x[175:-175] * 1e6, jx[175:-175], color="black", label="manually optimized")
    ax2.plot(x_gen * 1e6, jx_gen, label="from generated")
    ax2.plot(x_data * 1e6, jx_data / 1e6, label="from data")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-0.15, 1)
    ax2.legend()

plt.show()
