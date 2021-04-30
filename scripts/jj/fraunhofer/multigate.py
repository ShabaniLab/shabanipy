"""Plot multigate Fraunhofers and guess corresponding current densities."""
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cs

from shabanipy.jj.fraunhofer.dynesfulton import (
    critical_current_density,
)
from shabanipy.jj.fraunhofer.generate_pattern import fraunhofer
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center, symmetrize_fraunhofer
from shabanipy.jj.utils import extract_switching_current
from shabanipy.utils.labber_io import LabberData

LABBER_DATA_DIR = os.environ["LABBER_DATA_DIR"]

# Fraunhofer data for multigates -1-2-3-4-5-
# Gates 2 and 4 are fixed at Vg2 = Vg4 = constant while gates 1, 3, and 5 are
# swept together.
# A Fraunhofer pattern is measured for 6 values of Vg1 = Vg3 = Vg5.
DATA_FILE_PATH = Path(LABBER_DATA_DIR) / "2019/11/Data_1104/JS123A_BM003_054.hdf5"

# channel names
CH_MAGNET = "Keithley Magnet 1 - Source current"
CH_BIAS = "Yoko 1 - Voltage"  # current bias
CH_GATE = "SRS - Aux 2 output"  # gate voltage (gates 1, 3, and 5)
CH_RESIST = "SRS - Value"

# conversion factors
CURR_TO_FIELD = 1e3 / 18.2  # coil current to B-field (in mT)
VOLT_TO_RESIST = 1 / 10e-9  # lock-in voltage to resistance (inverse current)

# constants
PHI0 = cs.h / (2 * cs.e)  # magnetic flux quantum
jj_width = 4e-6
jj_length = 1800e-9  # includes London penetration depths
b2beta = 2 * np.pi * jj_length / PHI0  # B-field to beta factor
fraunhofer_period = 2 * np.pi / (b2beta * jj_width)

resist = []
ic = []
with LabberData(str(DATA_FILE_PATH)) as f:
    channels = f.list_channels()

    field = np.unique(f.get_data(CH_MAGNET))[:-10] * CURR_TO_FIELD

    gate = np.unique(f.get_data(CH_GATE))
    for g in gate:
        bias = f.get_data(CH_BIAS, filters={CH_GATE: g})[:-10]
        resist.append(
            np.abs(
                np.real(
                    VOLT_TO_RESIST * f.get_data(CH_RESIST, filters={CH_GATE: g})[:-10]
                )
            )
        )
        ic.append(extract_switching_current(bias, resist[-1], 5, "positive"))
resist = np.array(resist)
ic = np.array(ic)
# bias sweeps should be the same for all gate values
bias_min, bias_max = np.min(bias), np.max(bias)

jss = [[0.05, 0.26, 0.1], [], [0.5, 0.26, 0.18], [], [0.9, 0.26, 0.4]]
gate_V_selection = [-1, 0, 1]
for gate_, resist_, ic_, js in zip(gate, resist, ic, jss):
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

    gen_ic = fraunhofer(field_ / 1e3, b2beta, jx, x)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r"$V_{g,odd}$ = " + f"{gate_}")
    ax.set_xlabel("Magnetic field (mT)")
    ax.set_ylabel("Bias current (µA)")
    ax.plot(field_, gen_ic * 1e6, label="generated")
    ax.plot(field_, ic_, label="data")
    ax.set_ylim(0, 1.6)
    ax.legend()

    x_data, jx_data = critical_current_density(
        field_ / 1e3, ic_, b2beta, jj_width, 200
    )
    x_gen, jx_gen = critical_current_density(
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
