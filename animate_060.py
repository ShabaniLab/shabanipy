"""Animate current density distribution of Maryland multigate device.

Forked from:
    Repository: github.com/shabanilab/shabanipy
    Script:     scripts/jj/fraunhofer/analyze_md_multigate
    Commit SHA: 3a23cfdc32180e639e0d69bbec2185fd37f7f52c

Device ID: JS311_2HB-2JJ-5MGJJ-MD-001_MG2.
Scan ID: JS311-BHENL001-2JJ-2HB-5MGJJ-MG2-060.
Fridge: vector9

This scan contains Fraunhofer data for a linear multigate -1-2-3-4-5-
Gates 1 and 5 are grounded; gates 2 and 4 are shorted.
Both Vg3 and Vg2(=Vg4) are swept independently.
"""

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import constants as cs
from scipy.interpolate import interp1d

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

print('reconstructing current distributions...', end='', flush=True)
# 183 is the largest number of points returned by symmetrize_fraunhofer
# extract_current_distribution then returns max 183*2 = 366 points
POINTS = 366
x = np.empty(shape=ic.shape[:-1] + (POINTS,))
jx = np.empty(shape=ic.shape[:-1] + (POINTS,))
for i, g3 in enumerate(gate_3):
    for j, g24 in enumerate(gate_2_4):
        ic_ = ic[i, j]
        field_ = field - find_fraunhofer_center(field, ic_)
        field_, ic_ = symmetrize_fraunhofer(field_, ic_)
        x_, jx_ = extract_current_distribution(
            field_, ic_, FIELD_TO_WAVENUM, JJ_WIDTH, len(field_)
        )
        x[i, j] = np.pad(x_, (POINTS - len(x_)) // 2, mode="edge")
        jx[i, j] = np.pad(jx_, (POINTS - len(jx_)) // 2, mode="edge")
print('done', flush=True)

# There are 11x10 fraunhofers, 1 for each (Vg3, Vg2=Vg4) combination.
# Make 2 animations:
#     1. Plot all Vg2=Vg4 traces and sweep Vg3 with time;
#     2. Plot all Vg3 traces and sweep Vg2=Vg4 with time.

cmap = plt.get_cmap("inferno")
field = field * 1e3
ic = ic * 1e6
x = x * 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
ax1.set_xlabel(r"$B_\perp$ (mT)")
ax1.set_ylabel(r"$I_c$ (μA)")
ax2.set_xlabel(r"$x$ (μm)")
ax2.set_ylabel(r"$J(x)$ (μA/μm)")
ax1.set_ylim(0, 2.5)
ax2.set_ylim(-0.25, 1.75)

gate_3_fine = np.linspace(gate_3[0], gate_3[-1], len(gate_3) * 10)
interp_func = interp1d(gate_3, ic, axis=0)
ic_interp = interp_func(gate_3_fine)
interp_func = interp1d(gate_3, x, axis=0)
x_interp = interp_func(gate_3_fine)
interp_func = interp1d(gate_3, jx, axis=0)
jx_interp = interp_func(gate_3_fine)

lines_ic = ax1.plot(field, np.transpose(ic_interp[0]))
for l, line in enumerate(lines_ic):
    line.set_color(cmap(l / len(lines_ic)))
for k, g24 in enumerate(gate_2_4):
    ax2.plot(x_interp[0, k], jx_interp[0, k], color=cmap(k / len(gate_2_4)))
lines_jx = ax2.get_lines()

def update(frame_num, ic, lines_ic, x, jx, lines_jx):
    for l, line in enumerate(lines_ic):
        line.set_ydata(ic[frame_num, l])
    for l, line in enumerate(lines_jx):
        line.set_data(x_interp[frame_num, l], jx_interp[frame_num, l])
    return lines_ic + lines_jx

ani = animation.FuncAnimation(
    fig,
    update,
    frames=[i for i, _ in enumerate(gate_3_fine)]
    + [i for i, _ in reversed(list(enumerate(gate_3_fine)))],
    fargs=[ic_interp, lines_ic, x_interp, jx_interp, lines_jx],
    interval=20,
    blit=True,
)
print('saving gate_3.gif...', end='', flush=True)
ani.save('gate_3.gif')
print('done', flush=True)


for j, g24 in enumerate(gate_2_4):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r"$V_\mathrm{g1,g5} = 0$, $V_\mathrm{g2,g4} = $" + f"{g24} V")
    ax.set_xlabel(r"$B_\perp$ (mT)")
    ax.set_ylabel(r"$I_c$ (μA)")
    lines = ax.plot(field * 1e3, np.transpose(ic[:, j]) * 1e6)
    for l, line in enumerate(lines):
        line.set_color(cmap(l / len(lines)))
    lines[0].set_label(gate_3[0])
    lines[-1].set_label(gate_3[-1])
    ax.legend(title=r"$V_\mathrm{g3}$ (V)")
    fig.savefig(f"plots/061_fraunhofer_Vg24={g24}.pdf", format="pdf")
    plt.close(fig=fig)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r"$V_\mathrm{g1,g5} = 0$, $V_\mathrm{g2,g4} = $" + f"{g24} V")
    ax.set_xlabel(r"$x$ (μm)")
    ax.set_ylabel(r"$J(x)$ (μA/μm)")
    for i, g3 in enumerate(gate_3):
        ax.plot(x[i, j] * 1e6, jx[i, j], color=cmap(i / len(gate_3)))
    lines = ax.get_lines()
    lines[0].set_label(gate_3[0])
    lines[-1].set_label(gate_3[-1])
    ax.legend(title=r"$V_\mathrm{g3}$ (V)")
    fig.savefig(f"plots/061_current-density_Vg24={g24}.pdf", format="pdf")
    plt.close(fig=fig)
