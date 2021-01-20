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
from matplotlib.patches import Rectangle
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

########################
# V_g3 sweep animation #
########################

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), gridspec_kw={'width_ratios': [1, 1, 0.5]}, constrained_layout=True)
ax1.set_xlabel(r"$B_\perp$ (mT)")
ax1.set_ylabel(r"$I_c$ (μA)")
ax2.set_xlabel(r"$x$ (μm)")
ax2.set_ylabel(r"$J(x)$ (μA/μm)")
ax1.set_ylim(0, 2.5)
ax2.set_ylim(-0.25, 1.75)

# interpolate along V_g3 to smooth frame transitions
gate_3_fine = np.linspace(gate_3[0], gate_3[-1], len(gate_3) * 10)
interp_func = interp1d(gate_3, ic, axis=0)
ic_interp = interp_func(gate_3_fine)
interp_func = interp1d(gate_3, x, axis=0)
x_interp = interp_func(gate_3_fine)
interp_func = interp1d(gate_3, jx, axis=0)
jx_interp = interp_func(gate_3_fine)

# plot initial frame for V_g3 fraunhofer and current distribution
lines_ic = ax1.plot(field, np.transpose(ic_interp[0]))
for l, line in enumerate(lines_ic):
    line.set_color(cmap((l + 1) / len(lines_ic)))
for k, g24 in enumerate(gate_2_4):
    ax2.plot(x_interp[0, k], jx_interp[0, k], color=cmap((k + 1) / len(gate_2_4)))
lines_jx = ax2.get_lines()

# plot the multigate schematic and gate voltage legends for V_g3 sweep
ax3.imshow(plt.imread('./multigateJJ.png'))
ax3.set_axis_off()
ax3.set_anchor('S')
ax_g2 = fig.add_axes([0, 0, 0, 0], label='gate2_cbar')
ax_g3 = fig.add_axes([0, 0, 0, 0], label='gate3_cbar')
ax_g4 = fig.add_axes([0, 0, 0, 0], label='gate4_cbar')
for ax, num in zip([ax_g2, ax_g3, ax_g4], [2, 3, 4]):
    ax.set_title(r'$V_\mathrm{' + f'g{num}' + r'}$')
    ax.xaxis.set_visible(False)
    ax.set_anchor('SW')
for ax in [ax_g2, ax_g4]:
    ax.imshow(np.transpose([np.flip(gate_2_4)]), cmap=cmap, aspect=1 / 2)
    ax.set_yticks(np.arange(len(gate_2_4)))
    ax.set_yticklabels(gate_2_4)
    ax.tick_params(length=0, pad=-22, colors='white')
    for ticklabel in ax.yaxis.get_majorticklabels()[-2:]:
        ticklabel.set_color('black')
ax_g3.set_xlim((0, 1))
ax_g3.set_ylim((np.min(gate_3), np.max(gate_3)))
ax_g3.set_yticks([np.min(gate_3) + 0.2, np.max(gate_3) - 0.3])
ax_g3.set_yticklabels([np.min(gate_3), np.max(gate_3)])
ax_g3.tick_params(length=0, pad=-22)
ticklabels = ax_g3.yaxis.get_majorticklabels()
ticklabels[0].set_color('black')
ticklabels[1].set_color('white')
rect_g3 = Rectangle(xy=(0, 0), width=1, height=gate_3_fine[0], color='black')
ax_g3.add_patch(rect_g3)
ax3_bbox = ax3.get_position()
ax_g2.set_position([ax3_bbox.x0 + 0.099, ax3_bbox.y0 + 0.4, 0.026, 0.4])
ax_g3.set_position([ax3_bbox.x0 + 0.134, ax3_bbox.y0 + 0.4, 0.026, 0.4])
ax_g4.set_position([ax3_bbox.x0 + 0.1705, ax3_bbox.y0 + 0.4, 0.026, 0.4])

def update(frame_num, ic, lines_ic, x, jx, lines_jx):
    for l, line in enumerate(lines_ic):
        line.set_ydata(ic[frame_num, l])
    for l, line in enumerate(lines_jx):
        line.set_data(x_interp[frame_num, l], jx_interp[frame_num, l])
    rect_g3.set_height(gate_3_fine[frame_num])
    return lines_ic + lines_jx + [rect_g3]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=[i for i, _ in enumerate(gate_3_fine)]
    + [i for i, _ in reversed(list(enumerate(gate_3_fine)))],
    fargs=[ic_interp, lines_ic, x_interp, jx_interp, lines_jx],
    interval=20,
    blit=True,
)
# showing before saving irons out some positioning issues
plt.show()
print('saving gate_3_new.mp4...', end='', flush=True)
ani.save('gate_3_new.mp4', dpi=200)
print('done', flush=True)
plt.close(fig)

#############################
# V_g2=V_g4 sweep animation #
#############################

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), gridspec_kw={'width_ratios': [1, 1, 0.5]}, constrained_layout=True)
ax1.set_xlabel(r"$B_\perp$ (mT)")
ax1.set_ylabel(r"$I_c$ (μA)")
ax2.set_xlabel(r"$x$ (μm)")
ax2.set_ylabel(r"$J(x)$ (μA/μm)")
ax1.set_ylim(0, 2.5)
ax2.set_ylim(-0.25, 1.75)

# interpolate along V_g2(=V_g4) to smooth frame transitions
gate_2_4_fine = np.linspace(gate_2_4[0], gate_2_4[-1], len(gate_2_4) * 10)
interp_func = interp1d(gate_2_4, ic, axis=1)
ic_interp = interp_func(gate_2_4_fine)
interp_func = interp1d(gate_2_4, x, axis=1)
x_interp = interp_func(gate_2_4_fine)
interp_func = interp1d(gate_2_4, jx, axis=1)
jx_interp = interp_func(gate_2_4_fine)

# plot initial frame for V_g2(=V_g4) fraunhofer and current distribution
lines_ic = ax1.plot(field, np.transpose(ic_interp[:, 0]))
for l, line in enumerate(lines_ic):
    line.set_color(cmap((l+1) / len(lines_ic)))
for k, g3 in enumerate(gate_3):
    ax2.plot(x_interp[k, 0], jx_interp[k, 0], color=cmap((k+1) / len(gate_3)))
lines_jx = ax2.get_lines()

# plot the multigate schematic and gate voltage legends for V_g2(=V_g4) sweep
ax3.imshow(plt.imread('./multigateJJ.png'))
ax3.set_axis_off()
ax3.set_anchor('S')
ax_g2 = fig.add_axes([0, 0, 0, 0], label='gate2_cbar')
ax_g3 = fig.add_axes([0, 0, 0, 0], label='gate3_cbar')
ax_g4 = fig.add_axes([0, 0, 0, 0], label='gate4_cbar')
for ax, num in zip([ax_g2, ax_g3, ax_g4], [2, 3, 4]):
    ax.set_title(r'$V_\mathrm{' + f'g{num}' + r'}$')
    ax.xaxis.set_visible(False)
    ax.set_anchor('SW')
ax_g3.imshow(np.transpose([np.flip(gate_3)]), cmap=cmap, aspect=1 / 2.2)
ax_g3.set_yticks(np.arange(len(gate_3)))
ax_g3.set_yticklabels(gate_3)
ax_g3.tick_params(length=0, pad=-21, colors='white')
for ticklabel in ax_g3.yaxis.get_majorticklabels()[-2:]:
    ticklabel.set_color('black')
for ax in [ax_g2, ax_g4]:
    ax.set_xlim((0, 1))
    ax.set_ylim((np.min(gate_2_4), np.max(gate_2_4)))
    ax.set_yticks([np.min(gate_2_4) + 0.3, np.max(gate_2_4) - 0.4])
    ax.set_yticklabels([np.min(gate_2_4), np.max(gate_2_4)])
    ax.tick_params(length=0, pad=-23)
    ticklabels = ax.yaxis.get_majorticklabels()
    ticklabels[0].set_color('black')
    ticklabels[1].set_color('white')
rect_g2 = Rectangle(xy=(0, 0), width=1, height=gate_2_4_fine[0], color='black')
rect_g4 = Rectangle(xy=(0, 0), width=1, height=gate_2_4_fine[0], color='black')
ax_g2.add_patch(rect_g2)
ax_g4.add_patch(rect_g4)
ax3_bbox = ax3.get_position()
ax_g2.set_position([ax3_bbox.x0 + 0.099, ax3_bbox.y0 + 0.4, 0.026, 0.4])
ax_g3.set_position([ax3_bbox.x0 + 0.134, ax3_bbox.y0 + 0.4, 0.026, 0.4])
ax_g4.set_position([ax3_bbox.x0 + 0.1705, ax3_bbox.y0 + 0.4, 0.026, 0.4])

def update(frame_num, ic, lines_ic, x, jx, lines_jx):
    for l, line in enumerate(lines_ic):
        line.set_ydata(ic[l, frame_num])
    for l, line in enumerate(lines_jx):
        line.set_data(x_interp[l, frame_num], jx_interp[l, frame_num])
    rect_g2.set_height(gate_2_4_fine[frame_num])
    rect_g4.set_height(gate_2_4_fine[frame_num])
    return lines_ic + lines_jx + [rect_g2, rect_g4]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=[i for i, _ in enumerate(gate_2_4_fine)]
    + [i for i, _ in reversed(list(enumerate(gate_2_4_fine)))],
    fargs=[ic_interp, lines_ic, x_interp, jx_interp, lines_jx],
    interval=20,
    blit=True,
)
# showing before saving irons out some positioning issues
plt.show()
print('saving gate_2_4_new.mp4...', end='', flush=True)
ani.save('gate_2_4_new.mp4', dpi=200)
print('done', flush=True)
