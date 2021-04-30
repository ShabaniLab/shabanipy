"""Check the degradation of the reconstructed current after several round-trips.

Do: J(x) -> I_c(β) -> J(x) -> I_c(β) -> ... etc.
"""
import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.dynesfulton import (
    extract_current_distribution,
)
from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast

# enable import from current_profiles.py in this directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from current_profiles import j_uniform

# constants
PHI0 = 2e-15
JJ_WIDTH = 1e-6
JJ_LENGTH = 100e-9
IC0 = 1e-6
B2BETA = 2 * np.pi * JJ_LENGTH / PHI0
B_NODE = PHI0 / (JJ_LENGTH * JJ_WIDTH)
N_NODES = 5
PTS_PER_NODE = 4
N_POINTS = 2 * (PTS_PER_NODE * N_NODES)

# set up plots
mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["lines.marker"] = "."
mpl.rcParams["lines.markersize"] = 0
fig, ax = plt.subplots(constrained_layout=True)
ax.set_ylabel("J(x)")
fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.set_ylabel(r"$I_c(B)$")
cmap = mpl.cm.get_cmap("inferno")

# input current distribution
xs = np.linspace(-JJ_WIDTH, JJ_WIDTH, N_POINTS)
js = j_uniform(xs, ic0=IC0, jj_width=JJ_WIDTH)
ax.plot(xs, js, lw=1, label="input (0)", color=cmap(0))

# analytical Fraunhofer
fine_bs = np.linspace(-N_NODES * B_NODE, N_NODES * B_NODE, 10 * N_POINTS)
ax2.plot(
    fine_bs,
    IC0 * np.abs(np.sinc(fine_bs * B2BETA * JJ_WIDTH / 2 / np.pi)),
    lw=1,
    markersize=0,
    label="analytical",
    color=cmap(0),
)

bs = np.linspace(-N_NODES * B_NODE, N_NODES * B_NODE, N_POINTS)
TRIPS = 50
for i in range(TRIPS):
    # generate Fraunhofer
    ics = produce_fraunhofer_fast(bs, B2BETA, js, xs)
    ax2.plot(bs, ics, label=f"{2*i + 1}", color=cmap((2 * i + 1) / (2 * TRIPS)))

    # reconstruct current density
    xs, js = extract_current_distribution(bs, ics, B2BETA, JJ_WIDTH, len(xs) / 2)
    ax.plot(xs, js, label=f"{2*i + 2}", color=cmap((2 * i + 2) / (2 * TRIPS)))


# fig.legend()
# fig2.legend()
fig3, ax3 = plt.subplots(constrained_layout=True, figsize=[1, 5])
sm = mpl.cm.ScalarMappable(cmap=cmap)
sm.set_clim(0, TRIPS)
cbar = fig3.colorbar(sm, cax=ax3)
cbar.set_label("round trips")
plt.show()
