"""Test DFT implementation."""
import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.dynesfulton import (
    critical_current_density,
)
from shabanipy.jj.fraunhofer.generate_pattern import _produce_fraunhofer_dft

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
mpl.rcParams["lines.markersize"] = 5
fig, ax = plt.subplots(constrained_layout=True)
ax.set_ylabel("J(x)")
fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.set_ylabel(r"$I_c(B)$")

# input current distribution
xs = np.linspace(-JJ_WIDTH, JJ_WIDTH, N_POINTS)
js = j_uniform(xs, ic0=IC0, jj_width=JJ_WIDTH)
ax.plot(xs, js, label="input")

# analytical Fraunhofer
# fine_bs = np.linspace(-N_NODES * B_NODE, N_NODES * B_NODE, 10 * N_POINTS)
# ax2.plot(
#    fine_bs,
#    IC0 * np.abs(np.sinc(fine_bs * B2BETA * JJ_WIDTH / 2 / np.pi)),
#    lw=1,
#    markersize=0,
#    label="analytical",
#    color=cmap(0),
# )

# generate Fraunhofer
ics, bs = _produce_fraunhofer_dft(js, abs(xs[0] - xs[1]), B2BETA)
ax2.plot(bs, ics, label="generated")

# reconstruct current density
# xs, js = critical_current_density(bs, ics, B2BETA, JJ_WIDTH, len(xs) / 2)
# ax.plot(js)

fig.legend()
fig2.legend()
plt.show()
