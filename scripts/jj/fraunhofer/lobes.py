"""Investigate how Fraunhofer lobes impact reconstructed current density.

In particular, how do
    1) the number of side lobes (i.e. B-field range), and
    2) the number of points per lobe (i.e. B-field resolution)
in the Fraunhofer pattern I_c(B) impact the fidelity of a reconstructed uniform
current density J(x).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
        extract_current_distribution)


PHI0 = 2e-15
JJ_WIDTH = 1e-6
JJ_LENGTH = 100e-9
B2BETA = 2 * np.pi * JJ_LENGTH / PHI0
B_NODE = PHI0 / (JJ_WIDTH * JJ_LENGTH)
IC0 = 1e-6

def j_true(x):
    """Uniform current density with which to compare output."""
    return np.piecewise(x,
            [np.abs(x) < JJ_WIDTH / 2, np.abs(x) >= JJ_WIDTH / 2], [1, 0])

node_start, node_stop, node_step = 1, 12, 1  # number of nodes
ppl_start, ppl_stop, ppl_step = 5, 55, 5    # points per lobe
n_nodes = np.arange(node_start, node_stop, node_step)
n_ppls = np.arange(ppl_start, ppl_stop, ppl_step)
fidelity = np.empty(shape=(len(n_nodes), len(n_ppls)))

fig, ax = plt.subplots()
ax.set_xlabel('x (m)')
ax.set_ylabel('J(x) (A/m)')
x = np.linspace(-JJ_WIDTH, JJ_WIDTH, 200)
ax.plot(x, j_true(x))

# choose a few reconstructions to plot (n_node, n_ppl)
should_plot = [(2, 5), (6, 5), (4, 20), (6, 10), (11, 15)]

for i, n_node in enumerate(n_nodes):
    for j, n_ppl in enumerate(n_ppls):
        b = np.linspace(-n_node*B_NODE, n_node*B_NODE, n_ppl*n_node*2)
        # factor of 1/π is due to np.sinc definition
        ic = np.abs(IC0 * np.sinc(b * B2BETA * JJ_WIDTH / 2 / np.pi))
        x_out, j_out = extract_current_distribution(
                b, ic, B2BETA, JJ_WIDTH, 100)
        fidelity[i, j] = np.sqrt(np.mean((j_out - j_true(x_out))**2))

        if (n_node, n_ppl) in should_plot:
            ax.plot(x_out, j_out, label=f'{n_node}, {n_ppl}')
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel('points per lobe')
ax.set_ylabel('# of nodes')
ax.set_xticks(n_ppls)
ax.set_yticks(n_nodes)
im = ax.imshow(fidelity, origin='lower', aspect='auto', extent=[
    ppl_start - ppl_step / 2, ppl_stop - ppl_step / 2,
    node_start - node_step / 2, node_stop - node_step / 2])
cb = fig.colorbar(im)
cb.set_label(r'rms error')
fig.tight_layout()
