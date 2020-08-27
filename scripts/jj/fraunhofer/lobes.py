"""Investigate how Fraunhofer lobes impact reconstructed current density.

In particular, how do
    1) the number of side lobes (i.e. B-field range), and
    2) the number of points per lobe (i.e. B-field resolution)
in the Fraunhofer pattern I_c(B) impact the fidelity of a reconstructed
uniform, Guassian, or generalized normal current density J(x).
"""
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
)
from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast

PHI0 = 2e-15
JJ_WIDTH = 4e-6
JJ_LENGTH = 100e-9
B2BETA = 2 * np.pi * JJ_LENGTH / PHI0
B_NODE = PHI0 / (JJ_WIDTH * JJ_LENGTH)
IC0 = 1e-6

def j_uniform(x):
    """Uniform current density with which to compare output."""
    return IC0 / JJ_WIDTH * np.piecewise(x,
            [np.abs(x) < JJ_WIDTH / 2, np.abs(x) >= JJ_WIDTH / 2], [1, 0])

def j_gaussian(x):
    """Gaussian current density."""
    return IC0 * scipy.stats.norm.pdf(x, loc=0, scale=JJ_WIDTH/4)

def j_gennorm(x):
    """Generalized normal distributed current density."""
    return IC0 * scipy.stats.gennorm.pdf(x, 8, loc=0, scale=JJ_WIDTH/2)

def j_multigate(x, distr):
    """Five-gate multigate distribution.

    Parameters
    ----------
    distr : list or np.ndarray
        1D list or array, of length 5, describing the relative distribution of
        current density in the regions below each minigate. Will be
        automatically normalized.
    """
    NUM_GATES = 5
    GATE_WIDTH = JJ_WIDTH / 5
    left_edge = -JJ_WIDTH / 2
    distr = np.asarray(distr) / np.sum(distr)
    return np.piecewise(x,
        [np.logical_and(x >= left_edge, x < left_edge + GATE_WIDTH),
        np.logical_and(x >= left_edge + GATE_WIDTH,
            x < left_edge + 2*GATE_WIDTH),
        np.logical_and(x >= left_edge + 2*GATE_WIDTH,
            x < left_edge + 3*GATE_WIDTH),
        np.logical_and(x >= left_edge + 3*GATE_WIDTH,
            x < left_edge + 4*GATE_WIDTH),
        np.logical_and(x >= left_edge + 4*GATE_WIDTH, x < JJ_WIDTH / 2)
        ], distr * IC0 / GATE_WIDTH)

# select a current density profile
j_true = lambda x: j_multigate(x, [3, 5, 2, 5, 3])

x = np.linspace(-JJ_WIDTH, JJ_WIDTH, 200)
jx = j_true(x)

node_start, node_stop, node_step = 2, 15, 1  # number of nodes
ppl_start, ppl_stop, ppl_step = 10, 65, 5    # points per lobe
n_nodes = np.arange(node_start, node_stop, node_step)
n_ppls = np.arange(ppl_start, ppl_stop, ppl_step)
fidelity = np.empty(shape=(len(n_nodes), len(n_ppls)))

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel('x (um)')
ax.set_ylabel('J (uA/um)')
ax.plot(x / 1e-6, jx, color='k', label='original')

# choose a few reconstructions to plot (n_node, n_ppl)
should_plot = [(4, 40), (7, 60), (9, 30), (11, 40), (14, 40)]

for i, n_node in enumerate(n_nodes):
    for j, n_ppl in enumerate(n_ppls):
        b = np.linspace(-n_node*B_NODE, n_node*B_NODE, n_ppl*n_node*2)
        ic = produce_fraunhofer_fast(b, B2BETA, jx, x)
        x_out, j_out = extract_current_distribution(
                b, ic, B2BETA, JJ_WIDTH, 100)
        fidelity[i, j] = np.sqrt(np.mean((j_out - j_true(x_out))**2))

        if (n_node, n_ppl) in should_plot:
            ax.plot(x_out / 1e-6, j_out, label=f'{n_node}, {n_ppl}')
ax.legend()

fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.set_xlabel('B [mT]')
ax2.set_ylabel(r'$I_c$ [uA]')
ax2.plot(b / 1e-3, ic / 1e-6)

fig3, ax3 = plt.subplots(constrained_layout=True)
ax3.set_xlabel('points per lobe')
ax3.set_ylabel('# of nodes')
ax3.set_xticks(n_ppls)
ax3.set_yticks(n_nodes)
im = ax3.imshow(fidelity, origin='lower', aspect='auto', extent=[
    ppl_start - ppl_step / 2, ppl_stop - ppl_step / 2,
    node_start - node_step / 2, node_stop - node_step / 2])
cb = fig3.colorbar(im)
cb.set_label(r'rms error')

plt.show()
