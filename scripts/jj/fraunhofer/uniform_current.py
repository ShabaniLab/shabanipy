"""Test reconstruction of a zero-padded uniform current density."""
import matplotlib.pyplot as plt
import numpy as np

from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast
from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
        extract_theta, extract_current_distribution)

# constants
PHI0 = 2e-15  # magnetic flux quantum
jj_width = 1e-6
jj_length = 100e-9  # includes London penetration depths
b2beta = 2 * np.pi * jj_length / PHI0  # B-field to beta factor

# zero-padded uniform current distribution
x = np.linspace(-2*jj_width, 2*jj_width, 513)
jx = np.zeros_like(x)
jx[np.where(np.abs(x) < jj_width / 2)] = 1

# generate fraunhofer
b = np.linspace(-0.25, 0.25, 513)
g = produce_fraunhofer_fast(b, b2beta, jx, x, ret_fourier=True)
ic = np.abs(g)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel('B [mT]')
ax.set_ylabel(r'$I_c$ [uA]')
ax.plot(b / 1e-3, ic / 1e-6)
fig.show()

# compare true and reconstructed phase distributions
theta = extract_theta(b, ic, b2beta, jj_width)
theta_true = np.angle(g)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel('B [mT]')
ax.set_ylabel(r'$\theta [\pi]$')
ax.set_yticks(np.arange(-0.5, 2, 0.5))
# wrap reconstructed phase at [-π/2, 3π/2] (in units of π) to allow better
# comparison with true phase
ax.plot(b / 1e-3, ((theta / np.pi + 1/2) % 2 - 1/2),
        linewidth=0, marker='.', markersize=2, label='reconstructed')
# plot true phase modulo 2π (in units of π)
# small shift θ + ε forces points at 2π - ε into the 0 bin
ax.plot(b / 1e-3, (theta_true / np.pi + 1e-9) % 2,
        linewidth=0, marker='.', markersize=2, label='true')
ax.legend()
fig.show()

# compare current reconstruction using true vs. reconstructed phases
x2, jx2 = extract_current_distribution(b, ic, b2beta, jj_width, 100)
x2_true, jx2_true = extract_current_distribution(b, ic, b2beta, jj_width, 100,
        theta=theta_true)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel('x [um]')
ax.set_ylabel('J [uA/um]')
ax.plot(x / 1e-6, jx, linewidth=2, label='original', color='black')
ax.plot(x2 / 1e-6, jx2, linewidth=2, label='reconstructed')
ax.plot(x2_true / 1e-6, jx2_true, linewidth=2, label='phase-corrected')
ax.legend()
fig.show()
