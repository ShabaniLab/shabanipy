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
ic = produce_fraunhofer_fast(b, b2beta, jx, x)

# reconstruct current distribution
theta = extract_theta(b, ic, b2beta, jj_width)
x2, jx2 = extract_current_distribution(b, ic, b2beta, jj_width, 100)

# plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[7,6.25])
ax1.set_ylabel('J(x)')
ax1.plot(x, jx, linewidth=0, marker='.', label='original')
ax1.plot(x2, jx2, linewidth=0, marker='.', label='reconstructed')
ax2.set_ylabel(r'$I_c(B)$')
ax2.plot(b, ic)
ax3.set_ylabel(r'$\theta(B)$')
# remove (1/2)*beta*a shift for easier comparison with Dynes & Fulton (1971)
ax3.plot(b, theta + b*b2beta*jj_width / 2, linewidth=0, marker='.')
fig.tight_layout()
