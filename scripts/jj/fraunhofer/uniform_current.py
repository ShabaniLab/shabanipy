"""Test reconstruction of a zero-padded uniform current density."""
import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
    extract_theta,
)
from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast

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

# compare true and reconstructed phase distributions
theta_r = extract_theta(b, ic, b2beta, jj_width, method='romb')
theta_q = extract_theta(b, ic, b2beta, jj_width, method='quad')
theta_h = extract_theta(b, ic, b2beta, jj_width, method='hilbert')
theta_true = np.angle(g)

fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.set_xlabel('B [mT]')
ax2.set_ylabel(r'$\theta [\pi]$')

# bad hack: manually unwrap this particular theta_true (np.unwrap fails)
theta_true_unwrapped = (
    -np.pi * (
        np.cumsum(
            np.abs(
                np.diff(
                    (theta_true + 1e-6) % (2*np.pi) - 1e-6,  # 0-pi square wave
                    prepend=theta_true[0]
                )
            ) > np.pi / 2
        ) - 12
    )
)

ax2.plot(b / 1e-3, theta_r / np.pi, lw=1, marker='.', markersize=2,
        label='romb')
ax2.plot(b / 1e-3, theta_q / np.pi, lw=1, marker='.', markersize=2,
        label='quad')
ax2.plot(b / 1e-3, theta_h / np.pi, lw=1, marker='.', markersize=2,
        label='hilbert')
ax2.plot(b / 1e-3, theta_true_unwrapped / np.pi, lw=1, marker='.', markersize=2,
        label='true')
ax2.legend()

# compare current reconstruction using true vs. reconstructed phases
x_r, jx_r = extract_current_distribution(b, ic, b2beta, jj_width, 100,
        theta=theta_r)
x_q, jx_q = extract_current_distribution(b, ic, b2beta, jj_width, 100,
        theta=theta_q)
x_h, jx_h = extract_current_distribution(b, ic, b2beta, jj_width, 100,
        theta=theta_h)
x_true, jx_true = extract_current_distribution(b, ic, b2beta, jj_width, 100,
        theta=theta_true)

fig3, ax3 = plt.subplots(constrained_layout=True)
ax3.set_xlabel('x [um]')
ax3.set_ylabel('J [uA/um]')
ax3.plot(x / 1e-6, jx, linewidth=1.5, label='original', color='black')
ax3.plot(x_r / 1e-6, jx_r, linewidth=1.5, label='romb')
ax3.plot(x_q / 1e-6, jx_q, linewidth=1.5, label='quad')
ax3.plot(x_h / 1e-6, jx_h, linewidth=1.5, label='hilbert')
ax3.plot(x_true / 1e-6, jx_true, linewidth=1.5, label='true')
ax3.legend()

plt.show()
