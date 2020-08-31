"""Test Hilbert transform implementation against a known analytical example.

Note that our _extract_theta* subroutines implement the Hilbert transform of
the *logarithm* of the input.
"""
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    _extract_theta_hilbert,
    _extract_theta_quad,
    _extract_theta_romb,
)


def f(x):
    """A function with an analytical Hilbert transform."""
    return 1 / (x ** 2 + 1)


def hilbert_f(x):
    """The Hilbert transform of f."""
    return x / (x ** 2 + 1)


xs = np.linspace(-10, 10, 105)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title(r"Hilbert transform of $\frac{1}{x^2 + 1}$")
ax.plot(xs, hilbert_f(xs), label=r"$\frac{x}{x^2+1}$")

mpl.rcParams["lines.linewidth"] = 0
mpl.rcParams["lines.marker"] = "."

ax.plot(
    xs, _extract_theta_hilbert(np.exp(f(xs))), label="scipy.signal.hilbert",
)
ax.plot(
    xs, _extract_theta_romb(xs, np.exp(f(xs))), label="scipy.integrate.romb",
)
ax.plot(
    xs, _extract_theta_quad(xs, np.exp(f(xs))), label="scipy.integrate.quad",
)
ax.plot(
    xs,
    _extract_theta_quad(xs, np.exp(f(xs)), use_points=True),
    label="scipy.integrate.quad (points)",
)
ax.legend()
plt.show()
