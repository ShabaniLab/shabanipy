"""Test Hilbert transform implementation against a known analytical example.

Note that our _extract_theta* subroutines implement the Hilbert transform of
the *logarithm* of the input.
"""
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

from shabanipy.jj.fraunhofer.dynesfulton import (
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

# test currently-implemented routines
ax.plot(
    xs, _extract_theta_hilbert(np.exp(f(xs))), label="scipy.signal.hilbert",
)
ax.plot(
    xs, _extract_theta_romb(xs, np.exp(f(xs))), label="scipy.integrate.romb",
)
ax.plot(
    xs, _extract_theta_quad(xs, np.exp(f(xs))), label="scipy.integrate.quad",
)

# try implementation based on weights
def _extract_theta_quad_weight(fields: np.ndarray, ics: np.ndarray) -> np.ndarray:
    """Compute Eq. (5) of Dynes & Fulton (1971) using quad with cauchy weights."""
    ics_interp = interp1d(fields, ics, "cubic")

    # split the integration interval in two
    lim1, lim2, lim3 = np.min(fields), 0, np.max(fields)

    theta = np.empty_like(fields)
    for i, (field, ic) in enumerate(zip(fields, ics)):
        # wvar cannot equal integration limits
        if field in [lim1, lim2, lim3]:
            theta[i] = np.nan
            continue

        i1 = quad(
            lambda b, ic: (np.log(ics_interp(b)) - np.log(ic)) / (abs(field) - b),
            lim1,
            lim2,
            args=(ic,),
            weight="cauchy",
            wvar=-abs(field),
        )[0]
        i2 = quad(
            lambda b, ic: -(np.log(ics_interp(b)) - np.log(ic)) / (abs(field) + b),
            lim2,
            lim3,
            args=(ic,),
            weight="cauchy",
            wvar=abs(field),
        )[0]
        theta[i] = field / np.pi * (i1 + i2)
    return theta


ax.plot(
    xs, _extract_theta_quad_weight(xs, np.exp(f(xs))), label="quad (cauchy weight)",
)
ax.legend()
plt.show()
