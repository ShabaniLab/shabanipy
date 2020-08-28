"""Test Hilbert transform implementation against a known analytical example."""
import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.deterministic_reconstruction import extract_theta


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
# n.b. extract_theta computes the hilbert transform of the logarithm
ax.plot(
    xs,
    extract_theta(xs, np.exp(f(xs)), 1, 0, method="hilbert"),
    lw=0,
    marker=".",
    label="shabanipy (hilbert)",
)
ax.plot(
    xs,
    extract_theta(xs, np.exp(f(xs)), 1, 0, method="quad"),
    lw=0,
    marker=".",
    label="shabanipy (quad)",
)
ax.legend()
plt.show()
