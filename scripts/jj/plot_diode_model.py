"""Plot the diode model from https://arxiv.org/abs/2303.01902 Eq. (1).

Illustrate the effect of each parameter on the shape of the model function."""
import numpy as np
from matplotlib import pyplot as plt


def icp_model(x, imax, b, c, bstar):
    """Positive critical current, Ic+."""
    return imax * (1 - b * (1 + c * np.sign(x - bstar)) * (x - bstar) ** 2)


def icm_model(x, imax, b, c, bstar):
    """Negative critical current, Ic-."""
    return imax * (1 - b * (1 - c * np.sign(x + bstar)) * (x + bstar) ** 2)


cmap = plt.get_cmap("viridis")
x = np.linspace(-0.1, 0.1, 1000)
imax = 1

n = 5
c = 0
bstar = 0
for i, b in enumerate(np.linspace(0, 20, n)):
    plt.plot(x, icp_model(x, imax, b, c, bstar), color=cmap(i / n), label=f"${b=}$")
    plt.plot(x, icm_model(x, imax, b, c, bstar), color=cmap(i / n), linestyle="--")
    plt.title(f"$c = {c}$, $B_* = {bstar}$")
plt.legend()
plt.show()

n = 6
b = 1
bstar = 0
for i, c in enumerate(np.linspace(0, 1, n)):
    plt.plot(
        x,
        icp_model(x, imax, b, c, bstar),
        color=cmap(i / n),
        label=f"$c={round(c, 1)}$",
    )
    plt.plot(x, icm_model(x, imax, b, c, bstar), color=cmap(i / n), linestyle="--")
    plt.title(f"$b = {b}$, $B_* = {bstar}$")
plt.legend()
plt.show()

n = 6
b = 1
c = 0
for i, bstar in enumerate(np.linspace(0, 0.05, n)):
    plt.plot(
        x, icp_model(x, imax, b, c, bstar), color=cmap(i / n), label=f"$B_*={bstar}$"
    )
    plt.plot(x, icm_model(x, imax, b, c, bstar), color=cmap(i / n), linestyle="--")
    plt.title(f"$b = {b}$, $c = {c}$")
plt.legend()
plt.show()
