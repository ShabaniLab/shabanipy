"""Plot constraints on the parameters of a dc SQUID given in Tinkham ed. 2 (6.50)."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann as k
from scipy.constants import Planck as h
from scipy.constants import elementary_charge as e
from scipy.constants import hbar

PHI0 = h / (2 * e)

plt.style.use(["fullscreen13", "usetex"])

# hysteretic constraints vs. Ic
fig, ax = plt.subplots()
ax.set_title(r"hysteretic constraints (Tinkham p. 227)")
ax.set_xlabel(r"$I_c$ (\si{\micro A})")
ax.set_ylabel(r"$L$ (nH) and $R^2C$ (\si{\ohm^2.nF})")

current = np.linspace(1e-9, 6e-6, 1000)
# Tinkham (6.50a): inductance limited by onset of magnetic hysteresis
inductance = PHI0 / (2 * current)
ax.fill_between(current / 1e-6, inductance / 1e-9, label=r"$L < \Phi_0 / 2I_c$")
# Tinkham (6.50b): resistance, capacitance limited by underdamping hysteresis
r2c = PHI0 / (2 * np.pi * current)
ax.fill_between(
    current / 1e-6, r2c / 1e-9, label=r"$R^2C < \Phi_0 / 2\pi I_c$", alpha=0.7
)

ax.set_xlim((0, current[-1] / 1e-6))
ax.set_ylim((0, 1))
ax.legend()

# thermal constraints vs. T
fig, ax = plt.subplots()
ax.set_title(r"thermal constraints (Tinkham p. 227)")
ax.set_xlabel(r"$T$ (K)")
ax.set_ylabel(r"$L$ (nH)")
ax2 = ax.twinx()
ax2.set_ylabel("$I_c$ (nA)")

temperature = np.linspace(10e-3, 2, 1000)
# Tinkham (6.50d): inductance is limited by thermal flux fluctuations
inductance = PHI0 ** 2 / (4 * k * temperature)
fill1 = ax.fill_between(temperature, inductance / 1e-9, label=r"$L < \Phi_0^2 / 4kT$")
# Tinkham (6.50c): critical current limited by phase fluctuations
current = 10 * e * k * temperature / hbar
fill2 = ax2.fill_between(
    temperature,
    current / 1e-9,
    max(current) / 1e-9,
    label=r"$I_c \propto E_J > 5kT$",
    color="tab:orange",
    alpha=0.7,
)

ax.set_xlim(0, temperature[-1])
ax.set_ylim((0, 100))
ax2.set_ylim((0, max(current) / 1e-9))
ax2.legend(handles=[fill1, fill2])

plt.show()
