"""Plot the effect of inductance on SQUID phases."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants

from shabanipy.jj import transparent_cpr as cpr
from shabanipy.squid import critical_behavior

PHI0 = physical_constants["mag. flux quantum"][0]
res = 1000
t1 = 0
t2 = 0
ic2 = 200e-9
ic1 = 10 * ic2
inductance = 500e-12
flux = np.linspace(0, PHI0, 100)

_, phi_ext_true, _, phi1_true, _, phi2_true = critical_behavior(
    flux * 2 * np.pi / PHI0,
    cpr,
    (0, ic1, t1),
    cpr,
    (0, ic2, t2),
    inductance / PHI0,
    branch="+",
    nbrute=res,
    return_jjs=True,
)
flux_ext_true = phi_ext_true / (2 * np.pi) * PHI0

plt.style.use(
    {
        "figure.constrained_layout.use": True,
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)

# plot external flux from 0 to Φ_0 on x axis
flux_ext_true = flux_ext_true % PHI0
sortidx = np.argsort(flux_ext_true)
flux_ext_true = flux_ext_true[sortidx]
phi1_true = phi1_true[sortidx]
phi2_true = phi2_true[sortidx]
flux = flux[sortidx]

fig, ax = plt.subplots()
ax.set_xlabel("$\Phi_\mathrm{ext}$")
ax.set_ylabel("phase")
ax.set_xticks(np.array([0, 1 / 2, 1]) * PHI0)
ax.set_xticklabels(["0", "$\Phi_0/2$", "$\Phi_0$"])
ax.set_yticks(np.array([-2, -3 / 2, -1, -1 / 2, 0, 1 / 2]) * np.pi)
ax.set_yticklabels(["-2π", "-3π/2", "-π", "-π/2", "0", "π/2"])
ax.plot(
    flux_ext_true, np.unwrap(phi1_true), label=r"$\varphi_1$", color="tab:blue",
)
ax.plot(
    flux_ext_true[[0, -1]],
    np.unwrap(phi1_true)[[0, -1]],
    linestyle="--",
    color="tab:blue",
)
ax.plot(
    flux_ext_true, np.unwrap(phi2_true), label=r"$\varphi_2$", color="tab:red",
)
ax.plot(
    flux_ext_true[[0, -1]],
    np.unwrap(phi2_true)[[0, -1]],
    linestyle="--",
    color="tab:red",
)
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel("$\Phi_\mathrm{ext}$")
ax.set_ylabel("$\Phi$")
ax.set_xticks(np.array([0, 1 / 2, 1]) * PHI0)
ax.set_xticklabels(["0", "$\Phi_0/2$", "$\Phi_0$"])
ax.set_yticks(np.array([0, 1 / 4, 1 / 2, 3 / 4, 1, 5 / 4]) * PHI0)
ax.set_yticklabels(
    ["0", "$\Phi_0/4$", "$\Phi_0/2$", "$3\Phi_0/4$", "$\Phi_0$", "$5\Phi_0/4$"]
)
ax.plot(
    flux_ext_true, np.unwrap(flux, period=PHI0), color="black",
)
ax.plot(
    flux_ext_true[[0, -1]],
    np.unwrap(flux, period=PHI0)[[0, -1]],
    linestyle="--",
    color="black",
)

plt.show()
