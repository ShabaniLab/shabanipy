"""Plot the current-phase relation of a Josephson junction for various transparencies."""


import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants
from tqdm import tqdm

from shabanipy.jj import finite_transparency_jj_current as cpr

PHI0 = physical_constants["mag. flux quantum"][0]
res = 1000
t1 = 0
t2 = 0
ic2 = 200e-9
ic1 = 10 * ic2
inductance = 500e-12
flux = np.linspace(0, PHI0, 100)
phi1 = np.linspace(0, 2 * np.pi, res)

phi2 = np.zeros((len(flux), len(phi1)))
flux_ext = np.zeros((len(flux), len(phi1)))
i_squid = np.zeros((len(flux), len(phi1)))
print(f"Doing {len(flux)} iterations")
for i, f in tqdm(enumerate(flux)):
    for j, p1 in enumerate(phi1):
        p2 = p1 - 2 * np.pi * f / PHI0  # flux quantization
        i1 = cpr(p1, ic1, t1)
        i2 = cpr(p2, ic2, t2)

        phi2[i, j] = p2
        flux_ext[i, j] = f + inductance * (i2 - i1) / 2
        i_squid[i, j] = i1 + i2

phi1_true = np.zeros(len(flux))
phi2_true = np.zeros(len(flux))
flux_ext_true = np.zeros(len(flux))
for i, f in enumerate(flux):
    idx_true = np.argmax(i_squid[i])
    phi2_true[i] = phi2[i, idx_true]
    flux_ext_true[i] = flux_ext[i, idx_true]
    phi1_true[i] = phi2_true[i] + 2 * np.pi * f / PHI0

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
fig.savefig(
    f"phase-vs-fluxext_{int(inductance/1e-12)}pH-{int(round(ic1/1e-6))}uA-{int(round(ic2/1e-9))}nA-t1={round(t1, 1)}-t2={round(t2, 1)}.png",
    format="png",
)
fig.savefig(
    f"phase-vs-fluxext_{int(inductance/1e-12)}pH-{int(round(ic1/1e-6))}uA-{int(round(ic2/1e-9))}nA-t1={round(t1, 1)}-t2={round(t2, 1)}.svg",
    format="svg",
)

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
fig.savefig(
    f"flux-vs-fluxext_{int(inductance/1e-12)}pH-{int(round(ic1/1e-6))}uA-{int(round(ic2/1e-9))}nA-t1={round(t1, 1)}-t2={round(t2, 1)}.png",
    format="png",
)
fig.savefig(
    f"flux-vs-fluxext_{int(inductance/1e-12)}pH-{int(round(ic1/1e-6))}uA-{int(round(ic2/1e-9))}nA-t1={round(t1, 1)}-t2={round(t2, 1)}.svg",
    format="svg",
)

plt.show()
