"""Plot the current-phase relation of a Josephson junction for various transparencies."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import eV

from shabanipy.squid.cpr import finite_transparency_jj_current as cpr

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--temperature", metavar="T", type=float, default=0, help="temperature in K"
)
parser.add_argument(
    "--gap", metavar="Δ", type=float, default=200e-6, help="superconducting gap in eV"
)
args = parser.parse_args()

phase = np.linspace(0, 2 * np.pi, 200)
transparency = np.arange(0, 0.81, 0.2)
transparency = np.append(transparency, [0.99, 0.9999])

fig, ax = plt.subplots()
ax.set_title(f"$T$={args.temperature}K, $\\Delta$={round(args.gap / 1e-3, 3)}meV")
ax.set_xlabel("phase [$2\pi$]")
ax.set_ylabel("supercurrent [$I_c$]")
for tau in transparency:
    lines = ax.plot(
        phase / (2 * np.pi),
        cpr(phase, 1, tau, temperature=args.temperature, gap=args.gap * eV),
        label=f"{round(tau, 4)}",
        color=plt.get_cmap("viridis")(tau),
    )

ax.legend(title="transparency")
plt.show()
