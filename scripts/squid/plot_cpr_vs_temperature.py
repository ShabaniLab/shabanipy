"""Plot the current-phase relation of a Josephson junction for various temperatures."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann as k_B
from scipy.constants import eV

from shabanipy.squid.cpr import finite_transparency_jj_current as cpr

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--transparency",
    metavar="τ",
    type=float,
    default=0.99,
    help="transparency of the junction, 0 <= τ < 1",
)
args = parser.parse_args()

phase = np.linspace(0, 2 * np.pi, 200)
gap = 200e-6 * eV
temperatures = np.linspace(0, gap / k_B, 6)

fig, ax = plt.subplots()
ax.set_title(f"transparency = {args.transparency}")
ax.set_xlabel("phase [$2\pi$]")
ax.set_ylabel("supercurrent [$I_c$]")
for temp in temperatures:
    lines = ax.plot(
        phase / (2 * np.pi),
        cpr(phase, 1, args.transparency, temperature=temp, gap=gap),
        label=f"{round(temp * k_B / gap, 2)}Δ",
        color=plt.get_cmap("viridis")(temp / np.max(temperatures)),
    )

ax.legend(title="temperature")
plt.show()
