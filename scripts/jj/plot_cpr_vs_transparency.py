"""Plot the current-phase relation of a Josephson junction for various transparencies."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import eV

from shabanipy.jj import transparent_cpr as cpr
from shabanipy.utils.plotting import jy_pink

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

plt.style.use(["fullscreen13"])
fig, ax = plt.subplots()
if args.temperature != 0 and args.gap != 0:
    ax.set_title(f"$T$={args.temperature}K, $\\Delta$={round(args.gap / 1e-3, 3)}meV")
ax.set_xlabel("phase")
ax.set_ylabel("supercurrent")
jy_pink.register()
for i, tau in enumerate(transparency):
    lines = ax.plot(
        phase,
        cpr(phase, 1, tau, temperature=args.temperature, gap=args.gap * eV),
        label=f"{round(tau, 4)}",
        color=plt.get_cmap("jy_pink")(i / len(transparency[:-1])),
    )
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(["0", "π", "2π"])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["$-I_c$", "0", "$+I_c$"])
ax.legend(title="transparency")

Path("output").mkdir(exist_ok=True)
fig.savefig(
    f"output/cpr-vs-transparency_{round(transparency[0], 4)}-{round(transparency[-1], 4)}.png"
)

plt.show()
