"""Plot the Andreev bound state energy for various transparencies."""

import argparse
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj import andreev_bound_state_energy as abse
from shabanipy.utils.plotting import jy_pink

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
args = parser.parse_args()

phase = np.linspace(0, 2 * np.pi, 200)
transparency = np.arange(0, 0.81, 0.2)
transparency = np.append(transparency, [0.99, 0.9999])

plt.style.use(
    {
        "figure.constrained_layout.use": True,
        "font.size": 12,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)
fig, ax = plt.subplots()
ax.set_xlabel("phase")
ax.set_ylabel("energy [$\Delta$]")
jy_pink.register()
for (i, tau), sign in product(enumerate(transparency), (1, -1)):
    ax.plot(
        phase,
        sign * abse(phase, transparency=tau, gap=1),
        label=f"{round(tau, 4)}" if sign == 1 else None,
        color=plt.get_cmap("jy_pink")(i / len(transparency)),
    )
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(["0", "π", "2π"])

ax.legend(title="transparency")
plt.show()
