"""Plot the Andreev bound state energy for various transparencies."""

import argparse
from itertools import product
from pathlib import Path

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

plt.style.use(["fullscreen13"])
fig, ax = plt.subplots()
ax.set_xlabel("phase")
ax.set_ylabel("energy")
jy_pink.register()
for (i, tau), sign in product(enumerate(transparency), (1, -1)):
    ax.plot(
        phase,
        sign * abse(phase, transparency=tau, gap=1),
        label=f"{round(tau, 4)}" if sign == 1 else None,
        color=plt.get_cmap("jy_pink")(i / len(transparency[:-1])),
    )
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(["0", "π", "2π"])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["$-\Delta$", "0", "$+\Delta$"])
ax.legend(title="transparency")

Path("output").mkdir(exist_ok=True)
fig.savefig(
    f"output/abs-vs-transparency_{round(transparency[0], 4)}-{round(transparency[-1], 4)}.png"
)

plt.show()
