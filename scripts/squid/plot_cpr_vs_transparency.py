"""Plot the current-phase relation of a Josephson junction for various transparencies."""

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.squid.cpr import finite_transparency_jj_current

phase = np.linspace(0, 2 * np.pi, 200)
transparency = np.linspace(0, 0.9, 10)
transparency = np.append(transparency, [0.9999])

fig, ax = plt.subplots()
ax.set_xlabel("phase [$2\pi$]")
ax.set_ylabel("$I_c$ [arb.u.]")
for tau in transparency:
    lines = ax.plot(
        phase / (2 * np.pi),
        finite_transparency_jj_current(phase, 1, tau),
        label=f"$\\tau$ = {round(tau, 2)}",
        color=plt.get_cmap("viridis")(tau),
    )
lines[-1].set_label(f"$\\tau$ = {transparency[-1]}")

ax.legend()
plt.show()
