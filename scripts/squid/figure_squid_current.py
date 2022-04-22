# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Generate a nice visualization of the impact of the transparency on the CPR.

"""

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.squid.squid_model import compute_squid_current
from shabanipy.jj import finite_transparency_jj_current as cpr

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 13
plt.rcParams['pdf.fonttype'] = 42

phase_diff = np.linspace(-2*np.pi, 2*np.pi, 1001)

fig, axes = plt.subplots(4, 3, figsize=(12,6),
                         constrained_layout=True, sharex=True, sharey='row')
for i, a in enumerate((0.1, 0.5, 0.95, 1.5)):
    for j, t in enumerate((0.1, 0.5, 0.9)):
        squid = compute_squid_current(phase_diff,
                                      cpr, (0, 1, 0.9),
                                      cpr, (0, a, t))
        amplitude = (np.amax(squid) - np.min(squid))
        baseline = (np.amax(squid) + np.min(squid))/2
        axes[i, j].plot(phase_diff/np.pi,
                        squid,
                        label=f'a={a}')
        line = axes[i, j].axvline(phase_diff[100 + np.argmax(squid[100:])]/np.pi)
        line.set_linestyle('--')
        axes[i, j].tick_params(direction='in', width=1.5)
        if i == 0:
            axes[i, j].set_title(f'Transparency {t}')
        if i == 2:
            ticks = [-2, -1, 0, 1, 2]
            labels = [f'{v} π' if v not in (0.0, 1.0) else
                        ('0.0' if v == 0 else 'π') for v in ticks]
            axes[i, j].set_xticks(ticks)
            axes[i, j].set_xticklabels(labels)
            axes[i, j].set_xlabel('Phase difference')
        if j == 0:
            axes[i, j].set_ylabel('I$_{c}$ (a.u.)')
        axes[i, j].legend(loc='lower right')
plt.show()
