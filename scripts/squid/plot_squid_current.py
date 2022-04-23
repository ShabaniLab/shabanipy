# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Plot the squid current for different set of parameters.

"""

#: Parameters of the first junction: amplitude, transparency
FIRST_JUNCTION = (1, .5)

#: Transparencies of the second junction
SECOND_JUNCTION_TRANSPARENCIES = [0, 0.2, 0.4, 0.6, 0.8, 0.99999]

#: Amplitudes of the second junction
SECOND_JUNCTION_AMPLITUDES = [1]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

from shabanipy.squid import critical_behavior
from shabanipy.jj import transparent_cpr as cpr

phase_diff = np.linspace(-2*np.pi, 2*np.pi, 1001)
offset = 1.2

for t in SECOND_JUNCTION_TRANSPARENCIES:
    # plt.figure()
    for i, a in enumerate(SECOND_JUNCTION_AMPLITUDES):
        squid, *_ = critical_behavior(phase_diff,
                                      cpr, (0, *FIRST_JUNCTION),
                                      cpr, (offset, a, t))
        neg_squid, *_ = critical_behavior(phase_diff,
                                      cpr, (0, *FIRST_JUNCTION),
                                      cpr, (offset, a, t),
                                      False)
        amplitude = (np.amax(squid) - np.min(squid))
        baseline = (np.amax(squid) + np.min(squid))/2
        plt.plot(phase_diff,
                 2*(squid - baseline)/amplitude + 1,
                 label=f't={t}, a={a}')#, color=f'C{i}')
        amplitude = (np.amax(neg_squid) - np.min(neg_squid))
        baseline = (np.amax(neg_squid) + np.min(neg_squid))/2
        plt.plot(phase_diff,
                 2*(neg_squid - baseline)/amplitude - 1,
                 color=f'C{i}')
    plt.legend()
    # plt.figure()
    # plt.plot(phase_diff, cpr(phase_diff, *FIRST_JUNCTION))
    # plt.plot(phase_diff, cpr(offset + phase_diff, 1, t))
    # plt.legend()
plt.show()
