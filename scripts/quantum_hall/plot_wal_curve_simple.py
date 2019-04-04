# -*- coding: utf-8 -*-
"""Plot the weak antilocalization conductance for different parameters.

This consider only the simplified model.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Field range over which to plot the WAL signal
FIELD_RANGE = (1, 150, 1501)

#: Values of the dephasing field for which to plot in unit of field.
DEPHASING = [0.1, 1, 10]

#: Values of the Rashba SOI to plot in unit of field.
RASHBA_SOI = [0.1, 1, 10]

#: Values of the cubic Dresselhaus contribution to plot in unit of field.
CUBIC_DRES_SOI = [0.1, 1, 10]

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.quantum_hall.wal.wal_no_dresselhaus\
    import compute_wal_conductance_difference

field = np.linspace(*FIELD_RANGE)

for d in DEPHASING:
    for lr in RASHBA_SOI:
        plt.figure()
        plt.title(f'Hφ = {d},' 'H$_{SO}$ = ' f'{lr}')
        for cd in CUBIC_DRES_SOI:

            tic = time.perf_counter()
            value = compute_wal_conductance_difference(field, d, lr, cd, 10,
                                                       500)
            print(time.perf_counter() - tic)
            plt.plot(field, value, label=f'Cubic {cd}')
        plt.legend()

plt.show()
