# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Plot the weak antilocalization conductance for different parameters.

This consider only the simplified diffusive model which neglects any linear
Dresselhaus term.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Field range over which to plot the WAL signal. The unit is arbitrary since
#: only the ratio of the following quantities to field is meaningful in this
#: model.
FIELD_RANGE = (-2, 2, 2001)

#: Values of the dephasing field for which to plot in unit of field.
DEPHASING = [0.1, 1, 10]

#: Values of the Rashba SOI to plot in unit of field.
RASHBA_SOI = [0.1, 1, 10]

#: Values of the cubic Dresselhaus contribution to plot in unit of field.
CUBIC_DRES_SOI = [0.0, 0.1, 1, 10]

#: Reference field used to compute Δσ
REFERENCE_FIELD = 0.01

#: Maximum number of Landay levels to consider when performing the truncation
#: of the series expansion involved in the calculation
TRUNCATION = 5000

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.quantum_hall.wal.wal_no_dresselhaus import (
    compute_wal_conductance_difference,
)

field = np.logspace(*FIELD_RANGE)

for d in DEPHASING:
    print(f"Performing calculations for Hφ = {d}")
    for lr in RASHBA_SOI:
        print(f"    Performing calculation for HS_O = {lr}")
        plt.figure()
        plt.title(f"Hφ = {d}," "H$_{SO}$ = " f"{lr}")
        for cd in CUBIC_DRES_SOI:
            print(f"        Performing calculation for cubic SOI = {cd}")
            tic = time.perf_counter()
            value = compute_wal_conductance_difference(
                field, d, lr, cd, REFERENCE_FIELD, TRUNCATION
            )
            print(f"        Calculation took {time.perf_counter() - tic}s")
            plt.plot(field, value, label=f"Cubic {cd}")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("Magnetic field (a.u.)")
        plt.ylabel("Δσ (e²/(2πℏ)")

plt.show()
