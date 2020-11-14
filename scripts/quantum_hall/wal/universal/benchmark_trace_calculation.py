# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# --------------------------------------------------------------------------------------
"""Script to benchmark the calculation of the trace over the trajectories.

"""
import os
import sys
import time
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from shabanipy.quantum_hall.wal.universal.trajectories import (
    get_detailed_trajectory_data,
)
from shabanipy.quantum_hall.wal.universal import (
    compute_trajectory_traces_no_zeeman,
    compute_trajectory_traces_zeeman,
)

# --- Parameters -----------------------------------------------------------------------

#: Number of trajectories to consider.
TRAJECTORY_NUMBERS = [1000, 10000, 40000, 60000, 80000, None]

#: Parameters to use in the calculation: alpha, beta1, beta3, B_Zeeman_x, B_Zeeman_y
PARAMETERS = (pi / 10, pi / 200, pi / 30, pi / 50, pi / 50)

#: Include Zeeman term in the calculation
INCLUDE_ZEEMAN = False

# --- Calculation ----------------------------------------------------------------------

func = (
    compute_trajectory_traces_zeeman
    if INCLUDE_ZEEMAN
    else compute_trajectory_traces_no_zeeman
)


for dtype in [np.float32, np.float64]:
    print(f"Loading data for {dtype}")
    index, l, c_phi, c_3phi, s_phi, s_3phi = get_detailed_trajectory_data(dtype)

    numbers = np.array([(n or index.shape[0]) for n in TRAJECTORY_NUMBERS])
    res = np.empty_like(numbers, dtype=float)

    for i, n in enumerate(numbers):
        print(f"    Computing for {n} traces")
        params = (
            index,
            l,
            c_phi,
            c_3phi,
            s_phi,
            s_3phi,
            PARAMETERS[0],
            PARAMETERS[1],
            PARAMETERS[2],
            n,
        ) + (PARAMETERS[3:] if INCLUDE_ZEEMAN else ())

        # Pre-compile to avoid timing the compilation time
        if i == 0:
            func(*params)

        tic = time.perf_counter()
        func(*params)
        res[i] = time.perf_counter() - tic

    plt.plot(numbers, res, label=f"{dtype}")

plt.xlabel("Number of traces used")
plt.ylabel("Time spent to compute the traces")
plt.title(f"Time for traces {'with' if INCLUDE_ZEEMAN else 'without'} Zeeman field")
plt.legend()
plt.show()
