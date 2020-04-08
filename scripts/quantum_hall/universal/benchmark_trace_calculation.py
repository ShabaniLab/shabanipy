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

from shabanipy.quantum_hall.wal.universal.trajectories import get_trajectory_data
from shabanipy.quantum_hall.wal.universal import (
    compute_trajectory_traces_no_zeeman,
    compute_trajectory_traces_zeeman,
)

os.environ["MKL_NUM_THREADS"] = "1"


# --- Parameters -----------------------------------------------------------------------

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

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
index, l, c_phi, c_3phi, s_phi, s_3phi = get_trajectory_data("trace")
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
    TRAJECTORY_NUMBER,
) + (PARAMETERS[3:] if INCLUDE_ZEEMAN else ())

# Pre-compile to avoid timing the compilation time
func(*params)

tic = time.time()
func(*params)
print("Computation", time.time() - tic)
