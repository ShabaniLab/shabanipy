# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""WAL computation based on Boltzmannian trajectories.

Inspired from:
A. Sawada, T. Koga, Universal modeling of weak antilocalization corrections in
quasi-two-dimensional electron systems using predetermined return orbitals.
Phys. Rev. E. 95, 023309 (2017).

"""

from .trace_computation import (
    compute_trajectory_traces_no_zeeman,
    compute_trajectory_traces_zeeman,
)
from .magnetoconductivity import wal_magneto_conductance
