# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# --------------------------------------------------------------------------------------
"""Routines to compute the correction to the magneto-conductivity.

"""
from math import cos, exp, pi

import numpy as np
from numba import njit

#: Corrective factor in the magneto conductance calculation depending on the
#: maximum number of scattering events considered.
F = np.sum(1 / (np.arange(3, 5001) - 2))


@njit(fastmath=True)
def wal_magneto_conductance(
    fields: np.ndarray,
    l_phi: float,
    traces: np.ndarray,
    surfaces: np.ndarray,
    lengths: np.ndarray,
    cosjs: np.ndarray,
) -> float:
    """Compute the magneto conductance using precomputed trajectories traces.

    Parameters
    ----------
    fields : np.ndarray
        1D array of the out of plane field (in rad/surface unit)
    l_phi : float
        Spin phase coherence length.
    traces : np.ndarray
        Trace of the propagation matrix for each trajectory.
    surfaces : np.ndarray
        Surface enclosed by each trajectory.
    lengths: np.ndarray
        Length of each trajectory.
    cosjs : np.ndarray
        Cosinus of the angle of the segment returning to the origin.

    Returns
    -------
    float
        Conductance at the given field in unit of e^2/(2πh)
        Same unit as used in W. Knap et al., PRB. 53, 3912–3924 (1996).

    """
    sigma = np.empty_like(fields)
    for i, f in enumerate(fields):
        xj = np.exp(-lengths / l_phi) * 0.5 * traces * (1 + cosjs)
        a = xj * np.cos(f * surfaces)

        sigma[i] = -2 * F * np.sum(a) / len(traces)

    return sigma
