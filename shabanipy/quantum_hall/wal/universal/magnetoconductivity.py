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

#: Corrective factor in the magneto conductance calculation depending on the
#: maximum number of scattering events considered.
F = np.sum(1 / (np.arange(3, 5001) - 2))


def wal_magneto_conductance(
    field: float,
    l_phi: float,
    traces: np.ndarray,
    surfaces: np.ndarray,
    lengths: np.ndarray,
    cosjs: np.ndarray,
) -> float:
    """[summary]

    Parameters
    ----------
    field : float
        Out of plane field (in rad/surface unit)
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
        Conductance at the given field in unit of e^2/h

    """
    xj = np.exp(-lengths / l_phi) * 0.5 * traces * (1 + cosjs)
    a = xj * np.cos(field * surfaces)

    return -2 * F * np.sum(a) / len(traces)
