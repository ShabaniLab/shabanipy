# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Compute some parameters in the trajectories

"""

from numba import njit
import numpy as np
from math import sqrt, atan2, cos, sin


@njit(fastmath=True)
def find_each_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the length of each segment.

    Parameters
    ----------
    x: np.ndarray
        (1, n_scat) array, the x position of each point
    y: np.ndarray
        (1, n_scat) array, the y position of each point

    Returns
    ----------
    length
        (1, n_scat) array, length of each segment

    """
    return np.sqrt((y[1:] - y[:-1]) ** 2 + (x[1:] - x[:-1]) ** 2)


@njit(fastmath=True)
def find_each_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the angle of each segment.

    Parameters
    ----------
    x: np.ndarray
        (1, n_scat) array, the x position of each point
    y: np.ndarray
        (1, n_scat) array, the y position of each point

    Returns
    ----------
    angle
        (1, n_scat) array, angle of each segment

    """
    return np.arctan2((y[1:] - y[:-1]), (x[1:] - x[:-1]))
