# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines used to compute Boltzmanians trajectories using ran1.

See the following references for details:
A. Sawada, T. Koga, Universal modeling of weak antilocalization corrections in
quasi-two-dimensional electron systems using predetermined return orbitals.
Phys. Rev. E. 95, 023309 (2017).

"""
from collections import namedtuple
from math import sqrt, log, cos, sin, pi

import numpy as np
from numba import njit

from .random_number_generator import seed_ran1, ran1


Point = namedtuple('Point', ['x', 'y'])


@njit
def check_return_condition(point: tuple,
                           next_point: tuple,
                           distance: float) -> bool:
    """Check if the particle returned to the origin.

    """
    l2 = (point.x - next_point.x)**2 + (point.y - next_point.y)**2
    if (l2 == 0.0):
        return sqrt(point.x**2 + point.y**2) < distance
    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of the origin p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = max(0, min(1, - (point.x*(next_point.x - point.x) +
                         point.y*(next_point.y - point.y) / l2)))
    # projection = v + t * (w - v) falls on the segment
    return (sqrt((point.x + t*(next_point.x - point.x))**2 +
                 (point.y + t*(next_point.y - point.y))**2) < distance)


@njit
def identify_trajectory(seed: int, n_scat_max: int, distance: float) -> int:
    """Identify the number of scattering event leading to a return trajectory.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    n_scat_max : int
        Maximum number of scattering event to consider.
    distance : float
        Distance below which we consider the particle returned to the origin.

    """
    state = seed_ran1(seed)

    # First scattering
    r, state = ran1(state)
    x = old_x = - log(r)
    y = old_y = 0
    i = 1

    while i <= n_scat_max:
        r, state = ran1(state)
        length = -log(r)
        r, state = ran1(state)
        theta = 2*pi*r
        x += length*cos(theta)
        y += length*sin(theta)
        if check_return_condition(Point(old_x, old_y), Point(x, y), distance):
            break
        i += 1
        old_x = x
        old_y = y

    return i


@njit
def generate_trajectory(seed: int, n_scat: int) -> np.ndarray:
    """Generate a trajectory containing a known number of points.

    """
    trajectory = np.zeros((n_scat+1, 2))
    state = seed_ran1(seed)

    # First scattering
    r, state = ran1(state)
    x = - log(r)
    y = 0
    trajectory[1, 0] = x
    trajectory[1, 1] = y

    for i in range(2, n_scat+1):
        r, state = ran1(state)
        length = -log(r)
        r, state = ran1(state)
        theta = 2*pi*r
        x += length*cos(theta)
        y += length*sin(theta)
        trajectory[i, 0] = x
        trajectory[i, 1] = y

    return trajectory
