# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to create and access the data required for different calculations.

See the following references for details:
A. Sawada, T. Koga, Universal modeling of weak antilocalization corrections in
quasi-two-dimensional electron systems using predetermined return orbitals.
Phys. Rev. E. 95, 023309 (2017).

"""
import os
import random
from collections import namedtuple
from math import cos, log, pi, sin, sqrt
from typing import Tuple, Optional

import h5py
import pandas as pd
import numpy as np
from numba import njit
from typing_extensions import Literal, overload

from .random_number_generator import ran1, seed_ran1
from .utils import find_each_angle, find_each_length

Point = namedtuple("Point", ["x", "y"])


@njit(fastmath=True)
def check_return_condition(point: Point, next_point: tuple, distance: float) -> bool:
    """ check if the particle return to the origin

    Parameters
    ----------
    point: tuple
        initial position
    next_point: tuple
        terminal position
    distance: float
        Distance below which we consider the particle returned to the origin.

    """

    dx = point.x - next_point.x
    dy = point.y - next_point.y
    # calculate the distance that the point move
    if ((point.x * dx + point.y * dy) * (next_point.x * dx + next_point.y * dy)) < 0.0:
        l_c = (next_point.y / dy - next_point.x / dx) ** 2
        l_ab = (1 / dx) ** 2 + (1 / dy) ** 2
        # use the formula of the distance of point(0, 0) to the line Ax+By+C=0
        return sqrt(l_c / l_ab) < distance
    else:
        return (
            min(
                sqrt(point.x ** 2 + point.y ** 2),
                sqrt(next_point.x ** 2 + next_point.y ** 2),
            )
            < distance
        )


@njit(fastmath=True)
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
    x = old_x = -log(r)
    y = old_y = 0.0
    i = 1

    while i <= n_scat_max:
        r, state = ran1(state)
        length = -log(r)
        r, state = ran1(state)
        theta = 2 * pi * r
        x += length * cos(theta)
        y += length * sin(theta)
        i += 1
        if i > 2 and check_return_condition(Point(old_x, old_y), Point(x, y), distance):
            break

        old_x = x
        old_y = y

    return i


@njit(fastmath=True)
def generate_trajectory(seed: int, n_scat: int) -> np.ndarray:
    """Generate a trajectory containing a known number of points.

    Parameters
    ----------
    seed : int
        Seed for the random generator.
    n_scat : int

    Returns
    -------
    trajectories :
        (2, n_scat) array. First line is x at each scattering event,
        second line is y.

    """
    trajectory = np.zeros((n_scat + 1, 2))
    state = seed_ran1(seed)

    # First scattering
    r, state = ran1(state)
    x = -log(r)
    y = 0
    trajectory[1, 0] = x
    trajectory[1, 1] = y

    for i in range(2, n_scat + 1):
        r, state = ran1(state)
        length = -log(r)
        r, state = ran1(state)
        theta = 2 * pi * r
        x += length * cos(theta)
        y += length * sin(theta)
        trajectory[i, 0] = x
        trajectory[i, 1] = y

    return trajectory.T


def create_all_data() -> None:
    """Function to create a data file to store all the trajectory data.

    The data file is organized as follow:
    - "valid": contains the complete details about each trajectory under "n={k}"
      where k is the number of the trajectory which match the expectated values.
      Each dataset store the x, y per point and the following global properties
      in the attrs: length, n_scat, seed, surface, cosj, calculated_n_scat,
      calculated_length, calculated_cosj
      If the dataset is empty it means that computed number of scattering events
      required to go back to the origin did not match the value reported in Sawada.
    - "invalid": same as above for trajectories not matching the expected values
      from Sawada.
    - "trace_calculation": contains the following columns useful in the computation
      of the trajectory contribution to the transport: l, c_phi, c_3phi, s_phi, s_3phi
      Only valid trajectories are reported.
    - "magneto_conductance": contains the following columns representing the
      global properties of each trajectories used in magneto-conductance
      calculation: Length, n_scat, seed, Surface, cosj
      Only valid trajectories are reported.

    """
    dir_name = os.path.dirname(__file__)
    temp_file = os.path.join(dir_name, "_trajectories_data.hdf5")
    with h5py.File(temp_file, "w") as f:
        valid = f.create_group("valid")
        invalid = f.create_group("invalid")
        trace_calc = f.create_group("trace_calculation")
        magn_cond = f.create_group("magneto_conductance")

        with open(os.path.join(dir_name, "Supplemental_Sawada_2016.txt"), "r") as f2:
            dt = pd.read_csv(f2, delim_whitespace=True)

        # Read the column from the Sawada paper reference
        number = dt["n"].values
        seed = dt["seed"].values
        n_scat = dt["n_scat"].values
        L = dt["L"].values
        S = dt["S"].values
        cosj = dt["cosj'"].values
        valid_traces = np.empty_like(number, dtype=np.bool)

        # Maximum number of scattering events to consider in a trajectory.
        # (set in Sawada 2016)
        n_scat_max = 5000

        # Distance in term of the mean free path use to identify a return to the
        # origin.
        d = 2.5e-5

        # Temporary storage for values stored in "trace_calculation" and
        # "magneto-conductance"
        indexes = []
        lengths = []
        c_phi = []
        c_3phi = []
        s_phi = []
        s_3phi = []

        j = 0
        print(f"starting generation of {len(number)} trajectories")
        for i in range(0, len(number)):
            if i % 1000 == 0:
                print(f"Generating trajectory {i+1} to {i+1000}")

            # Compute the number of scattering events required to come back to the
            # origin
            n_scat_cal = identify_trajectory(seed[i], n_scat_max, d)

            # Check if the trajectory match our expectation
            is_valid = n_scat_cal == n_scat[i]

            tj = generate_trajectory(seed[i], n_scat_cal)
            x = tj[0]
            y = tj[1]
            x[-1] = 0
            y[-1] = 0

            # Compute extra quantities
            l = find_each_length(x, y)
            # Introduce a random angle since in the presence of Rashba and
            # Dresselhaus the system is not invariant by rotation and
            # we need to average the initial direction.
            angle = find_each_angle(x, y)

            is_valid &= abs(np.sum(l) - L[i]) < 0.005
            is_valid &= abs(cos(angle[-1]) - cosj[i]) < 5e-6
            # XXX add surface based validation
            valid_traces[i] = is_valid

            # Randomize the angle to handle no rotational symmetric systems
            angle += random.uniform(0, 2 * pi)

            if is_valid:
                g1 = valid.create_group(f"n={i}")

                # Also add the data to be stored later
                indexes.append((j, j + len(l)))
                lengths.append(l)
                c_phi.append(np.cos(angle))
                c_3phi.append(np.cos(3 * angle))
                s_phi.append(np.sin(angle))
                s_3phi.append(np.sin(3 * angle))
                j += len(l)

            else:
                g1 = invalid.create_group(f"n={i}")

            # Store the data in the group matching the validity of the trajectory.
            dset = g1.create_dataset("x", data=x)

            dset = g1.create_dataset("y", data=y)

            g1.attrs["length"] = L[i]
            g1.attrs["calculated_length"] = np.sum(l)
            g1.attrs["n_scat"] = n_scat[i]
            g1.attrs["calculated_n_scat"] = n_scat_cal
            g1.attrs["seed"] = seed[i]
            g1.attrs["surface"] = S[i]
            g1.attrs["cosj'"] = cosj[i]
            g1.attrs["calculated_cosj"] = cos(angle[-1])

        # Store the parameters for the valid trajectories
        magn_cond.create_dataset("length", data=L[valid_traces])
        magn_cond.create_dataset("surface", data=S[valid_traces])
        magn_cond.create_dataset("cosj", data=cosj[valid_traces])

        trace_calc.create_dataset("index", data=np.array(indexes))
        trace_calc.create_dataset("l", data=np.concatenate(lengths, axis=None))
        trace_calc.create_dataset("c_phi", data=np.concatenate(c_phi, axis=None))
        trace_calc.create_dataset("c_3phi", data=np.concatenate(c_3phi, axis=None))
        trace_calc.create_dataset("s_phi", data=np.concatenate(s_phi, axis=None))
        trace_calc.create_dataset("s_3phi", data=np.concatenate(s_3phi, axis=None))

    os.rename(temp_file, os.path.join(dir_name, "trajectories_data.hdf5"))


def get_trajectory_data(data_name, trajectory_number: Optional[int] = None):
    """Retrieve trajectory data, computing them if no cache file exists.

    """
    path = os.path.join(os.path.dirname(__file__), "trajectories_data.hdf5")
    if not os.path.exists(path):
        create_all_data()

    if data_name == "all":
        f = h5py.File(path, "r")
        return f

    elif data_name == "trace":
        with h5py.File(path, "r") as f:
            group = f["trace_calculation"]
            index = group["index"][:]
            if trajectory_number:
                sl = slice(0, index[trajectory_number, 1])
            else:
                sl = slice(0, None)
            l = group["l"][sl]
            c_phi = group["c_phi"][sl]
            c_3phi = group["c_3phi"][sl]
            s_phi = group["s_phi"][sl]
            s_3phi = group["s_3phi"][sl]

        return index, l, c_phi, c_3phi, s_phi, s_3phi

    elif data_name == "mag":
        with h5py.File(path, "r") as f:
            group = f["magneto_conductance"]
            if trajectory_number:
                sl = slice(0, trajectory_number)
            else:
                sl = slice(0, None)
            l = group["length"][sl]
            s = group["surface"][sl]
            cosj = group["cosj"][sl]

        return l, s, cosj
