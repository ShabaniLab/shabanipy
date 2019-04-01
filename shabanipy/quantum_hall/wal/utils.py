# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak anti-localisation analysis utility function.

"""
import scipy.constants as cs
import matplotlib.pyplot as plt

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_no_dresselhaus \
    import compute_wal_conductance_difference as simple_wal
from .wal_full_diagonalization \
    import compute_wal_conductance_difference as full_wal


def compute_dephasing_time(dephasing_field, diffusion):
    """Compute the dephasing from the dephasing field.

    Parameters
    ----------
    dephasing_field : float | np.ndarray
        Dephasing field in T.
    diffusion : float | np.ndarray
        Diffusion coefficient.

    Returns
    -------
    dephasing_time : float | np.ndarray
        Dephasing time in ps.

    """
    return cs.hbar/(dephasing_field*4*cs.e*diffusion)*1e12


def compute_linear_soi(soi_field, mobility, density):
    """Compute the linear SOI term associated with a given effective field.

    Parameters
    ----------
    soi_field : float | np.ndarray
        Linear SOI field in T.
    mobility : float | np.ndarray
        Sample mobility.
    density : float | np.ndarray
        Sample density.

    Returns
    -------
    soi : float | np.ndarray
        Spin-orbit coupling in meV.

    """
    t_free = mean_free_time_from_mobility(mobilities)
    diff = diffusion_constant_from_mobility_density(mobilities, densities)
    return np.sqrt(soi_field*cs.e*diff*cs.hbar /
                   (np.pi*density*t_free) )*1e13/cs.e


def recenter_data(field, resistance):
    """[summary]

    Parameters
    ----------
    field : [type]
        [description]
    resistance : [type]
        [description]

    """

