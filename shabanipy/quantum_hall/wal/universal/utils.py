# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Utility conversion functions.

"""
import scipy.constants as cs


def linear_soi_to_linear_theta(soi: float, kf: float, t_free: float) -> float:
    """Convert a linear SOI term to an angle per unit of mean free path.

    Parameters
    ----------
    soi : float
        SOI in SI units.
    kf : float
        Fermi wave vector SI units.
    t_free : float
        Mean free time in SI units.

    Returns
    -------
    float
        SOI as an angle describing the induced rotation per unit of the mean free path.

    """
    return soi * (2 * kf * t_free / cs.hbar)


def linear_theta_to_linear_soi(theta: float, kf: float, t_free: float) -> float:
    """Convert an angle per unit of mean free path into a linear SOI term.

    Parameters
    ----------
    theta : float
        SOI as an angle describing the induced rotation per unit of the mean free path.
    kf : float
        Fermi wave vector SI units.
    t_free : float
        Mean free time in SI units.

    Returns
    -------
    float
        SOI in SI units.

    """
    return theta / (2 * kf * t_free / cs.hbar)


def cubic_soi_to_cubic_theta(soi: float, kf: float, t_free: float) -> float:
    """Convert a cubic SOI term to an angle per unit of mean free path.

    Parameters
    ----------
    soi : float
        Cubic SOI in SI units.
    kf : float
        Fermi wave vector SI units.
    t_free : float
        Mean free time in SI units.

    Returns
    -------
    float
        SOI as an angle describing the induced rotation per unit of the mean free path.

    """
    return soi * (2 * kf ** 3 * t_free / cs.hbar)


def cubic_theta_to_cubic_soi(theta: float, kf: float, t_free: float) -> float:
    """Convert an angle per unit of mean free path into a cubic SOI term.

    Parameters
    ----------
    theta : float
        SOI as an angle describing the induced rotation per unit of the mean free path.
    kf : float
        Fermi wave vector SI units.
    t_free : float
        Mean free time in SI units.

    Returns
    -------
    float
        SOI in SI units.

    """
    return theta / (2 * kf ** 3 * t_free / cs.hbar)
