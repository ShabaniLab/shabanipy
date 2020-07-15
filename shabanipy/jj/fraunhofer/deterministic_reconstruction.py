# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Deterministic reconstruction of the j(x) in a JJ from a Fraunhofer pattern.

The method implemented here is based on:
Dynes, R. C. & Fulton, T. A.
Supercurrent Density Distribution in Josephson Junctions.
Phys. Rev. B 3, 3015–3023 (1971).

This method has the advantage of being algebraic but can suffer from a lack of
precision due to the finite extend of the measured Fraunhofer pattern.

We need to use the more complex approach of the paper since we are interested
in non-symmetric current distributions.

"""
from typing import Optional

import numpy as np
from scipy.integrate import romb

from shabanipy.utils.integrate import can_romberg, resample_evenly


def extract_theta(
    fields: np.ndarray,
    ics: np.ndarray,
    f2k: float,  # field-to-k conversion factor (i.e. beta/B)
    jj_width: float,
    use_interpolation: bool = True,
    interpolation_kind: str = "cubic",
    n_points: Optional[int] = None,
) -> np.ndarray:
    """Compute the Ic Hilbert transform.

    Parameters
    ----------
    fields : np.ndarray
        Magnetic field at which the critical current was measured. For ND input
        the sweep should occur on the last axis.
    ics : np.ndarray
        Measured value of the critical current. For ND input the sweep should
        occur on the last axis.
    use_interpolation : bool, optional
        Allow to resample the points using spline interpolation. This allows to
        use the more precise Romberg integration method instead of the Simpson
        one. The default is True.
    interpolation_kind : str, optional
        Order of the spline use in the interpolation (see `interp1d` for
        details), by default "cubic"
    n_points : Optional[int], optional
        Number of points to generate using the interpolation, by default None

    Returns
    -------
    np.ndarray
        Hilbert tranform of Ic to be used when rebuilding the current
        distribution.

    """
    if not can_romberg(fields):
        fine_fields, fine_ics = resample_evenly(fields, ics)
    else:
        fine_fields, fine_ics = fields, ics
    log_fine_ics = np.log(fine_ics)

    # scale from B to beta
    fields = fields * f2k
    fine_fields = fine_fields * f2k
    step = abs(fine_fields[0] - fine_fields[1])

    theta = np.empty_like(fields)
    for i, (field, ic) in enumerate(zip(fields, ics)):
        samples = log_fine_ics - np.log(ic)
        diff = field**2 - fine_fields**2
        diff[diff == 0] = 1e-9
        # TODO below is off by factor of 2 but gives the correct output
        theta[i] = (field / np.pi * romb(samples / diff, step)
                    - field * jj_width / 2)
    return theta


def extract_current_distribution(
    fields: np.ndarray,
    ics: np.ndarray,
    f2k: float,
    jj_width: float,
    jj_points: int,
    use_interpolation: bool = True,
    interpolation_kind: str = "cubic",
) -> np.ndarray:
    """Extract the current distribution from Ic(B).

    Parameters
    ----------
    fields : np.ndarray
        Magnetic field at which the critical current was measured. For ND input
        the sweep should occur on the last axis.
    ics : np.ndarray
        Measured value of the critical current.For ND input the sweep should
        occur on the last axis.
    f2k : float
        Field to wave-vector conversion factor. This can be estimated from the
        Fraunhofer periodicity.
    jj_width : float
        Size of the junction. The current distribution will be reconstructed on
        a larger region (2 * jj_width)
    jj_points : int
        Number of points used to describe the junction inside jj_width.
    use_interpolation : bool, optional
        Allow to resample the points using spline interpolation. This allows to
        use the more precise Romberg integration method instead of the Simpson
        one. The default is True.
    interpolation_kind : str, optional
        Order of the spline use in the interpolation (see `interp1d` for
        details), by default "cubic"
    n_points : Optional[int], optional
        Number of points to generate using the interpolation, by default None

    Returns
    -------
    np.ndarray
        Positions at which the current density was calculated. No matter the
        input shape the returned array is 1D.
    np.ndarray
        Current density.

    """
    if not can_romberg(fields):
        fine_fields, fine_ics = resample_evenly(fields, ics)
    else:
        fine_fields, fine_ics = fields, ics

    theta = extract_theta(fine_fields, fine_ics, f2k, jj_width)

    # scale from B to beta
    fine_fields = f2k*fine_fields
    step = abs(fine_fields[0] - fine_fields[1])

    xs = np.linspace(-jj_width, jj_width, int(2*jj_points))
    j = np.empty(xs.shape, dtype=complex)
    for i, x in enumerate(xs):
        j[i] = (1 / (2 * np.pi) * romb(fine_ics * np.exp(
                1j*(theta - fine_fields * x)
                ), step))
    return xs, j
