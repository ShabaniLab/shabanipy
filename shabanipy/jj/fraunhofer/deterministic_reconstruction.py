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
import warnings
from math import pi
from typing import Optional

import numpy as np
from scipy.integrate import romb, simps

from shabanipy.utils.integrate import can_romberg, resample_evenly


def extract_theta(
    fields: np.ndarray,
    ics: np.ndarray,
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
        Hilbert tranform of Ic to be used when rebuilding the current distribution.

    """
    if use_interpolation:
        use_romb = use_interpolation
        if n_points is None:
            # Need 2**n + 1 for romb integration
            n_points = 2 ** (int(np.log2(len(fields))) + 1) + 1

        if not can_romberg(fields):
            fine_fields, fine_ics = resample_evenly(fields, ics, n_points,
                                                  interpolation_kind)
        else:
            fine_fields, fine_ics = fields, ics

        log_fine_ics = np.log(fine_ics)
    else:
        # If the data are properly sampled use romb even if we did not interpolate.
        use_romb = can_romberg(fields)
        fine_fields = fields
        log_fine_ics = np.log(ics)

    theta = np.empty_like(fields)
    with np.nditer(
        (fields, ics, theta),
        flags=["external_loop"],
        op_flags=(["readonly"], ["readonly"], ["writeonly"]),
    ) as it:
        for inner_field, inner_ics, theta in it:
            for i, (field, ic) in enumerate(zip(fields, ics)):
                samples = log_fine_ics - np.log(ic)
                diff = fine_fields ** 2 - field ** 2
                # Replace zeros in the denominators by a small value
                # (the numerator should be zero too at those points anyway).
                diff[diff == 0] = 1e-9

                if use_romb:
                    step = abs(fine_fields[0] - fine_fields[1])
                    theta[i] = field / (2 * pi) * romb(samples / diff, step)
                else:
                    theta[i] = field / (2 * pi) * simps(samples / diff, fields)

    return theta


def extract_current_distribution(
    fields: np.ndarray,
    ics: np.ndarray,
    conversion_factor: float,
    jj_size: float,
    jj_points: int,
    use_interpolation: bool = True,
    interpolation_kind: str = "cubic",
    n_points: Optional[int] = None,
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
    conversion_factor : float
        Field to wave-vector conversion factor. This can be estimated from the
        Fraunhofer periodicity.
    jj_size : float
        Size of the junction. The current distribution will be reconstructed on
        a slightly larger region (1.25 * jj_size)
    jj_points : int
        Number of points used to describe the junction inside jj_size.
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
        Positions at which the current density was calculated. No matter the input
        shape the returned array is 1D.
    np.ndarray
        Current density.

    """
    if use_interpolation:
        use_romb = use_interpolation
        if n_points is None:
            # Need 2**n + 1 for romb integration
            n_points = 2 ** (int(np.log2(len(fields))) + 1) + 1
        if not can_romberg(fields):
            fine_fields, fine_ics = resample_evenly(fields, ics, n_points,
                                                  interpolation_kind)
            log_fine_ics = np.log(fine_ics)
    else:
        # If the data are properly sampled use romb even if we did not interpolate.
        use_romb = can_romberg(fields)
        fine_fields = fields
        log_fine_ics = ics

    # First compute the Hilbert transform of Ic
    theta = extract_theta(fine_fields, fine_ics)

    xs = np.linspace(-jj_size * 1.25 / 2, jj_size * 1.25 / 2,
                     int(jj_points*5/4))
    step = xs[1] - xs[0]
    j = np.empty(xs.shape, dtype=complex)
    for i, x in enumerate(xs):
        j[i] = romb(fine_ics*np.exp(1j*(theta
                                    - conversion_factor
                                    * fine_fields * x)),
                    step)

    return xs, j
