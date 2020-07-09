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
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romb, simps


def is_compatible_with_romberg(n_points: int) -> bool:
    """Determine if a number of points is of the form 2**n + 1

    We need that kind of number for intergrating using the Romberg method.

    """
    return n_points - 1 > 0 and not (n_points - 1 & (n_points - 2))


def generate_finer_data(
    fields: np.ndarray,
    ics: np.ndarray,
    interpolation_kind: str = "cubic",
    n_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate finer data for fields and ics with 2**n + 1.

    Parameters
    ----------
    fields : np.ndarray
        Magnetic field at which the critical current was measured. For ND input
        the sweep should occur on the last axis.
    ics : np.ndarray
        Measured value of the critical current. For ND input the sweep should
        occur on the last axis.
    interpolation_kind : str, optional
        Order of the spline use in the interpolation (see `interp1d` for
        details), by default "cubic"
    n_points : Optional[int], optional
        Number of points to generate using the interpolation, by default None

    Returns
    -------
    np.ndarray
        Fields interpolated to have n_points on the last axis.
    np.ndarray
        Critical currents interpolated to have n_points on the last axis.

    """
    if not is_compatible_with_romberg:
        raise ValueError("n_points should of the form 2**n + 1")

    # Create a finer ic and field to use in the integration
    fine_fields = np.empty(fields.shape[:-1] + (n_points,))
    log_fine_ics = np.empty_like(fine_fields)
    with np.nditer(
        (fields, fine_fields, ics, fine_ics),
        flags=["external_loop"],
        op_flags=[["readonly"], ["readwrite"], ["readonly"], ["writeonly"]],
    ) as it:
        for inner_fields, inner_fine_fields, inner_ics, inner_log_fine_ics in it:
            inner_fine_fields[:] = np.linspace(
                inner_fields[0], inner_fields[-1], n_points
            )
            ic_func = interp1d(inner_fields, inner_ics, interpolation_kind)
            step = abs(fine_fields[0] - fine_fields[1])
            inner_log_fine_ics[:] = ic_func(inner_fine_fields)

    return fine_fields, fine_ics


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
            n_points = 2 ** (int(np.log(len(fields), 2)) + 1) + 1
        if not is_compatible_with_romberg(fields.shape[-1]):
            fine_fields, fine_ics = generate_finer_data(
                fields, ics, interpolation_kind, n_points
            )
            log_fine_ics = np.log(fine_ics)
    else:
        # If the data are properly sampled use romb even if we did not interpolate.
        use_romb = n_points - 1 > 0 and not (n_points - 1 & (n_points - 2))
        fine_fields = fields
        log_fine_ics = np.log(ics)

    theta = np.empty_like(fields)
    with np.nditer(
        (fields, ics, theta),
        flags=["external_loop"],
        op_flags=["readonly", "readonly", "writeonly"],
    ) as it:
        for inner_field, inner_ics, theta in it:
            for i, (field, ic) in enumerate(zip(fields, ics)):
                samples = log_fine_ics - np.log(ic)
                diff = fine_fields ** 2 - field ** 2
                # Replace zeros in the denominators by a small value
                # (the numerator should be zero too at those points anyway).
                mask = np.logical_not(np.nonzero(diff))
                diff[mask] = 1e-9

                if use_romb:
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
    interpolation_points: Optional[int] = None,
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
    interpolation_points : Optional[int], optional
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
            n_points = 2 ** (int(np.log(len(fields), 2)) + 1) + 1
        if not is_compatible_with_romberg(fields.shape[-1]):
            fine_fields, fine_ics = generate_finer_data(
                fields, ics, interpolation_kind, n_points
            )
            log_fine_ics = np.log(fine_ics)
    else:
        # If the data are properly sampled use romb even if we did not interpolate.
        use_romb = n_points - 1 > 0 and not (n_points - 1 & (n_points - 2))
        fine_fields = fields
        log_fine_ics = ics

    # First compute the Hilbert transform of Ic
    theta = extract_theta(fine_fields, fine_ics)

    xs = np.linspace(
        -jj_size * 1.25 / 2, jj_size * 1.25 / 2, jj_points + (jj_points - 1) // 4
    )
    step = xs[1] - xs[0]
    j = np.empty(fields.shape[:-1] + (len(xs),))

    with np.nditer(
        (fields, ics, theta, j),
        flags=["external_loop"],
        op_flags=["readonly", "readonly", "readonly", "writeonly"],
    ) as it:
        for inner_fields, inner_ics, inner_theta, inner_j in it:
            for i, x in enumerate(xs):
                inner_j[i] = romb(
                    inner_ics
                    * np.exp(
                        1j
                        * (
                            inner_theta
                            - conversion_factor
                            * inner_fields
                            * (x + jj_size * 1.25 / 2)
                        )
                    ),
                    step,
                )

    return xs, j
