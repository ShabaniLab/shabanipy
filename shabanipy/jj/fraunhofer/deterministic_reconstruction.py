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
from matplotlib import pyplot as plt
from scipy.integrate import quad, romb
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from typing_extensions import Literal

from shabanipy.utils.integrate import can_romberg, resample_evenly


def extract_theta(
    fields: np.ndarray,
    ics: np.ndarray,
    f2k: float,
    jj_width: float,
    method: Optional[Literal["romb", "quad", "hilbert"]] = "hilbert",
) -> np.ndarray:
    """Compute the phase as the Hilbert transform of ln(I_c).

    Note the integral expressions for θ(β) below differ from Dynes and Fulton
    (1971) by a factor of 2, but gives the correct output. (The source of this
    discrepancy is as yet unknown. But note the Hilbert transform is usually
    defined, as below, with a prefactor 1/π.)

    Parameters
    ----------
    fields : np.ndarray
        Magnetic field at which the critical current was measured. For ND input
        the sweep should occur on the last axis.
    ics : np.ndarray
        Measured value of the critical current. For ND input the sweep should
        occur on the last axis.
    f2k : float
        Field-to-wavenumber conversion factor (i.e. β / B).
    jj_width: float
        Width of the junction transverse to the field and current.
    method: str, optional
        Method used to compute the phase; `romb` and `quad` specify
        numerical integration methods (note `quad` is particularly slow), while
        `hilbert` uses `scipy.signal.hilbert` to compute the discrete Hilbert
        transform.

    Returns
    -------
    np.ndarray
        Phase θ, the Hilbert transform of ln(I_c), to be used when rebuilding
        the current distribution. The phases are shifted by a factor
        `field * jj_width / 2` to give a reconstructed current density centered
        about the origin.

    """
    # scale B to beta first; then forget about it
    fields = fields * f2k

    if method == "hilbert":
        return np.imag(hilbert(np.log(ics))) - fields * jj_width / 2
    elif method == "romb":
        if not can_romberg(fields):
            fine_fields, fine_ics = resample_evenly(fields, ics)
        else:
            fine_fields, fine_ics = fields, ics
        step = abs(fine_fields[0] - fine_fields[1])

        def integrand(beta, ic):
            denom = beta ** 2 - fine_fields ** 2
            denom[denom == 0] = 1e-9
            return (np.log(fine_ics) - np.log(ic)) / denom

    elif method == "quad":
        fine_ics = interp1d(fields, ics, "cubic")

        def integrand(b, beta, ic):
            # quad will provide b when calling this
            denom = beta ** 2 - b ** 2
            if denom == 0:
                denom = 1e-9
            return (np.log(fine_ics(b)) - np.log(ic)) / denom

    else:
        raise ValueError(f"Integration method '{method}' unsupported")

    theta = np.empty_like(fields)
    for i, (field, ic) in enumerate(zip(fields, ics)):
        if method == "romb":
            theta[i] = (
                field / np.pi * romb(integrand(field, ic), step) - field * jj_width / 2
            )
        elif method == "quad":
            theta[i] = (
                field
                / np.pi
                * quad(integrand, np.min(fields), np.max(fields), args=(field, ic))[0]
                - field * jj_width / 2
            )
    return theta


def extract_current_distribution(
    fields: np.ndarray,
    ics: np.ndarray,
    f2k: float,
    jj_width: float,
    jj_points: int,
    debug: bool = False,
    theta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract the current distribution from Ic(B).

    Parameters
    ----------
    fields : np.ndarray
        1D array of the magnetic field at which the critical current was measured.
    ics : np.ndarray
        1D array of the measured value of the critical current.
    f2k : float
        Field to wave-vector conversion factor. This can be estimated from the
        Fraunhofer periodicity.
    jj_width : float
        Size of the junction. The current distribution will be reconstructed on
        a larger region (2 * jj_width)
    jj_points : int
        Number of points used to describe the junction inside jj_width.
    theta : np.ndarray, optional
        Phase distribution to use in the current reconstruction. If None, it
        will be extracted from the given Fraunhofer pattern.

    Returns
    -------
    np.ndarray
        Positions at which the current density was calculated.
    np.ndarray
        Current density.

    """
    if not can_romberg(fields):
        fine_fields, fine_ics = resample_evenly(fields, ics)
    else:
        fine_fields, fine_ics = fields, ics

    fine_ics[np.less_equal(fine_ics, 1e-10)] = 1e-10

    if theta is None:
        theta = extract_theta(fine_fields, fine_ics, f2k, jj_width)

    # scale from B to beta
    fine_fields = f2k * fine_fields
    step = abs(fine_fields[0] - fine_fields[1])

    xs = np.linspace(-jj_width, jj_width, int(2 * jj_points))
    j = np.empty(xs.shape, dtype=complex)
    for i, x in enumerate(xs):
        j[i] = (
            1
            / (2 * np.pi)
            * romb(fine_ics * np.exp(1j * (theta - fine_fields * x)), step)
        )

    if debug:
        f, axes = plt.subplots(2, 1)
        axes[0].plot(f2k * fields, ics)
        axes[1].plot(xs, j.real)
        plt.show()

    return xs, j.real
