# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Generate a Fraunhofer pattern based on a current distribution."""
import warnings

import numpy as np
from numba import cfunc
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import romb, quad, IntegrationWarning
from typing_extensions import Literal

from shabanipy.utils.integrate import resample_evenly, can_romberg

warnings.filterwarnings("ignore", category=IntegrationWarning)


def generate_current_integrand(
    width: float,
    field_to_k_factor: float,
    current_distribution: np.ndarray,
    phase_distribution: np.ndarray,
):
    """Integrand to compute the current through a junction at a given field.

    """
    step = width / len(current_distribution)

    @cfunc(float64(intc, CPointer(float64)))
    def real_current_integrand(n, args):
        """Cfunc to be used with quad to calculate the current from the distribution.

        """
        pos, field = args[0], args[1]
        x = int(pos // step)
        return current_distribution[x] * np.cos(
            (phase_distribution[x] + field_to_k_factor * pos) * field
        )

    @cfunc(float64(intc, CPointer(float64)))
    def imag_current_integrand(n, args):
        """Cfunc to be used with quad to calculate the current from the distribution.

        """
        pos, field = args[0], args[1]
        x = int(pos // step)
        return current_distribution[x] * np.sin(
            (phase_distribution[x] + field_to_k_factor * pos) * field
        )

    return (
        LowLevelCallable(real_current_integrand.ctypes),
        LowLevelCallable(imag_current_integrand.ctypes),
    )


def produce_fraunhofer(
    magnetic_field: np.ndarray,
    jj_size: float,
    field_to_k_factor: np.ndarray,
    current_distribution: np.ndarray,
    method: Literal["quad", "romb"] = "quad",
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    n_points: int = 2 ** 10 + 1,
) -> np.ndarray:
    """Compute the Fraunhoffer pattern for a given current density.

    WARNING: This is not currently tested or maintained. Use
    produce_fraunhofer_fast() instead.

    The Fraunhofer pattern is normalized by the size of the junction.
    This is independent of the current distribution used or of the junction size.

    Parameters
    ----------
    magnetic_field : np.ndarray
        Fields at which to compute the Fraunhoffer pattern.

    jj_size : float
        Size of the junction.

    field_to_k_factor : float
        Local field to wavevector conversion factor. B = k ϕ_0 /(2π L_eff)

    current_distribution : np.ndarray
        Density of current in the junction.

    method : {"quad", "romb"}, optional
        Numerical integration method to use.

    eps_abs : float
        Absolute error to use in quad.

    eps_rel : float
        Relative error to use in quad.

    n_points : int
        Number of points to use in romb.

    """
    cd, pd = np.asarray(current_distribution), np.asarray(phase_distribution)

    if method == "quad":
        norm = 1 / jj_size
        re, im = generate_current_integrand(jj_size, field_to_k_conversion, cd, pd)
        f = np.empty_like(magnetic_field)
        for i, b in enumerate(magnetic_field):
            f[i] = np.abs(
                quad(re, 0, jj_size, (b,), epsrel=eps_rel, epsabs=eps_abs)[0]
                + 1j * quad(im, 0, jj_size, (b,), epsrel=eps_rel, epsabs=eps_abs)[0]
            )

        return f * norm

    elif method == "romb":
        return produce_fraunhofer_fast(
            magnetic_field, field_to_k_conversion, jj_size, cd, pd, n_points
        )

    else:
        raise ValueError("Unsupported method")


def produce_fraunhofer_fast(
    magnetic_field: np.ndarray,
    f2k: float,  # field-to-wavevector conversion factor
    cd: np.ndarray,  # current distribution
    xs: np.ndarray,
) -> np.ndarray:
    """Generate Fraunhofer from current density using Romberg integration."""
    current = np.empty_like(magnetic_field)
    if not can_romberg(xs):
        xs, cd = resample_evenly(xs, cd)

    dx = abs(xs[0] - xs[1])
    for i, field in enumerate(magnetic_field):
        current[i] = np.abs(romb(cd * np.exp(1j * f2k * field * xs), dx))

    return current
