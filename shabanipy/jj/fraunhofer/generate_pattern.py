# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Generate a Fraunhofer pattern based on a current distribution..

"""
import warnings
from typing import Union, Tuple

import numpy as np
from numba import cfunc, njit
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad, romb, IntegrationWarning
from typing_extensions import Literal

warnings.filterwarnings("ignore", category=IntegrationWarning)


@njit(cache=True, fastmath=True)
def romb_1d(y, dx):
    """Specialized implementation of the romb algorithm found in scipy for 1d arrays.

    This is a fast and accurate integration method for 2**k+1 points.

    """
    n_interv = len(y) - 1
    n = 1
    k = 0
    while n < n_interv:
        n <<= 1
        k += 1
    if n != n_interv:
        raise ValueError(
            "Number of samples must be one plus a " "non-negative power of 2."
        )

    R = np.empty((k + 1, k + 1))
    h = n_interv * dx
    R[0, 0] = (y[0] + y[-1]) / 2.0 * h
    start = stop = step = n_interv
    for i in range(1, k + 1):
        start >>= 1
        R[(i, 0)] = 0.5 * (R[i - 1, 0] + h * y[start:stop:step].sum())
        step >>= 1
        for j in range(1, i + 1):
            prev = R[i, j - 1]
            R[i, j] = prev + (prev - R[i - 1, j - 1]) / ((1 << (2 * j)) - 1)
        h /= 2.0

    return R[k, k]


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


@njit(cache=True, fastmath=True)
def resample_distribution(
    n_points: int,
    jj_size: float,
    field_to_k_factor: np.ndarray,
    current_distribution: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the integrands using n_points to use romb to integrate.

    Parameters
    ----------
    n_points : int
        Number of points to use to represent the junction in real space.
    jj_size : float
        Size of the junction (in whatever unit).
    field_to_k_factor : np.ndarray
        Local field to wavevector conversion factor. B = k ϕ_0 /(2π L_eff)
    current_distribution : np.ndarray
        Density of current in the junction.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Resampled field to wavevector and current distribution to match the
        number of points.

    """
    # Convert the position in indexes to access the current and the local
    # field to wavevector conversion ie local penetration length acounting for
    # flux focusing.
    step = jj_size / len(current_distribution)
    pos = np.linspace(0, jj_size, n_points)
    indexes = (pos // step).astype(np.int64)

    # Resample the current and field conversion to match the position.
    cd, f2k = np.empty_like(pos), np.empty_like(pos)
    for i, x in enumerate(indexes):
        cd[i] = current_distribution[x]
        f2k[i] = field_to_k_factor[x]

    return cd, f2k


@njit(cache=True, fastmath=True)
def produce_fraunhofer_fast(
    magnetic_fields: np.ndarray,
    jj_size: float,
    field_to_k_conversion: np.ndarray,
    current_distribution: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Fast version of produce_fraunhofer relying on Romberg integration method.

    """
    current = np.empty_like(magnetic_field)
    step_size = jj_size / (n_points - 1)
    if len(current_distribution) != n_points:
        cd, f2k = resample_distribution(
            n_points, jj_size, field_to_k_conversion, current_distribution
        )
    else:
        cd, f2k = current_distribution, field_to_k_conversion
    for i, field in enumerate(magnetic_field):
        current[i] = np.abs(
            romb_1d(cd * np.cos(f2k * pos * field), step_size)
            + 1j * romb_1d(cd * np.sin(f2k * pos * field), step_size)
        )

    return current / jj_size
