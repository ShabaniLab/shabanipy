# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Generate a Fraunhofer pattern based on a current distribution..

"""
import warnings

import numpy as np
from numba import cfunc, njit
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad, romb, IntegrationWarning

from dynesty import DynamicNestedSampler

warnings.filterwarnings('ignore', category=IntegrationWarning)


@njit(cache=True, fastmath=True)
def romb_1d(y, dx):
    """Specialized implementation of the romb algorithm found in scipy for 1d arrays.

    """
    n_interv = len(y)-1
    n = 1
    k = 0
    while n < n_interv:
        n <<= 1
        k += 1
    if n != n_interv:
        raise ValueError("Number of samples must be one plus a "
                         "non-negative power of 2.")

    R = np.empty((k+1, k+1))
    h = n_interv * dx
    R[0, 0] = (y[0] + y[-1])/2.0 * h
    start = stop = step = n_interv
    for i in range(1, k+1):
        start >>= 1
        R[(i, 0)] = 0.5*(R[i-1, 0] + h*y[start:stop:step].sum())
        step >>= 1
        for j in range(1, i+1):
            prev = R[i, j-1]
            R[i, j] = prev + (prev - R[i-1, j-1]) / ((1 << (2*j))-1)
        h /= 2.0

    return R[k, k]


@njit(cache=True, fastmath=True)
def sample_integrands(n_points: int,
                      field: np.ndarray,
                      field_to_k_factor: float,
                      jj_size: float,
                      current_distribution: np.ndarray,
                      phase_distribution: np.ndarray,
                      ) -> np.ndarray:
    """Sample the integrands using n_points to use romb to integrate.

    """
    step = jj_size/len(current_distribution)
    pos = np.linspace(0, jj_size, n_points)
    indexes = (pos // step).astype(np.int64)
    cd, pd = np.empty_like(pos), np.empty_like(pos)
    for i, x in enumerate(indexes):
        cd[i] = current_distribution[x]
        pd[i] = phase_distribution[x]
    return (cd * np.cos((pd + field_to_k_factor*pos)*field),
            cd * np.sin((pd + field_to_k_factor*pos)*field))


def generate_current_integrand(width: float,
                               field_to_k_factor: float,
                               current_distribution: np.ndarray,
                               phase_distribution: np.ndarray):
    """Integrand to compute the current through a junction at a given field.

    """
    step = width/len(current_distribution)

    @cfunc(float64(intc, CPointer(float64)))
    def real_current_integrand(n, args):
        """Cfunc to be used with quad to calculate the current from the distribution.

        """
        pos, field = args[0], args[1]
        x = int(pos // step)
        return (current_distribution[x] *
                np.cos((phase_distribution[x] + field_to_k_factor*pos)*field))

    @cfunc(float64(intc, CPointer(float64)))
    def imag_current_integrand(n, args):
        """Cfunc to be used with quad to calculate the current from the distribution.

        """
        pos, field = args[0], args[1]
        x = int(pos // step)
        return (current_distribution[x] *
                np.sin((phase_distribution[x] + field_to_k_factor*pos)*field))

    return (LowLevelCallable(real_current_integrand.ctypes),
            LowLevelCallable(imag_current_integrand.ctypes))


def produce_fraunhofer(magnetic_field: np.ndarray,
                       field_to_k_conversion: float,
                       jj_size: float,
                       current_distribution: np.ndarray,
                       phase_distribution: np.ndarray,
                       method: str="quad",
                       eps_abs: float=1e-6,
                       eps_rel: float=1e-6,
                       n_points: int=2**10 + 1,
                       ) -> np.ndarray:
    """Compute the Fraunhoffer pattern for a given current density.

    Parameters
    ----------
    magnetic_field: np.ndarray
        Fields at which to compute the Fraunhoffer pattern.

    b_to_k_conversion: float
        Field to wavevector conversion factor. B = k ϕ_0 /(2π L_eff)

    jj_size: float
        Size of the junction.

    current_distribution: np.ndarray
        Density of current in the junction.

    phase_distribution: np.ndarray
        Field to phase conversion distribution. The first element is expect to
        always be 0 since we need a phase reference.

    eps_abs: float
        Absolute error to use in quad.

    eps_rel: float
        Relative error to use in quad.

    """
    cd, pd = np.asarray(current_distribution), np.asarray(phase_distribution)

    if method == "quad":
        re, im = generate_current_integrand(jj_size, field_to_k_conversion, cd, pd)
        f = np.empty_like(magnetic_field)
        for i, b in enumerate(magnetic_field):
            f[i] = np.abs(quad(re, 0, jj_size, (b,), epsrel=eps_rel, epsabs=eps_abs)[0] +
                        1j * quad(im, 0, jj_size, (b,), epsrel=eps_rel, epsabs=eps_abs)[0]
                        )

        return f

    elif method == "romb":
        return produce_fraunhofer_fast(magnetic_field, field_to_k_conversion, jj_size,
                                       cd, pd, n_points)

    else:
        raise ValueError("Unsupported method")


@njit(cache=True, fastmath=True)
def produce_fraunhofer_fast(magnetic_field: np.ndarray,
                            field_to_k_conversion: float,
                            jj_size: float,
                            current_distribution: np.ndarray,
                            phase_distribution: np.ndarray,
                            n_points: int,
                            ) -> np.ndarray:
    f = np.empty_like(magnetic_field)
    step_size = jj_size/(n_points - 1)
    for i, b in enumerate(magnetic_field):
        samples = sample_integrands(n_points, b, field_to_k_conversion,
                                    jj_size, current_distribution, phase_distribution)
        f[i] = np.abs(romb_1d(samples[0], step_size) +
                      1j*romb_1d(samples[1], step_size)
                      )

    return f
