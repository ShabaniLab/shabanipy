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
from typing import Optional, Tuple

import numpy as np
from numba import cfunc
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.integrate import IntegrationWarning, quad, romb
from typing_extensions import Literal

from shabanipy.utils.integrate import can_romberg, resample_evenly

warnings.filterwarnings("ignore", category=IntegrationWarning)


def fraunhofer(
    magnetic_field: np.ndarray,
    f2k: float,  # field-to-wavevector conversion factor
    cd: np.ndarray,  # current distribution
    xs: np.ndarray,
    ret_fourier: Optional[bool] = False,
) -> np.ndarray:
    """Generate Fraunhofer from current density using Romberg integration.

    If ret_fourier is True, return the Fourier transform instead of the
    critical current (useful for debugging).
    """
    g = np.empty_like(magnetic_field, dtype=complex)
    if not can_romberg(xs):
        xs, cd = resample_evenly(xs, cd)

    dx = abs(xs[0] - xs[1])
    for i, field in enumerate(magnetic_field):
        g[i] = romb(cd * np.exp(1j * f2k * field * xs), dx)

    return g if ret_fourier else np.abs(g)


def _fraunhofer_dft(
    j: np.ndarray, dx: float = 1, f2k: float = 1, ret_fourier: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Fraunhofer from current density using discrete Fourier transform."""
    j_fourier = fftshift(fft(j))
    b = 2 * np.pi / f2k * fftshift(fftfreq(len(j), dx))
    return (b, j_fourier) if ret_fourier else (b, np.abs(j_fourier))
