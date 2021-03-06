# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Numerical integration routines and helpers."""
from typing import Optional, Tuple

from numba import njit
import numpy as np
from scipy.interpolate import interp1d


def can_romberg(x: np.ndarray) -> bool:
    """Determine if data over domain `x` can be Romberg integrated.

    Data over domain `x` can be Romberg integrated if:
    1) There are 2**k + 1 values of `x` for some nonnegative integer k.
    2) The values of `x` are evenly spaced.
    """
    dx = np.diff(x)
    return len(x) > 1 and not (len(x) - 1) & (len(x) - 2) and np.allclose(dx, dx[0])


def resample_evenly(
    x: np.ndarray,
    y: np.ndarray,
    n_points: Optional[int] = None,
    interp_kind: Optional[str] = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate `n_points` evenly-spaced samples of y(x) by interpolation.

    Parameters
    ----------
    x : np.ndarray
        Values of the independent variable.
    y : np.ndarray
        Values of the dependent variable y(x) corresponding to `x`.
    n_points : int, optional
        Number of evenly-spaced samples to generate. If None, generate 1+2**k
        samples for Romberg integration, where k is the smallest integer such
        that the output contains more samples than the input.
    interp_kind : str, optional
        Kind of interpolator to use. See `interp1d` for details.

    Returns
    -------
    np.ndarray
        Resampled `x` array containing `n_points` samples.
    np.ndarray
        Resampled `y` array containing `n_points` samples.
    """
    n_points = n_points if n_points else 1 + 2 ** (1 + int(np.log2(len(x))))
    y_func = interp1d(x, y, interp_kind)
    x_ = np.linspace(x[0], x[-1], n_points)
    y_ = y_func(x_)
    return x_, y_


@njit(cache=True, fastmath=True)
def romb_1d(y, dx):
    """Romberg integration of 1D data.

    This is a specialized implementation of the romb algorithm found in scipy
    for 1d arrays. Requires 2**k + 1 samples.
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
