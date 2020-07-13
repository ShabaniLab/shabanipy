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

import numpy as np
from scipy.interpolate import interp1d


def can_romberg(n_points: int) -> bool:
    """Determine if n_points is of the form 2**n + 1.

    Romberg integration requires 2**n + 1 samples.
    """
    return n_points > 1 and not (n_points - 1) & (n_points - 2)


def resample_data(x: np.ndarray,
                  y: np.ndarray,
                  n_points: int,
                  interp_kind: Optional[str] = 'cubic'
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate `n_points` evenly-spaced samples of y(x) by interpolation.

    Parameters
    ----------
    x : np.ndarray
        Values of the independent variable.
    y : np.ndarray
        Values of the dependent variable y(x) corresponding to `x`.
    n_points : int
        Number of evenly-spaced samples to generate.
    interp_kind : str, optional
        Kind of interpolator to use. See `interp1d` for details.

    Returns
    -------
    np.ndarray
        Resampled `x` array containing `n_points` samples.
    np.ndarray
        Resampled `y` array containing `n_points` samples.
    """
    y_func = interp1d(x, y, interp_kind)
    x_ = np.linspace(x[0], x[-1], n_points)
    y_ = y_func(x_)
    return x_, y_
