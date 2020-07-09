# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Convert between Fraunhofer patterns and supercurrent distributions.

This module converts back and forth between:
    a) The critical dc Josephson current as a function of the perpendicularly
        applied magnetic field strength (i.e. a "Fraunhofer pattern"); and
    b) The supercurrent density distribution in the junction as a function of
        the spatial dimension transverse to both the current and field.

References:
    [1] Dynes and Fulton, Phys. Rev. B 3, 3015 (1971)
"""
import numpy as np
from numpy.fft import rfft
from scipy.signal import hilbert


def fraunhofer(bfield, current_dist, jj_width, jj_length, london_depth=0.0):
    """Construct Fraunhofer pattern from supercurrent distribution.

    This implements the Fourier transform defined in Eqs. (3) and (4) of [1].
    Note the discrepancy in sign convention between Eq. (3) and np.fft is
    resolved upon taking the absolute value in Eq. (4).

    Arguments:
        bfield (np.ndarray):
            Magnetic field values at which to compute Fraunhofer.
        current_dist (np.ndarray):
            Distribution J(x) of current in the junction.
        jj_width (float):
            Width a of junction along x (perpendicular to current and field).
        jj_length (float):
            Distance d between superconducting contacts (i.e. along current).
            The effective length is (2λ + d) where λ is london_depth.
        london_depth (float):
            London penetration depth λ of the superconductors; see jj_length.

    Returns (np.ndarray):
        Critical current values I_c(B) at corresponding values of bfield.
    """
    # TODO: determine x and B scales of discrete Fourier transform (output step
    # size is reciprocal of input window width)
    #beta = 2*np.pi * (2*london_depth + jj_length) * bfield / flux_quantum
    return np.abs(rfft(current_dist))


def phase(bfield, crit_curr):
    """Reconstruct phase of Fourier coefficients from their absolute values.

    This implements the Hilbert transform defined in Eq. (5) of [1].

    Arguments:
        bfield (np.ndarray):
            Magnetic field values at which to compute Fourier phases.
        crit_curr (np.ndarray):
            Critical current values I_c(B); i.e. Fraunhofer pattern.

    Returns (np.ndarrary):
        TODO
    """
    # TODO most certainly wrong
    return np.angle(hilbert(crit_curr))
