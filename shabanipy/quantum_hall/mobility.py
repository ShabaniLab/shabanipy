# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Carriers mobility analysis.

"""
import numpy as np
import scipy.constants as cs


def extract_mobility(field, rxx, ryy, density, geometric_factor):
    """Compute the mobilities from transverse resistance measurements.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values for which the the longitudinal resistance was
        measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    rxx : np.ndarray
        Longitudinal resistance values which were measured along xx.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    ryy : np.ndarray
        Longitudinal resistance values which were measured along yy.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    density : float | np.ndarray
        Carriers density. If field is more than 1D the shape of this array
        should match the field.shape[:-1]

    Returns
    -------
    mobility : np.ndarray
        Array (2, ...) containing the xx and yy mobility

    """
    # Identify the shape of the data and make them suitable for the following
    # treatment.
    if len(field.shape) >= 2:
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxx = rxx.reshape((trace_number, -1))
        ryy = ryy.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        rxx = np.array((rxx,))
        ryy = np.array((ryy,))

    r0 = np.empty((2, trace_number))

    for i in range(trace_number):
        min_field_ind = np.argmin(np.abs(field[i]))
        r0[0, i] = rxx[i, min_field_ind]
        r0[1, i] = ryy[i, min_field_ind]

    r0.reshape((2, ) + original_shape)
    r0 *= geometric_factor
    return 1/cs.e/density/r0
