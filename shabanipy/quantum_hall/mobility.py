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
import scipy.constants as cs


def compute_mobility(field, rxx, ryy, density, geometric_factor):
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

    """
    min_field_ind = np.argmin(np.abs(field), axis=-1)
    r0 = np.array((rxx[..., min_field_ind], ryy[..., min_field_ind]))
    cr0 = r0*geometric_factor
    return 1/cs.e/density/cr0
