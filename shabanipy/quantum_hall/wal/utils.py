# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak anti-localisation analysis utility function.

"""
from scipy.scipy import argrelmin
import scipy.constants as cs
import matplotlib.pyplot as plt

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_no_dresselhaus \
    import compute_wal_conductance_difference as simple_wal
from .wal_full_diagonalization \
    import compute_wal_conductance_difference as full_wal


def compute_dephasing_time(dephasing_field, diffusion):
    """Compute the dephasing from the dephasing field.

    Parameters
    ----------
    dephasing_field : float | np.ndarray
        Dephasing field in T.
    diffusion : float | np.ndarray
        Diffusion coefficient.

    Returns
    -------
    dephasing_time : float | np.ndarray
        Dephasing time in ps.

    """
    return cs.hbar/(dephasing_field*4*cs.e*diffusion)*1e12


def compute_linear_soi(soi_field, mobility, density):
    """Compute the linear SOI term associated with a given effective field.

    Parameters
    ----------
    soi_field : float | np.ndarray
        Linear SOI field in T.
    mobility : float | np.ndarray
        Sample mobility.
    density : float | np.ndarray
        Sample density.

    Returns
    -------
    soi : float | np.ndarray
        Spin-orbit coupling in meV.

    """
    t_free = mean_free_time_from_mobility(mobilities)
    diff = diffusion_constant_from_mobility_density(mobilities, densities)
    return np.sqrt(soi_field*cs.e*diff*cs.hbar /
                   (np.pi*density*t_free) )*1e13/cs.e


def recenter_wal_data(field, resistance, fraction=0.4):
    """Recenter WAL data around zero field where we expect an extremum.

    We look for the local minima of the resistance around zero field (between
    ± max(abs(B))*fraction). If we find one (edges are not detected), we use
    it to re-center the data. If we find more than one, we use the deepest
    minimum. If we find none, we look for the maximum in the same interval.
    The data should be smooth in the considered interval for this method to
    give good results.

    Parameters
    ----------
    field : np.ndarray
        Magnetic fields at which the sample resistance was measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    resistance : np.ndarray
        Resistance values in Ω which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    fraction : float
        Fraction of the total field excursion in which to look for the WAL
        resistance minimum.

    Returns
    -------
    field : np.ndarray
        Magnetic fields such that 0 field corresponds to the WAL minimum.
    resistance : np.ndarray
        Resistance values in Ω which were measured.

    """
    if len(field.shape) >= 2:
        original_shape = field.shape
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        resistance = resistance.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        rxy = np.array((rxy,))

    for i, (f, r) in enumerate(zip(field, resistance)):
        max_field = np.max(f)*fraction
        mask = np.where(np.less(np.abs(f), max_field))
        masked_r = r[mask]
        mins, = argrelmin(resistance[mask])  # We need to unpack the tuple
        # No local minima were found, look for a maximum
        if not mins:
            center_index = np.argmax(masked_r)
        else:
            if len(mins) == 1:
                center_index = mins[0]
            else:
                center_index = mins[np.argmin([masked_r[i] for i in mins])]

        fields[i] -= f[center_index]

    return field.reshape(original_shape), resistance.reshape(original_shape)


def symmetrize_wal_data(field, resistance, mode='average'):
    """Symmetrize WAL data with respect to 0 field.

    This method assumes that the data have been properly recentered around 0.
    Points that cannot be symmetrized are left untouched.

    Parameters
    ----------
    field : np.ndarray
        Magnetic fields at which the sample resistance was measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    resistance : np.ndarray
        Resistance values in Ω which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    mode : {'average', 'positive', 'negative'}, optional
        Symmetrize the data by either flipping them and averaging on keeping
        only the positive/negative field part of the data.

    Returns
    -------
    field : np.ndarray
        Magnetic fields such that 0 field corresponds to the WAL minimum.
    resistance : np.ndarray
        Resistance values symmetrized with respect to 0 field.

    """
    if len(field.shape) >= 2:
        original_shape = field.shape
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        resistance = resistance.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        rxy = np.array((rxy,))

    for i, (f, r) in enumerate(zip(field, resistance)):
        center = np.argmin(np.abs(f))
        neg_len = len(r[:center])
        pos_len = len(r[center+1:])
        if pos_len == neg_len:
            if mode == 'average':
                field[i] = (f + f[::-1])/2
            elif mode == 'positive':
                field[i][:center] = reversed(r[center+1:])
            else:
                field[i][center+1:] = reversed(r[:center])
        elif len(pos_len) > len(neg_len):
            neg_r = r[:center].copy()
            pos_r = r[center+1:center+1+neg_len].copy()
            if mode == 'average':
                field[:center] += pos_r
                field[:center] /= 2
                field[center+1:center+1+neg_len] += neg_r
                field[center+1:center+1+neg_len] /= 2
            elif mode == 'positive':
                field[:center] = pos_r
            else:
                field[center+1:center+1+neg_len] = neg_r
        else:
            neg_r = r[abs(pos_len-center):center].copy()
            pos_r = r[center+1:].copy()
            if mode == 'average':
                field[abs(pos_len-center):center] += pos_r
                field[abs(pos_len-center):center] /= 2
                field[center+1:] += neg_r
                field[center+1:] /= 2
            elif mode == 'positive':
                field[abs(pos_len-center):center] = pos_r
            else:
                field[center+1:] = neg_r

    return field.reshape(original_shape), resistance.reshape(original_shape)
