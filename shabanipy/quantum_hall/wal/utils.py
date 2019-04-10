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
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
from scipy.signal import argrelmin, savgol_filter

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_full_diagonalization import \
    compute_wal_conductance_difference as full_wal
from .wal_no_dresselhaus import \
    compute_wal_conductance_difference as simple_wal


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


def compute_linear_soi(soi_field, mobility, density, effective_mass):
    """Compute the linear SOI term associated with a given effective field.

    Parameters
    ----------
    soi_field : float | np.ndarray
        Linear SOI field in T.
    mobility : float | np.ndarray
        Sample mobility.
    density : float | np.ndarray
        Sample density.
    effective_mass : float
        Effective mass of the carriers.

    Returns
    -------
    soi : float | np.ndarray
        Spin-orbit coupling in meV.A

    """
    t_free = mean_free_time_from_mobility(mobility, effective_mass)
    diff = diffusion_constant_from_mobility_density(mobility, density,
                                                    effective_mass)
    return np.sqrt(soi_field*cs.e*diff*cs.hbar /
                   (np.pi*density*t_free) )*1e13/cs.e


def flip_field_axis(field, *quantities):
    """Ensure that the field is always in increasing order.

    """
    if len(field.shape) == 1:
        if field[0] > field[1]:
            field = field[::-1]
            for q in quantities:
                q[:] = q[::-1]
        return

    for i, f in enumerate(field):
        if f[0] > f[1]:
            field[i] = f[::-1]
            for q in quantities:
                q[i] = q[i][::-1]


def recenter_wal_data(field, resistance, fraction=0.2, minrel_order=10):
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
        Resistance (xx or yy) values in Ω which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    fraction : float, optional
        Fraction of the total field excursion in which to look for the WAL
        resistance minimum.
    minrel_order : int, optional
        Number or points to consider when looking for a minimum value.

    Returns
    -------
    field : np.ndarray
        Magnetic fields such that 0 field corresponds to the WAL minimum.
    resistance : np.ndarray
        Resistance values in Ω which were measured.

    """
    if len(field.shape) >= 2:
        original_shape = field.shape
        trace_number = np.prod(original_shape[:-1])
        field = field.reshape((trace_number, -1))
        resistance = resistance.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        resistance = np.array((resistance,))

    for i, (f, r) in enumerate(zip(field, resistance)):
        max_field = np.max(f)*fraction
        mask = np.where(np.less(np.abs(f), max_field))
        masked_r = r[mask]
        mins, = argrelmin(masked_r, order=minrel_order)  # We need to unpack
        # No local minima were found, look for a maximum
        if not len(mins):
            center_index = np.argmax(masked_r)
        else:
            if len(mins) == 1:
                center_index = mins[0]
            else:
                center_index = mins[np.argmin([masked_r[i] for i in mins])]

        offset_field = f[mask][center_index]
        field[i] -= offset_field

    if trace_number == 1:
        return field[0], resistance[0]
    return field.reshape(original_shape), resistance.reshape(original_shape)


def symmetrize_wal_data(field, resistance, mode='average'):
    """Symmetrize WAL data with respect to 0 field.

    This method assumes that the data have been properly recentered around 0
    and that the field is always increasing.
    Points that cannot be symmetrized are left untouched.

    Parameters
    ----------
    field : np.ndarray
        Magnetic fields at which the sample resistance was measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    resistance : np.ndarray
        Resistance (xx or yy) values in Ω which were measured.
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
        trace_number = np.prod(original_shape[:-1])
        field = field.reshape((trace_number, -1))
        resistance = resistance.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        resistance = np.array((resistance,))

    for i, (f, r) in enumerate(zip(field, resistance)):
        center = np.argmin(np.abs(f))
        neg_len = len(r[:center])
        pos_len = len(r[center+1:])
        if pos_len == neg_len:
            if mode == 'average':
                resistance[i] = (r + r[::-1])/2
            elif mode == 'positive':
                resistance[i][:center] = r[center+1:][::-1]
            else:
                resistance[i][center+1:] = r[:center][::-1]
        elif pos_len > neg_len:
            neg_r = r[:center].copy()[::-1]
            pos_r = r[center+1:center+1+neg_len].copy()[::-1]
            if mode == 'average':
                resistance[i][:center] += pos_r
                resistance[i][:center] /= 2
                resistance[i][center+1:center+1+neg_len] += neg_r
                resistance[i][center+1:center+1+neg_len] /= 2
            elif mode == 'positive':
                resistance[i][:center] = pos_r
            else:
                resistance[i][center+1:center+1+neg_len] = neg_r
        else:
            neg_r = r[abs(pos_len-center):center].copy()[::-1]
            pos_r = r[center+1:].copy()[::-1]
            if mode == 'average':
                resistance[i][abs(pos_len-center):center] += pos_r
                resistance[i][abs(pos_len-center):center] /= 2
                resistance[i][center+1:] += neg_r
                resistance[i][center+1:] /= 2
            elif mode == 'positive':
                resistance[i][abs(pos_len-center):center] = pos_r
            else:
                resistance[i][center+1:] = neg_r

    if trace_number == 1:
        return field[0], resistance[0]
    return field.reshape(original_shape), resistance.reshape(original_shape)


def weight_wal_data(field, dsigma, mask='gauss', stiffness=1, htr=None):
    """Generate weigth to use when fitting WAL data.

    First we identify the minimum in the conductance. If it not localized at
    zero field we use this value to determine on what scale to have the weight
    decays.

    Parameters
    ----------
    field : [type]
        [description]
    dsigma : [type]
        [description]
    stiffness : [type]
        [description]

    """
    filtered_dsigma = savgol_filter(dsigma, 31, 3)
    index = np.argmin(filtered_dsigma)
    wf = max(field[index], 2*htr) if htr is not None else field[index]
    if mask == 'exp':
        return np.exp(-np.abs(field/wf)*stiffness)
    elif mask == 'peak-gauss':
        return (1*np.exp(-np.abs(field/wf)**2*stiffness) +
                2*np.exp(-np.abs(field/wf*2)**2*stiffness))
    elif mask == 'gauss':
        return np.exp(-np.abs(field/wf)**2*stiffness)
    elif mask == 'lorentz':
        return 1/(1 + np.abs(field/wf)**2*stiffness)

