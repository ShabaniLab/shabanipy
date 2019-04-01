# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak antilocalisation analysis routines.

The routines is this module assume that no linear Dresselhaus is present.

"""
import numpy as np
from scipy.special import digamma


def compute_wal_conductance_difference(field, dephasing_field, linear_soi,
                                       cubic_soi, low_field_reference,
                                       series_truncation=5000):
    """Variation in the conductance induced by SOI.

    The calculation is extracted from:
    W. Knap et al., Weak Antilocalization and Spin Precession in Quantum Wells.
    Physical Review B. 53, 3912–3924 (1996).

    We use Formula 37,38 in which we consider no linear Dresselhaus term.

    Parameters
    ----------
    field : float | np.array
        Magnetic field at which to compute the conductance.
    dephasing_field : float
        Dephasing field used in the model. The unit should the same as the one
        used to express the magnetic field.
    linear_soi : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling. The unit should the same as the one used to express the
        magnetic field.
    cubic_soi : float
        Field describing the contribution of the cubic term arising from
        Dresselhaus SOI. The unit should the same as the one used to express
        the magnetic field.
    low_field_reference : float
        Reference field at which to compute the conductance difference.
    series_truncation : int, optional
        Last term of the series involved in the conductance expression to
        compute (the default is 5000)

    Returns
    -------
    delta_sigma : float | np.ndarray
        Difference between the conductance at the reference field and the
        conductance at the specified field. The conductance is expressed in
        term of the quantum of conductance.

    """
    # Field dependant fitting parameters
    dephasing_r = np.abs(dephasing_field/B)  # H_phi / B
    linear_soi_r = np.abs(linear_soi/B)  # H_SO^2/B assuming Dresselhaus = 0
    cubic_soi_r  = np.abs(cubic_soi/B)  # Cubic term in H_S0 proportional to τ3

    # Lower field cutoff for calculation
    dephasing_ref = abs(dephasing_field / low_field_reference)
    linear_soi_ref = abs(linear_soi / low_field_reference)
    cubic_soi_ref  = abs(cubic_soi / low_field_reference)

    sigma_b   = compute_wal_conductance(dephasing_r, linear_soi_r, cubic_soi_r,
                                        series_truncation)
    sigma_ref = compute_wal_conductance(dephasing_ref, linear_soi_ref,
                                        cubic_soi_ref, series_truncation)

    return sigma_b - sigma_ref


def compute_wal_conductance(dephasing_ratio, linear_soi_ratio, cubic_soi_ratio,
                            series_truncation: int):
    """Compute the conductance in the presence of SOI.

    This formula is meant to be used in the computation of variation of
    conductance and as a consequence omit some constants and use a fixed value
    for H_tr since it simplifies in differences.

    Parameters
    ----------
    dephasing_field_ratio : float
        Dephasing field over the applied field.
    linear_soi : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling over the applied field.
    cubic_soi : float
        Field describing the contribution of the cubic term arising from
        Dresselhaus SOI over the applied field.
    series_truncation : int, optional
        Last term of the series involved in the conductance expression to
        compute (the default is 5000)

    Returns
    -------
    sigma : float | np.ndarray
        Conductance expressed in term of the quantum of conductance.

    """
    soi = linear_soi_ratio + cubic_soi_ratio
    a0  = dephasing_field_ratio + soi + 0.5
    s   = truncate_wal_series(series_truncation, dephasing_field_ratio,
                              linear_soi_ratio, cubic_soi_ratio)
    return -(1/a0 +
             (2*a0 + 1 + soi)/((1 + a0)*(a0 + soi) - 2*linear_soi_ratio) -
             s + 2*np.log(1/B) + digamma(1/2 + dephasing))


def truncate_wal_series(series_truncation, dephasing_field_ratio,
                        linear_soi_ratio, cubic_soi_ratio):
    """Compute the truncate series used in wal conductance calculation.

    Parameters
    ----------
    dephasing_field_ratio : float
        Dephasing field over the applied field.
    linear_soi : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling over the applied field.
    cubic_soi : float
        Field describing the contribution of the cubic term arising from
        Dresselhaus SOI over the applied field.
    series_truncation : int, optional
        Last term of the series involved in the conductance expression to
        compute (the default is 5000)

    Returns
    -------
    truncated_series : float | np.ndarray
        Truncated series.

    """
    try:
        size = len(dephasing_field_ratio)
    except TypeError:
        size = 1
    index = np.array([range(1, series_truncation + 1)] * size)
    soi = linear_soi_ratio + cubic_soi_ratio
    a0 = dephasing_field_ratio + soi + 0.5
    an = index + a0

    s = np.sum(3/index -
               (3*an**2 + 2*an*soi - 1 - 2*linear_soi_ratio.*(2*index + 1)) /
               ((an + soi)*(an - 1)*(an + 1)
               - 2*linear_soi_ratio.*((2*index + 1)*an - 1)), axis=-1)

    return s