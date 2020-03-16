# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Model functions used to describe SQUID oscillations.

"""
from math import pi

import numpy as np
from scipy.constants import h, e
from scipy.interpolate import interp1d


def compute_squid_current(
    phase,
    cpr1,
    parameters1,
    cpr2,
    parameters2,
    positive=True,
    aux_res=101,
    inductance=0.0,
    compute_phase=False,
):
    """Compute the SQUID current from 2 CPRs.

    Parameters
    ----------
    phase : np.ndarray
        Phase at which to compute the SQUID current flow (1D array at most).
    cpr1 : callable
        Function used to compute the current in the first junction. The
        callable should take the phase as first argument.
    parameters1 : tuple
        Parameters to use to compute the current in the first junction.
    cpr2 : callable
        Function used to compute the current in the second junction. The
        callable should take the phase as first argument.
    parameters1 : tuple
        Parameters to use to compute the current in the second junction.
    positive : bool, optional
        Should the computed current be positive or negative
    aux_res : int, optional
        Number of points to use when optimizing the phase to get the maximum
        current.
    inductance : float, optional
        Inductance of the loop in H
    compute_phase : bool, optional
        Computethe phase at which the squid current is maximal instead of the
        current.

    """
    phi1, *p1 = parameters1
    phi2, *p2 = parameters2
    if inductance == 0.0:
        aux = np.tile(np.linspace(0, 2 * np.pi, aux_res), (len(phase), 1))
        phi1 += (aux.T + phase).T
        phi2 += aux
        cp1 = cpr1(phi1, *p1)
        cp2 = cpr2(phi2, *p2)
        total_current = cp1 + cp2
        if positive:
            if compute_phase:
                index = np.argmax(total_current, axis=-1)
                return aux[0, index]
            return np.max(total_current, axis=-1)
        else:
            if compute_phase:
                index = np.argmin(total_current, axis=-1)
                return aux[0, index]
            return np.min(total_current, axis=-1)

    else:
        # In the presence of an inductance compute first the phase giving the
        # larger current as a function of the phase difference between the two
        # JJ.
        delta_phi = np.linspace(2 * phase[0], 2 * phase[-1], 2 * len(phase) * 10)
        optimal_aux = compute_squid_current(
            delta_phi,
            cpr1,
            parameters1,
            cpr2,
            parameters2,
            positive,
            aux_res,
            0.0,
            True,
        )

        # Next compute the external phase that gave rise to the phase
        # difference
        phi1 += delta_phi + optimal_aux
        phi2 += optimal_aux
        cp1 = cpr1(phi1, *p1)
        cp2 = cpr2(phi2, *p2)
        total_current = cp1 + cp2
        phi_ext = delta_phi + 2 * pi * e / h * inductance * (cp2 - cp1)
        # Interpolate the current as a function of the external phase and
        # compute the current at the requested points
        return interp1d(phi_ext, total_current, kind="cubic", copy=False)(phase)
