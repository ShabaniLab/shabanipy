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
import numpy as np


def compute_squid_current(phase, cpr1, parameters1, cpr2, parameters2,
                          positive=True, aux_res=101):
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

    """
    aux = np.tile(np.linspace(0, 2*np.pi, aux_res), (len(phase), 1))
    phi1, *p1 = parameters1
    phi1 += (aux.T + phase).T
    phi2, *p2 = parameters2
    phi2 += aux
    cp1 = cpr1(phi1, *p1)
    cp2 = cpr2(phi2, *p2)
    total_current = cp1 + cp2
    if positive:
        return np.max(total_current, axis=-1)
    else:
        return np.min(total_current, axis=-1)
