# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Current phase relationship for usual Josephson junction.

All functions in this module should take the the phase as first argument.

"""
import numpy as np
import scipy.constants as cs


def josephson_cpr(phase, critical_current):
    """Compute the current flowing through a junction of zero transparency.

    This is the 1st Josephson relation and is equivalent to
    `transparent_cpr` with transparency=0.

    Parameters
    ----------
    phase : float | np.ndarray
        Phase across the junction.
    critical_current : float
        Critical current of the junction.

    """
    return critical_current*np.sin(phase)


def _finite_helper(phase, transparency, temperature, gap):
    """Helper function to compute current in a finite transparency jj.

    """
    sq = np.sqrt(1-transparency*np.sin(phase/2)**2)
    cpr = np.sin(phase)/sq
    if temperature != 0:
        cpr *= np.tanh(gap/(2*cs.Boltzmann*temperature) * sq)
    return  cpr


def transparent_cpr(phase, critical_current, transparency, temperature=0, gap=1):
    """Compute the current flowing through a junction of finite transparency.

    Parameters
    ----------
    phase : float | np.ndarray
        Phase across the junction.
    critical_current : float
        Critical current of the junction.
    transparency : float
        Transparency of the junction.
    temperature : float
        Temperature of the junction in K.
    gap : float
        Gap of the superconductor in J.

    """
    aux = np.linspace(0, 2*np.pi, 101)
    norm = np.max(_finite_helper(aux, transparency, temperature, gap))
    return critical_current / norm * _finite_helper(phase, transparency,
                                                    temperature, gap)


def fraunhofer_envelope(phase):
    """Fraunhofer envelope.

    """
    return np.sinc(phase)
