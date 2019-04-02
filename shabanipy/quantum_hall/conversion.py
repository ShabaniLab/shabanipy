# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Typical Hall bar data conversion routines.

"""
from math import pi, log

import numpy as np
import scipy.constants as cs

GEOMETRIC_FACTORS = {
    'Van der Pauw': pi/log(2),
    'Standard Hall bar': 0.75,
}


def convert_lock_in_meas_to_diff_res(measured_voltage, bias_current):
    """Convert the voltage measured using a lock-in to differential resistance.

    """
    return measured_voltage/bias_current


def kf_from_density(density):
    """Compute the Fermi wavevector from the density.

    Parameters
    ----------
    density : float | np.ndarray
        Carriers density of the sample expected to be in cm^-2

    Returns
    -------
    kf : float | np.ndarray
        Fermi wavevector in m^-1.

    """
    return np.sqrt(2*np.pi*density) * 1e-2  # Conversion to m^-1


def mean_free_time_from_mobility(mobility, effective_mass):
    """Compute the mean free time from the sample mobility

    Parameters
    ----------
    mobility : float | np.ndarray
        Carriers mobility of the sample.
    effective_mass : float
        Effective mass of the carriers in kg.

    Returns
    -------
    mean_free_time : float | np.ndarray
        Fermi wavevector in s.

    """
    return mobility*effective_mass/cs.e


def fermi_velocity_from_kf(kf, effective_mass):
    """Compute the Fermi velocity from the Fermi wavelength

    Parameters
    ----------
    kf : float | np.ndarray
        Fermi wavevector in m^-1.

    Returns
    -------
    fermi_vel : float | np.ndarray
        Fermi velocity in m.s^-1.

    """
    return cs.hbar*kf/effective_mass


def fermi_velocity_from_density(density, effective_mass):
    """Compute the Fermi velocity directly from the density.

    Parameters
    ----------
    density :  : float | np.ndarray
        Carriers density of the sample expected to be in cm^-2

    Returns
    -------
    fermi_vel : float | np.ndarray
        Fermi velocity in m.s^-1.

    """
    return fermi_velocity_from_kf(kf_from_density(density), effective_mass)


def diffusion_constant_from_mobility_density(mobility, density,
                                             effective_mass):
    """Compute the diffusion constant from mobility and density.

    Parameters
    ----------
    mobility : float | np.ndarray
        Carriers mobility of the sample.
    density :  : float | np.ndarray
        Carriers density of the sample expected to be in cm^-2

    Returns
    -------
    diffusion_constant : float | np.ndarray
        Diffusion constant of the carriers.

    """
    vf = fermi_velocity_from_density(density, effective_mass)
    mft = mean_free_time_from_mobility(mobility, effective_mass)
    return vf**2*mft/2


def htr_from_mobility_density(mobility, density, effective_mass):
    """[summary]

    Parameters
    ----------
    mobilities : [type]
        [description]
    densities : [type]
        [description]

    Returns
    -------

    """
    d = diffusion_constant_from_mobility_density(mobility, density,
                                                 effective_mass)
    mft = mean_free_time_from_mobility(mobility, effective_mass)
    return cs.hbar/(4*cs.e*d*mft)
