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

GEOMETRIC_FACTORS = {
    'Van der Pauw': pi/log(2),
    'Standard Hall bar': 0.75,
}


def convert_lock_in_meas_to_diff_res(measured_voltage, bias_current):
    """Convert the voltage measured using a lock-in to differential resistance.

    """
    return measured_voltage/bias_current

def kf_from_density(densities):
    """[summary]

    Parameters
    ----------
    densities : [type]
        [description]

    Returns
    -------

    """
    pass


def mean_free_time_from_mobility(mobilities):
    """[summary]

    Parameters
    ----------
    mobilities : [type]
        [description]

    Returns
    -------

    """
    pass


def fermi_velocity_from_kf(kfs):
    """[summary]

    Parameters
    ----------
    kfs : [type]
        [description]

    """
    pass


def fermi_velocity_from_density(densities):
    """[summary]

    Parameters
    ----------
    densities : [type]
        [description]

    """
    pass


def diffusion_constant_from_mobility_density(mobilities, densities):
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
    pass


def htr_from_mobility_density(mobilities, densities):
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

