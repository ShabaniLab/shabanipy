# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines used in the analysis of shapiro steps experiments.

"""
import numpy as np
from scipy.constants import h, e


def shapiro_step(frequency):
    """ Compute the amplitude of a Shapiro step at a given frequency.

    """
    return h * frequency / (2 * e)


def normalize_db_power(power, norm_power):
    """Normalize a power in dB by a power in dB.

    Because the quantities are expressed in dB, we need to first convert to
    linearize power.

    """
    lin_power = np.power(10, power / 10)
    lin_norm_power = np.power(10, norm_power / 10)
    return 10 * np.log10(lin_power / lin_norm_power)
