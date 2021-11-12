# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Functions used to analyse squid oscillations.

"""
from warnings import warn

import numpy as np


def extract_switching_current(field, bias, diff, threshold):
    """Extract the switching current from a B vs bias map of resistance.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field at which the data have been acquired.
        Expected to be a 2D array with constant columns.
    bias : np.ndarray
        Bias current applied to the junction.
        Expected to be a 2D array with constant rows.
    diff : np.ndarray
        Differential resistance of the junction.
    threshold : float
        Threshold value used to determine the critical current.

    """
    warn(
        "shabanipy.squid.squid_analaysis.extract_switching_current is deprecated. "
        "Use shabanipy.dvdi.extract_switching_current instead."
    )
    temp = np.greater(diff, threshold)
    index = np.argmax(temp, axis=-1).reshape((-1, 1))
    return field[:, 0], np.ravel(bias.take(index))
