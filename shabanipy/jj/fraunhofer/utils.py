# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to analyse data taken on JJ.

"""
import logging
from typing import Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import peak_widths
from lmfit.models import GaussianModel


def find_fraunhofer_center(
    field: np.ndarray, ic: np.ndarray, debug: bool = False
) -> float:
    """Extract the field at which the Fraunhofer is centered.

    Parameters
    ----------
    field : np.ndarray
        1D array of the magnetic field applied of the JJ.
    ic : np.ndarray
        1D array of the JJ critical current.

    Returns
    -------
    float
        Field at which the center of the pattern is located.

    """
    max_loc = np.argmax(ic)
    width, *_ = peak_widths(ic, [max_loc], rel_height=0.5)
    width_index = int(round(width[0] * 0.65))
    subset_field = field[max_loc - width_index : max_loc + width_index + 1]
    subset_ic = ic[max_loc - width_index : max_loc + width_index + 1]
    model = GaussianModel()
    params = model.guess(subset_ic, subset_field)
    out = model.fit(subset_ic, params, x=subset_field)

    if False:
        plt.figure()
        plt.plot(field, ic)
        plt.plot(subset_field, out.best_fit)
        plt.show()

    return out.best_values["center"]


def recenter_fraunhofer(
    field: np.ndarray, ic: np.ndarray, debug: bool = False
) -> np.ndarray:
    """Correct the offset in field of a Fraunhofer pattern.

    Parameters
    ----------
    field : np.ndarray
        ND array of the magnetic field applied of the JJ, the last dimension is
        expected to be swept.
    ic : np.ndarray
        ND array of the JJ critical current.

    Returns
    -------
    np.ndarray
        Field array from which the offset has been removed.

    """
    it = np.nditer(field[..., 0], ["multi_index"])
    res = np.copy(field)
    for b in it:
        index = it.multi_index
        center = find_fraunhofer_center(field[index], ic[index], debug)
        res[index] -= center

    return res
