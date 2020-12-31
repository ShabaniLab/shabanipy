# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines used to study Shapiro steps.

"""
from typing import Tuple, Optional, Union

import numpy as np
from scipy.signal import find_peaks

from shabanipy.jj.utils import compute_voltage_offset
from . import shapiro_step
from .binning import bin_power_shapiro_steps


def correct_voltage_offset_per_power(
    power: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    frequency: Union[float, np.ndarray],
    n_peak_width: int,
    n_std_as_bin: int,
    bound: Optional[float] = None,
    debug: bool = False,
):
    """Correct the voltage offset in a Shapiro map (power, current bias) at each power.

    The correction assumes that the offset is smaller than half a Shapiro step and
    works by realigning peaks to quantified values. It may fail in the presence of
    strong fractional peaks.

    Arrays are assumed to be in such a shape that the bias varies on the last dimension
    and the power on the penultimate dimension.

    Parameters
    ----------
    power : np.ndarray
        N+2D array containing the microwave power used in the Shapiro.
    current : np.ndarray
        N+2D array containing the bias current.
    voltage : np.ndarray
        N+2D array containing the measured voltage.
    frequency : Union[float, np.ndarray]
        float or ND array containing the frequency at which the experiment was carried
        out.
    n_peak_width : int
        Number of peak width to remove when determining the superconducting region in a
        VI curve see shabanipy.jj.utils.compute_voltage_offset
    n_std_as_bin : int
        Number of standard deviation (as determined from the superconducting plateau of
        the lowest power measurement).
    bound : Optional[float]
        Bounds around midpoint to look for peaks (in uA)
    debug : bool, optional
        [description], by default False

    Returns
    -------
    np.ndarray
        N+2D array containing the corrected voltage.

    """
    # Copy the data to preserve the original
    new_voltage = np.copy(voltage)
    
    # Iterate on the extra dimensions if any
    it = np.nditer(power[..., 0, 0], ["multi_index"])

    for b in it:
        index = it.multi_index

        # Compute the value of the Shapiro step
        step = shapiro_step(
            frequency if isinstance(frequency, float) else frequency[index]
        )

        # Those arrays are guaranteed to be 2D
        p = power[index]
        c = current[index]
        v = new_voltage[index]

        # Determine the noise on the data by looking at the zero resistance state
        # of the lowest measurement power
        lpower_index = np.argmin(p[:, 0])
        _, std = compute_voltage_offset(
            c[lpower_index, :], v[lpower_index, :], n_peak_width, bound
        )

        # Compute the step fraction to use when binning to get a high resolution
        # histogram
        step_fraction = n_std_as_bin * std / step

        # Compute the histogram of the steps and get the voltage in unit of shapiro steps
        # As a consequence steps are an interger value
        volt_1d, histo = bin_power_shapiro_steps(p, c, v, frequency, step_fraction)

        # Iterate over the line of the histo and find the peaks (ie Shapiro steps)
        for j, h in enumerate(histo):

            # Enforce that the peaks are at least of about 1 (ignore fractional steps)
            # In some cases, height here may cause an issue (not large enough or too large)
            peaks, _ = find_peaks(h, distance=0.95 / step_fraction, height=max(h)/2)

            # Calculate deviation of each peak and average
            dev = np.average([volt_1d[i] - round(volt_1d[i]) for i in peaks])

            # Subctract the offset of each line
            v[j] -= dev * step

    return new_voltage