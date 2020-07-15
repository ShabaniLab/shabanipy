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

import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing_extensions import Literal

LOGGER = logging.getLogger(__name__)


def compute_voltage_offset(
    current_bias: np.ndarray, measured_voltage: np.ndarray, n_peak_width: int
) -> Tuple[float, float]:
    """Compute the voltage offset in the VI characteristic of a JJ.

    The algorithm assumes that the JJ presents a finite zero voltage region around
    zero bias. This region is determined by determining the positions of the peaks
    in the derivative.

    Parameters
    ----------
    current_bias : np.ndarray
        Current bias applied on the junction in A (1D array).
    measured_voltage : np.ndarray
        Voltage accross the junction in V (1D array).
    n_peak_width : int
        Number of peak width to substract from the region between the peaks.

    Returns
    -------
    avg: float
        Mean value of the voltage around zero bias.
    std : float
        Standard deviation of the voltage around zero bias.

    """
    # Determine the index of the zero current bias
    midpoint = np.argmin(abs(current_bias))

    # Compute the derivative of the signal
    dydx = np.diff(measured_voltage)
    l_dydx = dydx[:midpoint]
    r_dydx = dydx[midpoint:]

    # Find the most prominent peaks on each side of the zero bias current
    peaks_left, _ = find_peaks(l_dydx, max(dydx[:midpoint]) / 2)
    peaks_right, _ = find_peaks(r_dydx, max(dydx[midpoint:]) / 2)

    # Manually evaluate the width of the peaks since they can be very asymetric
    lw = 0
    l_peak_value = l_dydx[peaks_left[-1]]
    while l_dydx[peaks_left[-1] + lw] > l_peak_value / 2:
        lw += 1

    rw = 0
    r_peak_value = r_dydx[peaks_right[-1]]
    while r_dydx[peaks_right[-1] - rw] > r_peak_value / 2:
        rw += 1

    # Keep only the data between the two peaks
    area = measured_voltage[
        peaks_left[-1]
        + lw * n_peak_width : peaks_right[0]
        + midpoint
        - rw * n_peak_width
    ]

    # If we get an empty array simply return the middle point.
    if area.shape == (0,):
        LOGGER.info(
            "While computing voltage offset, the area between resistance peak was "
            "found to be 0."
        )
        return (
            measured_voltage[midpoint],
            np.std(measured_voltage[midpoint - 1 : midpoint + 2]),
        )
    else:
        return np.average(area), np.std(area)


def correct_voltage_offset(
    current_bias: np.ndarray,
    measured_voltage: np.ndarray,
    n_peak_width: int,
    index: Optional[Union[int, Tuple[int, ...]]] = None,
    debug: bool = False,
) -> np.ndarray:
    """Correct the voltage offset of VI curves.

    The algorithm uses the voltage of the supereconducting region around zero bias.

    Parameters
    ----------
    current_bias : np.ndarray
        N+1D array containing the current bias.
    measured_voltage : np.ndarray
        N+1D array containing the measured voltage accross the JJ.
    n_peak_width : int
        To identify the zero voltage region the algorithm computes the derivative
        and look for peaks signaling the transition out of the superconducting
        domain. This parameter specifies how many width of the peaks to ignore
        when determining the actual zero voltage region.
    index : Optional[Union[int, Tuple[int, ...]]
        Index to select only a single trace to use to determine the offset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [description]
    """
    # Copy the data to preserve the original
    mv = np.copy(measured_voltage)

    # Average only a single trace to compute the offset
    if index is not None:
        if isinstance(index, int):
            cb = current_bias[index]
            sv = measured_voltage[index]
        else:
            cb = current_bias
            sv = measured_voltage
            for i in index:
                cb = cb[i]
                sv = mv[i]

        avg, _ = compute_voltage_offset(cb, sv, n_peak_width)
        mv -= avg

    # Substract the offset for each line
    else:
        it = np.nditer(current_bias[..., 0], ["multi_index"])
        for b in it:
            mv[it.multi_index] -= compute_voltage_offset(
                current_bias[it.multi_index],
                measured_voltage[it.multi_index],
                n_peak_width,
            )[0]

    return mv


def compute_resistance(
    bias_current: np.ndarray, measured_voltage: np.ndarray, debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the differential from the VI characteristic of a JJ.

    """
    resistance = np.gradient(measured_voltage, bias_current[0], axis=-1)
    return bias_current, resistance


def extract_switching_current(
    bias: np.ndarray,
    volt_or_res: np.ndarray,
    threshold: float,
    side: Literal["positive", "negative"],
    debug: bool = False,
) -> np.ndarray:
    """Extract the switching current from a voltage or resistance map.

    If more than 1D array input are used, the last dimension if assumed to be swept.
    The current sweep does not have to be the same for all outer dimensions.

    Parameters
    ----------
    bias : np.ndarray
        N+1D array of the bias current applied to the junction.
    volt_or_res : np.ndarray
        N+1D of the voltage or differential resistance of the junction.
    threshold : float
        Threshold value used to determine the critical current.
    side : {"positive", "negative"}, optional
        On which branch of the bias current to extract the critical current,
        by default "positive"
    debug : bool, optional
        Should additional debug information be provided.

    Returns
    -------
    np.ndarray
        ND array of the extracted critical current.

    """
    if side not in ("positive", "negative"):
        raise ValueError(f"Side should be 'positive' or 'negative', found {side}.")

    # Index at which the bias current is zero
    mask = np.greater_equal(bias, 0) if side == "positive" else np.less_equal(bias, 0)
    if not mask.any():
        raise ValueError(f"No {side} bias data in the set.")

    # Mask the data to get only the data we care about
    masked_bias = bias[mask].reshape(bias.shape[:-1] + (-1,))
    masked_data = volt_or_res[mask].reshape(bias.shape[:-1] + (-1,))
    it = np.nditer(masked_bias[..., 0], ["multi_index"])
    for b in it:
        if np.argmin(np.abs(masked_bias[it.multi_index])) != 0:
            masked_bias[it.multi_index + (slice(None, None),)] = masked_bias[
                it.multi_index + (slice(None, None, -1),)
            ]
            masked_data[it.multi_index + (slice(None, None),)] = masked_data[
                it.multi_index + (slice(None, None, -1),)
            ]

    temp = np.greater(np.abs(masked_data), threshold)
    # Make sure we pinpoint the last current where we were below threshold
    index = np.argmax(temp, axis=-1) - 1

    return np.take_along_axis(masked_bias, index[..., None], axis=-1).reshape(
        bias.shape[:-1]
    )
