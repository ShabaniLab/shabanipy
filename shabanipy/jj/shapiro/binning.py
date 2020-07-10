# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
""" Routines to bin Shapiro steps measurements.

"""
from typing import Optional, Tuple, Union
from math import ceil, copysign, floor

import numpy as np

from . import shapiro_step


def center_bin(bins: np.ndarray) -> np.ndarray:
    """Shift a bin by half a step and remove the last item.

    This transform the bins produced by np.histogram in array representing the
    center of each bin. Both input and output arrays are 1D.

    """
    return 0.5 * (bins[1] - bins[0]) + bins[:-1]


def bin_shapiro_steps(
    voltage: np.ndarray,
    frequency: Optional[float] = None,
    step_fraction: float = 0.1,
    bins: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the histogram associated with an IV curve.

    Parameters
    ----------
    voltage : np.ndarray
        Measured voltages, 1D array.

    frequency : Optional[float], optional
        Frequency at which the experiment was carried out in Hz.
        Required if bins is None.

    step_fraction : float, optional
        Fraction of a shapiro step to use when binning the data.

    bins : np.ndarray, optional
        Precomputed bins to use to compute the histogram.

    Returns
    -------
    histogram : np.ndarray
        Histogram of the counts in each bins.

    bins : np.ndarray
        Edges of the bins used to build the histogram.
        Note : len(bins) == len(histogram) + 1
        Use `center_bin` to obtain an array representing the center of the bins.

    """
    # Determine the bins to use if necessary.
    if bins is None:

        # Compute the shapiro step voltage
        if frequency is None:
            raise ValueError(
                "Neither bins nor frequency was specified one at least is required."
            )
        step = shapiro_step(frequency) * step_fraction

        min_v, max_v = np.min(voltage), np.max(voltage)
        neg_bins, pos_bins = abs(floor(min_v / step)), abs(ceil(max_v / step))
        total_bins = neg_bins + pos_bins
        if (total_bins) % 2 == 0:
            bins = np.linspace(
                neg_bins * copysign(step, min_v),
                pos_bins * copysign(step, max_v),
                total_bins,
            )
        else:
            bins = np.linspace(
                (neg_bins + 0.5) * copysign(step, min_v),
                (pos_bins + 0.5) * copysign(step, max_v),
                total_bins,
            )

    return np.histogram(voltage, bins)


def bin_power_shapiro_steps(
    power: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    frequency: Union[float, np.ndarray],
    step_fraction: float = 0.1,
    debug: bool = False,
):
    """Bin a power vs current measurement of the junction voltage.

    power, current and voltage are expected to be of the same shape.

    Parameters
    ----------
    power : np.ndarray
        Power used in the measurement. 2D array.

    current : np.ndarray
        Bias current applied to the junction. 2D array.

    voltage : np.ndarray
        Measured voltages. 2D array.

    frequency : float, optional
        Frequency at which the experiment was carried out in Hz.

    step_fraction : float, optional
        Fraction of a shapiro step to use when binning the data.

    bins : np.ndarray, optional
        Precomputed bins to use to compute the histogram.

    debug : bool, optional
        Provide debug information.

    Returns
    -------
    voltage : np.ndarray
        Linear array of the center of the voltage bins, normalized by the
        Shapiro step size.

    histo : np.ndarray
        2D array of the voltage counts (expressed in current, ie multiplied by
        the current step). Each line corresponds to a constant power and
        varying bias current.

    """
    # Build the power array and compute the current step
    power, indices = np.unique(power, return_index=True)
    current = np.unique(current)
    c_step = abs(current[1] - current[0])

    _, bins = bin_shapiro_steps(voltage[0], frequency, step_fraction)
    results = np.empty((len(power), len(bins) - 1))
    for i in range(len(power)):
        results[i], _ = bin_shapiro_steps(voltage[i], bins=bins)
    results *= c_step

    # Flip the power axis if the power was decreasing during the scan.
    if indices[0] > indices[1]:
        results = results[::-1]

    return power, center_bin(bins) / shapiro_step(frequency), results


def extract_step_weight(voltage, histo, index):
    """Extract the line of an histogram matching a given step.

    Parameters
    ----------
    voltage : np.ndarray
        1D array of the voltage matching histo. Voltages are expected to be
        normalized to the height of a Shapiro step.

    histo : np.ndarray
        2D histogram of the counts as a function of power (first index), and
        voltage (second index).

    index : int
        Index of the Shapiro step for which data should be retrieved.

    Returns
    -------
    counts : np.ndarray
        1D array of the counts as a function of power for the specified step.

    """
    step_index = np.argmin(np.abs(voltage - index))
    if abs(voltage[step_index] - index) > 0.5:
        raise ValueError(f"Step {index} does not exist in the data.")
    return histo[:, step_index]
