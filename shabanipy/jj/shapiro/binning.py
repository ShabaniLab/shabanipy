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
from typing import Optional, Tuple, Union, List
from math import ceil, copysign, floor

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter

from . import shapiro_step


def center_bin(bins: np.ndarray) -> np.ndarray:
    """Shift a bin by half a step and remove the last item.

    This transform the bins produced by np.histogram in array representing the
    center of each bin. Both input and output arrays are 1D.

    """
    return 0.5 * (bins[1] - bins[0]) + bins[:-1]


def create_weigths(current: np.ndarray) -> np.ndarray:
    """Compute the proper weigths for a Shapiro histogram non-equistant current points.

    """
    weights = np.zeros_like(current)
    weights[0] = abs(current[1] - current[0]) / 2
    weights[-1] = abs(current[-1] - current[-2]) / 2
    weights[1:-1] = np.abs(current[:-2] - current[2:]) / 2
    return weights


def bin_shapiro_steps(
    voltage: np.ndarray,
    frequency: Optional[float] = None,
    step_fraction: float = 0.1,
    bins: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
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

    weights : np.ndarray, optional
        Weights to use for the histogram in case points are not linearly spaced.

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

    return np.histogram(voltage, bins, weights=weights)


def bin_power_shapiro_steps(
    power: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    frequency: Union[float, np.ndarray],
    step_fraction: float = 0.1,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a power vs current measurement of the junction voltage.

    Power, current and voltage are expected to be of the same shape and the
    current sweep is expected to occur on the last axis.

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
    # Extract the power and the current scans and compute the current step.
    power = power[:, 0]
    current = current[0]

    # Generate the bins from the highest power which we expect to display the
    # highest voltages
    index = -1 if power[1] > power[0] else 0
    incr = 1 if index == 0 else -1
    v = voltage[index]
    while np.any(np.isnan(v)):
        index += incr
        v = voltage[index]
    _, bins = bin_shapiro_steps(v, frequency, step_fraction)

    # Bin measurements at all power using the same bins.
    results = np.empty((voltage.shape[0], len(bins) - 1))
    for i in range(voltage.shape[0]):
        results[i], _ = bin_shapiro_steps(
            voltage[i], bins=bins, weights=create_weigths(current)
        )

    aux = center_bin(bins) / shapiro_step(frequency)

    return center_bin(bins) / shapiro_step(frequency), results


def extract_step_weight(
    voltage: np.ndarray, histo: np.ndarray, index: int
) -> np.ndarray:
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

def extract_shapiro_qfactor(
    power: np.ndarray,
    voltage: np.ndarray,
    counts: np.ndarray,
    steps: Optional[List[int]] = [0,1,2],
    steps_combo: Optional[List[int]] = [[1,2]],
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    gaussian_sigma: Optional[float] = 1.5,
    power_offset: Optional[float] = None,
    power_limits: Optional[np.ndarray] = None,
    debug: bool = False,
) -> np.ndarray:
    """Extract shapiro Q-factor
    Parameters
    ----------
    voltage : np.ndarray
        1D array of the voltage bins.
    counts : np.ndarray
        2D array of the histogram current counts.
    steps : List[int]
        Indexed of the steps that should be plotted.
    steps_combo : List[int]
        Step numbers to calculate Q-factor for.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    gaussian_sigma: float, optional
        Sigma value for gaussian filter before processing.
    power_offset: float, optional
        Power offset for plot.
    power_limits: float, optional
        Limit power range on plot.

    Returns
    -------
    q_factor : np.ndarray
        1D array of the q-factors for the specified step combos.

    """
    p = power[:, 0]
    if p[0] > p[1]:
        p = p[::-1]
        counts = counts[::-1]

    weights = [extract_step_weight(voltage, counts, i) for i in steps]
    weights = savgol_filter(weights,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else weights
    
    if power_offset:
        p_offset = np.float64(power_offset)
    else:
        #Extract the 0 step to normalize the power and fix power offset
        #In some cases sigma value for guassian_filter maybe be unsuitable. 
        #Same goes for max height ratio (height=coeffcient*max) and distance when finding peaks
        weight = extract_step_weight(voltage, counts, 0)
        weight = 1/(weight + 0.01*weight[0])
        peaks, _ = find_peaks(weight, height=0.2*np.max(weight))
        p_offset = p[peaks[0]]
    
    print("Power Offset = ", p_offset)
    data = dict(zip(steps,weights))
    q_factor = np.zeros(len(steps_combo))

    for i, q_combo in enumerate(steps_combo):
        if power_limits is not None:
            masked_power = (power_limits[0]<p-p_offset) & (p-p_offset<power_limits[1])
            q_factor[i] = np.max(gaussian_filter(data[q_combo[0]][masked_power],gaussian_sigma))/np.max(gaussian_filter(data[q_combo[1]][masked_power],gaussian_sigma))
        else:
            q_factor[i] = np.max(gaussian_filter(data[q_combo[0]],gaussian_sigma))/np.max(gaussian_filter(data[q_combo[1]],gaussian_sigma))

    return q_factor
