# -*- coding: utf-8 -*-
""" Routines to bin Shapiro steps measurement to obtain a colormap.

"""
from math import ceil, copysign, floor

import numpy as np

from . import shapiro_step


def center_bin(bins):
    """Shift a bin by half a step and remove the last item.

    This function can be used to prepare a suitable x axis for plotting.

    """
    return 0.5*(bins[1] - bins[0]) + bins[:-1]


def bin_shapiro_steps(voltage, frequency=None, step_fraction=0.1, bins=None):
    """Compute the histogram associated with an IV curve.

    Parameters
    ----------
    voltage : np.ndarray
        Measured voltages. Should be a 1D array.

    frequency : float, optional
        Frequency at which the experiment was carried out in Hz.

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

    """
    # Determine the bins to use if necessary.
    if bins is None:

        # Compute the shapiro step voltage
        step = shapiro_step(frequency)*step_fraction

        min_v, max_v = np.min(voltage), np.max(voltage)
        neg_bins, pos_bins = abs(floor(min_v/step)), abs(ceil(max_v/step))
        total_bins = neg_bins + pos_bins
        if (total_bins) % 2 == 0:
            bins = np.linspace(neg_bins*copysign(step, min_v),
                               pos_bins*copysign(step, max_v),
                               total_bins)
        else:
            bins = np.linspace((neg_bins + 0.5)*copysign(step, min_v),
                               (pos_bins + 0.5)*copysign(step, max_v),
                               total_bins)

    return np.histogram(voltage, bins)


def bin_power_shapiro_steps(power, current, voltage, frequency,
                            step_fraction=0.1):
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

    Returns
    -------
    power : np.ndarray
        Linear array of the power used in the experiment.

    voltage : np.ndarray
        Linear array of the center of the voltage bins.

    histo : np.ndarray
        2D array of the voltage counts (expressed in current, ie multiplied by
        the current step). Each column correpond to a constant power and
        varying bias current.

    """
    # Identify in the data are properly indexed (lines == constant power)
    if len(set(power[0])) != 0:
        voltage = voltage.T

    # Build the power array and compute the current step
    power, indices = np.unique(power, return_index=True)
    current = np.unique(current)
    c_step = abs(current[1] - current[0])

    _, bins = bin_shapiro_steps(voltage[0], frequency, step_fraction)
    results = np.empty((len(power), len(bins)-1))
    for i in range(len(power)):
        results[i], _ = bin_shapiro_steps(voltage[i], bins=bins)
    results *= c_step

    # Flip the power axis if the power was decreasing during the scan.
    if indices[0] > indices[1]:
        results = results[::-1]

    return power, center_bin(bins)/shapiro_step(frequency), results
