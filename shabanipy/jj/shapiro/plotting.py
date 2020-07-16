# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines used to plot Shapiro steps data.

"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt

from shabanipy.utils.plotting import add_title_and_save
from .binning import extract_step_weight


def plot_differential_resistance_map(
    power: np.ndarray,
    bias: np.ndarray,
    resistance: np.ndarray,
    transpose: bool = False,
    directory: Optional[str] = None,
    classifiers: Optional[Dict[int, Dict[str, Any]]] = None,
    dir_per_plot_format: bool = True,
    debug: bool = False,
) -> None:
    """Plot the differential resistance of a JJ in a Shapiro experiment.

    The sweep of the current bias is expected to occur on the last axis.

    Parameters
    ----------
    power : np.ndarray
        2D array of the applied microwave power.
    bias : np.ndarray
        2D array of the bias current.
    resistance : np.ndarray
        2D array of the differential resistance.
    transpose : bool, optional
        Should the data be transposed.
    directory : Optional[str], optional
        Path to a directory in which to save the figure, by default None
    classifiers : optional[Dict[int, Dict[str, Any]]], optional
        Dictionary of classifiers associated with the data, by default None
    dir_per_plot_format : bool, optional
        Should each plot format be saved in a separate subdirectory.
    debug : bool, optional
        Should debug information be provided, by default False

    """
    f = plt.figure(constrained_layout=True)
    m_ax = f.gca()

    if transpose:
        extent = (power[0, 0], power[-1, 0], bias[0, 0] * 1e6, bias[0, -1] * 1e6)
    else:
        extent = (bias[0, 0] * 1e6, bias[0, -1] * 1e6, power[0, 0], power[-1, 0])

    # Bin the data, find the peaks and its width to determine the most
    # appropriate range for the colorbar
    hist, bins = np.histogram(np.ravel(resistance), "auto")
    peaks, _ = find_peaks(hist, np.max(hist) / 2)
    widths = peak_widths(hist, peaks)
    # Use the right most peak and go to 2 times it full width
    cmax = bins[peaks[-1] + 3 * int(round(widths[0][-1]))]

    im = m_ax.imshow(
        resistance.T if transpose else resistance,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=0,
        # Use the average of the high bias resistance as reference for the color limits
        vmax=cmax,
    )
    cbar = f.colorbar(im, ax=m_ax, aspect=50)
    cbar.ax.set_ylabel("Differential resistance (Ω)")

    plabel = "RF Power (dBm)"
    clabel = "Current bias (µA)"
    if transpose:
        m_ax.set_xlabel(plabel)
        m_ax.set_ylabel(clabel)
    else:
        m_ax.set_ylabel(plabel)
        m_ax.set_xlabel(clabel)

    add_title_and_save(f, directory, classifiers, dir_per_plot_format)


def plot_shapiro_histogram(
    power: np.ndarray,
    voltage: np.ndarray,
    counts: np.ndarray,
    transpose: bool = False,
    mark_steps: Optional[List[int]] = None,
    mark_steps_limit: Optional[float] = None,
    voltage_limit: Optional[float] = None,
    directory: Optional[str] = None,
    classifiers: Dict[int, Dict[str, Any]] = None,
    dir_per_plot_format: bool = True,
    debug: bool = False,
) -> None:
    """Plot Shapiro steps histogram

    Parameters
    ----------
    power : np.ndarray
        2D array of the applied microwave power.
    voltage : np.ndarray
        1D array of the voltage bins.
    counts : np.ndarray
        2D array of the histogram current counts.
    transpose : bool, optional
        Should the data be transposed.
    mark_steps : Optional[List[int]], optional
        List of Shapiro steps that should be marked using dashed lines, by default None
    mark_steps_limit : Optional[float], optional
        Power limit (in fraction of the total range) for the dashed lines marking
        the steps, by default None
    directory : Optional[str], optional
        Path to a directory in which to save the figure, by default None
    classifiers : optional[Dict[int, Dict[str, Any]]], optional
        Dictionary of classifiers associated with the data, by default None
    dir_per_plot_format : bool, optional
        Should each plot format be saved in a separate subdirectory.
    debug : bool, optional
        Should debug information be provided, by default False

    """
    f = plt.figure(constrained_layout=True)
    m_ax = f.gca()

    p = power[:, 0]
    # If scan was taken with decreasing power flip the data
    if p[0] > p[1]:
        p = p[::-1]
        counts = counts[::-1]

    if transpose:
        extent = (p[0], p[-1], voltage[0], voltage[-1])
    else:
        extent = (voltage[0], voltage[-1], p[0], p[-1])

    # Copy the data before scaling them to be in µA
    counts = np.copy(counts)
    counts *= 1e6

    # Extract the -2 step and use the max value for color contrast
    weight = extract_step_weight(voltage, counts, -2)

    aux = 2 * np.max(weight)
    im = m_ax.imshow(
        counts.T if transpose else counts,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=0,
        # Use the average of the high bias resistance as reference for the color limits
        vmax=2 * np.max(weight),
    )
    cbar = f.colorbar(im, ax=m_ax, aspect=50)
    cbar.ax.set_ylabel("Counts (µA)")

    plabel = "RF Power (dBm)"
    clabel = "Voltage (hf/2e)"
    if transpose:
        m_ax.set_xlabel(plabel)
        m_ax.set_ylabel(clabel)
        if voltage_limit:
            m_ax.set_ylim((-voltage_limit, voltage_limit))
    else:
        m_ax.set_ylabel(plabel)
        m_ax.set_xlabel(clabel)
        if voltage_limit:
            m_ax.set_xlim((-voltage_limit, voltage_limit))

    if mark_steps:
        lims = [power[0], power[-1]]
        if mark_steps_limit:
            lims[1] = power[0] + mark_steps_limit * (power[-1] - power[0])
        if transpose:
            m_ax.hlines(mark_steps, *lims, linestyles="dashed")
        else:
            m_ax.vlines(mark_steps, *lims, linestyles="dashed")

    add_title_and_save(f, directory, classifiers, dir_per_plot_format)


def plot_step_weights(
    power: np.ndarray,
    voltage: np.ndarray,
    counts: np.ndarray,
    steps: List[int],
    ic: Union[float, np.ndarray],
    rn: Union[float, np.ndarray],
    directory: Optional[str] = None,
    classifiers: Dict[int, Dict[str, Any]] = None,
    dir_per_plot_format: bool = True,
    debug: bool = False,
) -> None:
    """Plot Shapiro steps weights as a function of power.

    Parameters
    ----------
    power : np.ndarray
        2D array of the applied microwave power.
    voltage : np.ndarray
        1D array of the voltage bins.
    counts : np.ndarray
        2D array of the histogram current counts.
    steps : List[int]
        Indexed of the steps that should be plotted.
    ic : float
        Critical current of the device.
    rn : float
        Normal resistance of the device.
    directory : Optional[str], optional
        Path to a directory in which to save the figure, by default None
    classifiers : optional[Dict[int, Dict[str, Any]]], optional
        Dictionary of classifiers associated with the daat, by default None
    dir_per_plot_format : bool, optional
        Should each plot format be saved in a separate subdirectory.
    debug : bool, optional
        Should debug information be provided, by default False

    """
    f = plt.figure(constrained_layout=True)
    m_ax = f.gca()

    if not isinstance(ic, float):
        ic = np.average(ic)
    if not isinstance(rn, float):
        rn = np.average(rn)

    # Extract the steps we want to plot and normalize by the critical current
    weights = [extract_step_weight(voltage, counts, i) / ic for i in steps]

    # Normalize the power by rn*ic**2 which should be the right scaling
    p = power[:, 0] - 10 * np.log10(rn * ic ** 2)

    for i, w in zip(steps, weights):
        m_ax.plot(p, w, label=f"Step {i}")

    m_ax.set_xlabel("RF Power (dB(Rn*Ic^2))")
    m_ax.set_ylabel("Counts (Ic)")
    m_ax.legend()

    add_title_and_save(f, directory, classifiers, dir_per_plot_format)
