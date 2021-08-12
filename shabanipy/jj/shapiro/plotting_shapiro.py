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
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from .binning import extract_step_weight

#Set plotting parameters
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["font.size"] = 30
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 4


def plot_differential_resistance_map(
    power: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    power_offset: Optional[float] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    power_limits: Optional[np.ndarray] = None,
    bias_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    transpose: Optional[bool] = False,
    debug: bool = False,
) -> None:
    """Plot the differential resistance of a JJ in a Shapiro experiment.

    Parameters
    ----------
    power : np.ndarray
        2D array of the applied microwave power.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    power_offset: float, optional
        Power offset for plot.
    cvmax : int, optional
        Colormap vmax value.
    cvmin : int, optional
        Colormap vmin value.
    power_limits : np.ndarray, optional
        Power axis plot limits.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    transpose : bool, optional
        Should the data be transposed.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (15,12))
    m_ax = f.gca()


    if power_offset:
         p_offset = np.float64(power_offset) 
    else:
        #Find index for zero current bias
        index = np.argmin(abs(bias[0]))
        #As a function of power, find first peak where resistance starts rising
        peaks, _ = find_peaks(abs(dV_dI[:,index]), max(dV_dI[:,index]*0.2))
        p_offset = power[:,0][peaks[0]]
    
    print("Power Offset = ", p_offset)

    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI.real,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI.real

    if transpose:
        extent = (power[0, 0]- p_offset, power[-1, 0]- p_offset, bias[0, 0] * 1e6, bias[0, -1] * 1e6)
    else:
        extent = (bias[0, 0] * 1e6, bias[0, -1] * 1e6, power[0, 0]- p_offset, power[-1, 0] -  p_offset)

    # Bin the data, find the peaks and its width to determine the most appropriate range for the colorbar
    hist, bins = np.histogram(np.ravel(dV_dI), "auto")
    peaks, _ = find_peaks(hist, np.max(hist) / 2)
    widths = peak_widths(hist, peaks)

    im = m_ax.imshow(
        dV_dI.T if transpose else dV_dI,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin = cvmin if cvmin else 0,
        vmax = cvmax if cvmax else bins[peaks[-1] + 5 * int(round(widths[0][-1]))],
    )

    cbar = f.colorbar(im, ax=m_ax, aspect=50)
    cbar.ax.set_ylabel(r"$\frac{dV}{dI}$ (Ω)")

    plabel = "RF Power (dBm)"
    clabel = "Current Bias (µA)"

    if transpose:
        m_ax.set_xlabel(plabel)
        m_ax.set_ylabel(clabel)
        if power_limits:
            m_ax.set_xlim(power_limits)
        if bias_limits:
            m_ax.set_ylim(bias_limits)
        
    else:
        m_ax.set_ylabel(plabel)
        m_ax.set_xlabel(clabel)
        if power_limits:
            m_ax.set_ylim(power_limits)
        if bias_limits:
            m_ax.set_xlim(bias_limits)


def plot_shapiro_histogram(
    power: np.ndarray,
    voltage: np.ndarray,
    counts: np.ndarray,
    I_c: float = 1.0,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    power_offset: Optional[float] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    power_limits: Optional[np.ndarray] = None,
    bias_limits: Optional[np.ndarray] = None,
    mark_steps: Optional[List[int]] = None,
    mark_steps_limit: Optional[float] = None,
    fig_size: Optional[np.ndarray] = None,
    transpose: bool = False,
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
    I_c : float
        Critical current value for normalzing counts.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    power_offset: float, optional
        Power offset for plot.
    cvmax : int, optional
        Colormap vmax value.
    cvmin : int, optional
        Colormap vmin value.
    power_limits : np.ndarray, optional
        Power axis plot limits.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    mark_steps : List[int], optional
        List of Shapiro steps that should be marked using dashed lines, by default None.
    mark_steps_limit : float, optional
        Power limit (in fraction of the total range) for the dashed lines marking
        the steps, by default None.
    fig_size : np.ndarray, optional
        Figure size of plot.
    transpose : bool, optional
        Should the data be transposed.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (15,12))
    m_ax = f.gca()

    p = power[:, 0]
    # If scan was taken with decreasing power flip the data
    if p[0] > p[1]:
        p = p[::-1]
        counts = counts[::-1]

    # Copy the data before scaling them to be in µA
    counts = np.copy(counts) 

    if power_offset:
        p_offset = np.float64(power_offset)
    else:
        # Extract the 0 step to normalize the power and fix power offset
        #In some cases sigma value for guassian_filter maybe be unsuitable. 
        #Same goes for max height ratio (height=coeffcient*max) and distance when finding peaks
        weight = extract_step_weight(voltage, counts, 0)
        weight = 1/(weight + 0.01*weight[0])
        peaks, _ = find_peaks(weight, height=0.2*np.max(weight))
        p_offset = p[peaks[0]]
    
    print("Power Offset = ", p_offset)

    # Use savgol_filter if params are available
    counts = savgol_filter(counts,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else counts

    #Normalize counts by critical current
    counts *= 1e6/I_c

    if transpose:
        extent = (p[0] - p_offset, p[-1] - p_offset, voltage[0], voltage[-1])
    else:
        extent = (voltage[0], voltage[-1], p[0] - p_offset, p[-1] - p_offset)
    
    if cvmax:
        cmax = cvmax
    else:
        # Extract the -2 step and use the max value for color contrast
        weight = extract_step_weight(voltage, counts, -2)
        cmax = 0.7 * np.max(weight)

    im = m_ax.imshow(
        counts.T if transpose else counts,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=cvmin if cvmin else 0,
        vmax=cmax,
    )

    cbar = f.colorbar(im, ax=m_ax, aspect=50,location="top",shrink=0.8)
    cbar.ax.set_xlabel('Counts (I${_c}$)',labelpad = 20)
    cbar.ax.tick_params(labelsize=20)

    plabel = "RF Power (dBm)"
    clabel = r"Voltage ($\dfrac{hf}{2e}$)"
    if transpose:
        m_ax.set_xlabel(plabel)
        m_ax.set_ylabel(clabel)
        if power_limits:
            m_ax.set_xlim(power_limits)
        if bias_limits:
            m_ax.set_ylim(bias_limits)
    else:
        m_ax.set_ylabel(plabel)
        m_ax.set_xlabel(clabel)
        if power_limits:
            m_ax.set_ylim(power_limits)
        if bias_limits:
            m_ax.set_xlim(bias_limits)

    if mark_steps:
        if power_limits:
            lims = (power_limits[0]+p_offset, p[int(len(p)/5)])
        else:
            lims = [p[0], p[int(len(p)/5)]]
        if mark_steps_limit:
            lims[1] = (power[0]) + mark_steps_limit * ((power[-1]) - (power[0]))
        if transpose:
            m_ax.hlines(mark_steps, *lims- p_offset, linestyles="dashed", color = 'white')
        else:
            m_ax.vlines(mark_steps, *lims- p_offset, linestyles="dashed", color = 'white')


def plot_step_weights(
    power: np.ndarray,
    voltage: np.ndarray,
    counts: np.ndarray,
    steps: List[int],
    ic: Union[float, np.ndarray],
    rn: Optional[Union[float, np.ndarray]] = None,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    power_offset: Optional[float] = None,
    counts_limits: Optional[np.ndarray] = None,
    power_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
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
    ic : Union[float, np.ndarray]
        Critical current of the device.
    rn : Union[float, np.ndarray], optional
        Normal resistance of the device.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    power_offset: float, optional
        Power offset for plot.
    counts_limits : np.ndarray, optional
       Counts limits in plot.
    power_limits: float, optional
        Limit power range on plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (15,12))
    m_ax = f.gca()

    if not isinstance(ic, float):
        ic = np.average(ic)
        
    # Extract the steps we want to plot and normalize by the critical current
    weights = [extract_step_weight(voltage, counts, i) / ic for i in steps]
    weights = savgol_filter(weights,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else weights


    # Normalize the power by rn*ic**2 which should be the right scaling if rn is provided
    if isinstance(rn, float):
        rn = np.average(rn)
        p = power[:, 0] - 10 * np.log10(rn * ic ** 2)
    else:
        p = power[:, 0]
    
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

    #Plot figure and smooth using gaussian_filter with a sigma of 1.5
    for i, w in zip(steps, weights):
        m_ax.plot(p - p_offset, gaussian_filter(w,1.5), label=f"Step {i}", linewidth = 7)
        if power_limits:
            masked_power = power_limits[0]<p-p_offset<power_limits[1]
            print(f"Step {i}",f"Max = {np.max(w[masked_power])}")
        else:
            print(f"Step {i}",f"Max = {np.max(w)}")
    
    if power_limits:
            m_ax.set_xlim(power_limits)
    if counts_limits:
            m_ax.set_ylim(counts_limits)
            
    m_ax.set_xlabel("RF Power (dB)")
    m_ax.set_ylabel("Counts (I$_c$)")
    m_ax.legend()
