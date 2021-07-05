# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------

"""Routines used to plot general current bias measurments as a function of exctracted values and external parameters:
magnetic field, gate voltage etc.
"""

from typing import Optional

import numpy as np
from scipy import constants as cs
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shabanipy.jj.utils import extract_switching_current
from shabanipy.jj.iv_analysis import extract_critical_current
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center, symmetrize_fraunhofer
from shabanipy.jj.fraunhofer.deterministic_reconstruction import extract_current_distribution


#Set plotting parameters
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["font.size"] = 50
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 4


#Create custom color map for plotting
def normalize_color(color):
    return [i/255 for i in color]

def make_color_dict(colors):
    color_dict = {
        'red': [],
        'blue': [],
        'green': [],
    }
    N = len(colors) - 1
    for i, color in enumerate(colors):
        if isinstance(color, str):
            color = hex_color_to_rgb(color)
        if any(i > 1 for i in color):
            color = normalize_color(color)
        for j, c in enumerate(['red', 'green', 'blue']):
            color_dict[c].append((i/N, color[j], color[j]))
    return color_dict

def register_color_map(cmap_name, colors):
    cdict = make_color_dict(colors)
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    plt.register_cmap(cmap = cmap)

color_pts = [
    (45, 96, 114),
    (243, 210, 181),
    (242, 184, 164),
    (242, 140, 143),
    (208, 127, 127)
]
register_color_map('jy_pink', color_pts)


def plot_fraunhofer(
    out_field: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    current_field_conversion: Optional[float] = None,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    bias_limits: Optional[np.ndarray] = None,
    out_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the differential resistance a function of out-of-plane field and bias: Fraunhofer pattern.

    Parameters
    ----------
    out_field : np.ndarray
        2D array of the applied out of plane field.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT. 
    savgol_windowl : int, optional
        Window length of savgol_filter (has to be an odd number).
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    cvmax : float, optional
        Colormap vmax value.
    cvmin : float, optional
        Colormap vmin value.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    out_field_limits : np.ndarray, optional
        Out-of-plane field axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()
    
    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Use savgol_filter if params are set on differential resistance
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI

    pm = m_ax.pcolormesh(out_field*1e3, #field: 1e3 to convert from T to mT,
        bias*1e6, #bias: 1e6 to convert from A to µA,
        dV_dI/1e2,  #dV_dI: 1/1e2 to account of gain of amplifier hooked up to DMM
        vmin = cvmin if cvmin else  0,
        vmax = cvmax if cvmax else  200,
        cmap = 'jy_pink', 
        linewidth=0,
        rasterized = True
        )

    if out_field_limits:
         m_ax.set_xlim(out_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.set_xlabel('Out-of-plane field (mT)', labelpad = 20)
    m_ax.set_ylabel('Bias (µA)')
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02)
    cb.ax.tick_params(direction='in',labelsize=50)
    cb.ax.set_xlabel(r'$\frac{dV}{dI} (\Omega)$', labelpad = 10)


def plot_extracted_switching_current(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    threshold: float,
    current_field_conversion: Optional[float] = None,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    bias_limits: Optional[np.ndarray] = None,
    out_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the extracted switching current (Ic) as a function of out-of-plane magnetic field from Fraunhofer pattern.

    Parameters
    ----------
    out_field : np.ndarray
        2D array of the applied out of plane field.
    bias : np.ndarray
        2D array of the bias current.
    voltage_drop : np.ndarray
        2D array of the voltage drop across junction.
    threshold : float
        Since there's a shift in the DMM the superconducting region isn't exactly around zero. This value is not constant and needs to be adjusted.
        This threshold sets the voltage range around zero used to determine the swicthing current. Usually the threshold is of the order of 1e-4.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT. 
    savgol_windowl : int, optional
        Window length of savgol_filter (has to be an odd number).
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    out_field_limits : np.ndarray, optional
        Out-of-plane field axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()
    
    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Extract switching current and use savgol_filter if params are available
    ic = extract_switching_current(bias, voltage_drop, threshold = threshold)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic
    
    pm = m_ax.plot(out_field*1e3, #field: 1e3 factor to convert from T to mT 
        ic*1e6,    #ic: 1e6 factor to convert from A to µA
        color = 'royalblue',
        linewidth = 5
        )

    if out_field_limits:
         m_ax.set_xlim(out_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)
    
    m_ax.grid()
    m_ax.set_xlabel('Out-of-plane field (mT)', labelpad = 20)
    m_ax.set_ylabel('I$_{c}$ (µA)')
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)


def plot_extracted_critical_current(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    points_mask: int = 10,
    peak_height: float = 0.8,
    current_field_conversion: Optional[float] = None,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    bias_limits: Optional[np.ndarray] = None,
    out_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the extracted critical current (Ic) as a function of out-of-plane magnetic field from Fraunhofer pattern.

    Parameters
    ----------
    out_field : np.ndarray
        2D array of the applied out of plane field.
    bias : np.ndarray
        2D array of the bias current.
     voltage_drop : np.ndarray
        2D array of the voltage drop across junction.
    points_mask : int
        Number of points to ignore on the sides of the VI curve when calculating derivative to find peaks 
        because sometimes there's abnormal peaks towards the sides.
    peak_heights : float
        Fraction of max height used for height in find_peaks.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT. 
    savgol_windowl : int, optional
        Window length of savgol_filter (has to be an odd number).
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    out_field_limits : np.ndarray, optional
        Out-of-plane field axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()

    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Extract critical current and use savgol_filter if params are available
    ic = extract_critical_current(bias, voltage_drop, points_mask = points_mask, peak_height = peak_height)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic

    pm = m_ax.plot(out_field*1e3, # field: 1e3 factor converts from T to mT
        ic*1e6,         #ic: 1e6 factor converts from A to µA
        color = 'royalblue',
        linewidth = 5
        )

    if out_field_limits:
         m_ax.set_xlim(out_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.grid()
    m_ax.set_xlabel('Out-of-plane field (mT)', labelpad = 20)
    m_ax.set_ylabel('I$_{c}$ (µA)')
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)


def plot_current_distribution(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    threshold: float,
    jj_length: float,
    jj_width: float,
    current_field_conversion: Optional[float] = None,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    x_limits: Optional[np.ndarray] = None,
    jx_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot to calculate the current distribution from the extracted switching current of the Fraunhofer pattern.

    Parameters
    ----------
    out_field : np.ndarray
        2D array of the applied out of plane field.
    bias : np.ndarray
        2D array of the bias current.
    voltage_drop : np.ndarray
        2D array of the differential resistance.
    threshold : float
        Since there's a shift in the DMM the superconducting region isn't exactly around zero. This value is not constant and needs to be adjusted.
        This threshold sets the voltage range around zero used to determine the swicthing current. Usually the threshold is of the order of 1e-4.
    jj_length : float
        Length of the junction in meters.
    jj_width : float
        Width of the junction in meters.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT. 
    savgol_windowl : int, optional
        Window length of savgol_filter (has to be an odd number).
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    x_limits: np.ndarray, optional
        x axis plot limits.
    jx_limits: np.ndarray, optional
        jx axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """

    PHI0 = cs.h / (2 * cs.e)  # magnetic flux quantum
    FIELD_TO_WAVENUM = 2 * np.pi * jj_length / PHI0  # B-field to beta wavenumber
    PERIOD = 2 * np.pi / (FIELD_TO_WAVENUM * jj_width)

    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()

    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Extract switching current and use savgol_filter if params are available
    ic = extract_switching_current(bias, voltage_drop, threshold = threshold)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic

    #Find max of fraunhofer(which should be in the center) and center field around 0
    centered_field = out_field[:,0] - find_fraunhofer_center(out_field[:,0], ic)
    
    #Symmetrize the field and ic
    sym_field, ic_sym = symmetrize_fraunhofer(centered_field, ic)

    #Extract current distributions with symmertizied field and ic
    x, jx = extract_current_distribution(sym_field, ic_sym, FIELD_TO_WAVENUM, jj_width, len(out_field))

    pm = m_ax.plot(x*1e6, #x: 1e6 factor converts from m into µm
        jx.real, #  Jx: is in units of µA/µm
        color = 'royalblue',
        linewidth = 7
        )

    m_ax.fill_between(x*1e6, jx.real,
     facecolor = 'lightblue'
     )

    if x_limits:
        m_ax.set_xlims(x_limits)
    if jx_limits:
        m_ax.set_ylims(jx_limits)

    m_ax.set_xlabel(r'$x$ (µm)')
    m_ax.set_ylabel(r'$J_{x}$ (µA/µm)')
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)


def plot_inplane_vs_bias(
    inplane_field: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    bias_limits: Optional[np.ndarray] = None,
    in_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the differential resistance as a function of in-plane magnetic field and bias.

    Parameters
    ----------
    inplane_field : np.ndarray
        2D array of the applied in-plane magnetic field.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    cvmax : float, optional
        Colormap vmax value.
    cvmin : float, optional
        Colormap vmin value.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    in_field_limits : np.ndarray, optional
        In-plane field plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()
    
    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI
    
    pm = m_ax.pcolormesh(inplane_field*1e3, #field: 1e3 to convert from T to mT,
        bias*1e6, #bias: 1e6 to convert from A to µA
        dV_dI/1e2, #dV_dI: 1/1e2 to account of gain of amplifier hooked up to DMM
        vmin = cvmin if cvmin else  0,
        vmax = cvmax if cvmax else  200,
        cmap = 'jy_pink',
        linewidth=0,
        rasterized = True
        )

    if in_field_limits:
         m_ax.set_xlim(in_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.set_xlabel('In-plane Field (mT)', labelpad = 20)
    m_ax.set_ylabel('Bias (µA)')

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02,)
    cb.ax.tick_params(direction='in',labelsize=50)
    cb.ax.set_xlabel(r'$\frac{dV}{dI} (\Omega)$', labelpad = 10)
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)


def plot_vg_vs_bias(
    vg: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    bias_limits: Optional[np.ndarray] = None,
    vg_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the differential resistance as a function of out gate voltage (Vg) and bias.

    Parameters
    ----------
    vg : np.ndarray
        2D array of the applied gate voltage in volts.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    current_field_conversion : float, optional
        Convert out of plane current to field if keithley is used.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    cvmax : float, optional
        Colormap vmax value.
    cvmin : float, optional
        Colormap vmin value.
    bias_limits : np.ndarray, optional
        Bias axis plot limits.
    vg_limits : np.ndarray, optional
        Vg axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, dpi = 400, figsize = fig_size if fig_size else (30,12))
    m_ax = f.gca()

    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI

    pm = m_ax.pcolormesh(vg,bias*1e6, #bias: 1e6 to convert from A to µA
    dV_dI/1e2, #dV_dI: 1/1e2 to account of gain of amplifier hooked up to DMM
    vmin = cvmin if cvmin else  0,
    vmax = cvmax if cvmax else  200,
    cmap = 'jy_pink',
    linewidth=0,
    rasterized = True
     )

    if vg_limits:
         m_ax.set_xlim(vg_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.set_xlabel(r'$V_{g}$(V)', labelpad = 20)
    m_ax.set_ylabel('Bias (µA)')

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02,)
    cb.ax.tick_params(direction='in',labelsize=50)
    cb.ax.set_xlabel(r'$\frac{dV}{dI} (\Omega)$', labelpad = 10)
    
    m_ax.tick_params(axis='x', labelsize=50)
    m_ax.tick_params(axis='y', labelsize=50)