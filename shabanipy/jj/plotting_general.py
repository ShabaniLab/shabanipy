# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------

"""Routines used to plot general current bias measurments as a function of extracted values and external parameters:
magnetic field, gate voltage etc.
"""

from typing import Optional

import numpy as np
from scipy import constants as cs
from scipy.signal import savgol_filter
import matplotlib as mpl
from matplotlib import pyplot as plt

from shabanipy.jj.utils import extract_switching_current
from shabanipy.jj.iv_analysis import extract_critical_current, analyse_vi_curve
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center, symmetrize_fraunhofer
from shabanipy.jj.fraunhofer.deterministic_reconstruction import extract_current_distribution
from shabanipy.plotting import jy_pink
jy_pink.register()


def plot_vi_dr_curve(
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    dr: np.ndarray,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    bias_limits: Optional[np.ndarray] = None,
    text: Optional[str] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the Voltage drop and dR 

    Parameters
    ----------
    bias : np.ndarray
        2D array of the bias current.
    voltage_drop : np.ndarray
        2D array of the measured voltage drop.
    savgol_windowl : int, optional
        Window length of savgol_filter applied to voltage drop.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter applied to voltage drop.
    bias_limits : np.ndarray, optional
        bias axis plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = fig.gca()
    
    # Use savgol_filter if params are available
    voltage_drop = savgol_filter(voltage_drop,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else voltage_drop
    

    m_ax.grid()
    ax2 = ax.twinx()
    ax.plot(bias*1e6, voltage_drop*1e3, color = 'blue', linewidth = 5)
    ax2.plot(bias*1e6, np.abs(dr),color = 'red', linewidth = 5)
    ax.set_xlabel('Bias(µA)', labelpad = 20)
    ax.set_ylabel(r'$\mathbf{V_{Drop}}$(mV)', labelpad = 20, color = 'blue')
    ax2.set_ylabel(r'$\mathbf{\frac{dV}{dI} (\Omega)}$', labelpad = 20, color = 'red')
    if bias_limits:
        ax.set_xlim(bias_limits)
    if text:
        ax.text(0.025,0.85,text, fontsize = 15, ha='left', va='center',transform=ax.transAxes)

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
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Use savgol_filter if params are set on differential resistance
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI

    pm = m_ax.pcolormesh(out_field*1e3, #field: 1e3 to convert from T to mT,
        bias*1e6, #bias: 1e6 to convert from A to µA,
        dV_dI,  #dV_dI: 1/1e2 to account of gain of amplifier hooked up to DMM
        vmin = cvmin if cvmin else  0,
        vmax = cvmax if cvmax else  200,
        cmap = 'jy_pink', 
        shading = 'auto',
        linewidth=0,
        rasterized = True
        )

    if out_field_limits:
         m_ax.set_xlim(out_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.set_xlabel('Out-of-plane field (mT)', labelpad = 20)
    m_ax.set_ylabel('Bias (µA)')

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02)
    cb.ax.set_xlabel(r'$\mathbf{\frac{dV}{dI} (\Omega)}$', labelpad = 10)


def plot_extracted_switching_current(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    threshold: float,
    out_field_range: Optional[float] = None,
    current_field_conversion: Optional[float] = None,
    correct_v_offset: Optional[bool] = True,
    symmetrize_fraun: Optional[bool] = False,
    center_fraun: Optional[bool] = True,
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
    out_field_range : float
        Range of out-of-plane field to extract critical current from.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT. 
    correct_v_offset : bool, optional
        Correct voltage offset when extracting switching current or not
    symmetrize_fraun : bool, optional
        Do you want to symmetrize the Fraunhofer or not. Symmetrizing is best when the Fraunhofer
        field range is uneven    
    center_fraun : bool, optional
        Center the fraunofer pattern around 0mT.  
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
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field
    
    if out_field_range:
        mask = (out_field[:,0]<out_field_range) & (-out_field_range<out_field[:,0])
        out_field = out_field[mask]
        bias = bias[mask]
        voltage_drop = voltage_drop[mask]

    #Extract switching current and use savgol_filter if params are available
    ic = extract_switching_current(bias, voltage_drop,
     threshold = threshold,
     correct_v_offset = True if correct_v_offset  else None
     )
    # ic = ic-min(ic)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic

    #Find max of fraunhofer(which should be in the center) and center field around 0
    field = out_field[:,0] - find_fraunhofer_center(out_field[:,0], ic) if center_fraun else out_field
    
    #Symmetrize the field and ic
    if symmetrize_fraun: field, ic = symmetrize_fraunhofer(field, ic)
    
    pm = m_ax.plot(field*1e3, #field: 1e3 factor to convert from T to mT 
        ic*1e6,    #ic: 1e6 factor to convert from A to µA
        color = 'royalblue',
        linewidth = 3
        )

    if out_field_limits:
         m_ax.set_xlim(out_field_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)
    
    m_ax.grid()
    m_ax.set_xlabel('Out-of-plane field (mT)', labelpad = 20)
    m_ax.set_ylabel('I$_{c}$ (µA)')


def plot_extracted_critical_current(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    points_mask: int = 10,
    peak_height: float = 0.8,
    current_field_conversion: Optional[float] = None,
    symmetrize_fraun: Optional[bool] = False,
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
    symmetrize_fraun : bool, optional
        Do you want to symmetrize the Fraunhofer or not. Symmetrizing is best when the Fraunhofer
        field range is uneven
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
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()

    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    #Extract critical current and use savgol_filter if params are available
    ic = extract_critical_current(bias, voltage_drop, points_mask = points_mask, peak_height = peak_height)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic

    #Find max of fraunhofer(which should be in the center) and center field around 0
    centered_field = out_field[:,0] - find_fraunhofer_center(out_field[:,0], ic)
    
    #Symmetrize the field and ic
    if symmetrize_fraun: centered_field, ic = symmetrize_fraunhofer(centered_field, ic)

    pm = m_ax.plot(centered_field*1e3, # field: 1e3 factor converts from T to mT
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


def plot_current_distribution(
    out_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    threshold: float,
    jj_length: float,
    jj_width: float,
    out_field_range: Optional[float] = None,
    current_field_conversion: Optional[float] = None,
    correct_v_offset: Optional[bool] = True,
    symmetrize_fraun: Optional[bool] = False,
    center_fraun: Optional[bool] = True,
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
    out_field_range : float
        Range of out-of-plane field to extract critical current from.
    current_field_conversion : float, optional
        Convert current being applied by Keithley to out-of-plane field in units of mA:mT
    correct_v_offset : bool, optional
        Correct voltage offset when extracting switching current or not
    symmetrize_fraun : bool, optional
        Do you want to symmetrize the Fraunhofer or not. Symmetrizing is best when the Fraunhofer
        field range is uneven
    center_fraun : bool, optional
        Center the fraunofer pattern around 0mT.  
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

    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()

    #Convert from current to field if conversion factor is available
    out_field = out_field/current_field_conversion if current_field_conversion else out_field

    if out_field_range:
        mask = (out_field[:,0]<out_field_range) & (-out_field_range<out_field[:,0])
        out_field = out_field[mask]
        bias = bias[mask]
        voltage_drop = voltage_drop[mask]
    #Extract switching current and use savgol_filter if params are available
    ic = extract_switching_current(bias, voltage_drop,
     threshold = threshold,
     correct_v_offset = True if correct_v_offset else None
     )
    # ic = ic-min(ic)
    ic = savgol_filter(ic, savgol_windowl, savgol_polyorder) if savgol_windowl and savgol_polyorder else ic

    #Find max of fraunhofer(which should be in the center) and center field around 0
    field = out_field[:,0] - find_fraunhofer_center(out_field[:,0], ic) if center_fraun else out_field
    
    #Symmetrize the field and ic
    if symmetrize_fraun: field, ic = symmetrize_fraunhofer(field, ic)

    #Extract current distributions with symmertizied field and ic
    x, jx = extract_current_distribution(field, ic, FIELD_TO_WAVENUM, jj_width, len(out_field))
    # jx = jx[::-1]
    pm = m_ax.plot(x*1e6, #x: 1e6 factor converts from m into µm
        jx.real, #  Jx: is in units of µA/µm
        linewidth = 7
        )

    m_ax.fill_between(x*1e6, jx.real,
     facecolor = 'lightblue'
     )

    if x_limits:
        m_ax.set_xlims(x_limits)
    if jx_limits:
        m_ax.set_ylims(jx_limits)

    m_ax.set_xlabel(r'x (µm)')
    m_ax.set_ylabel(r'J$_{x}$ (µA/µm)')


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
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI
    
    pm = m_ax.pcolormesh(inplane_field*1e3, #field: 1e3 to convert from T to mT,
        bias*1e6, #bias: 1e6 to convert from A to µA
        dV_dI, #dV_dI: 1 to account of gain of amplifier hooked up to DMM
        shading = 'auto',
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
    cb.ax.set_xlabel(r'$\mathbf{\frac{dV}{dI} (\Omega)}$', labelpad = 10)

def plot_inplane_vs_outofplane(
    inplane_field: np.ndarray,
    out_field: np.ndarray,
    dR: np.ndarray,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    cvmax: Optional[float] = None,
    cvmin: Optional[float] = None,
    in_field_limits: Optional[np.ndarray] = None,
    out_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the differential resistance as a function of in-plane magnetic field and bias.

    Parameters
    ----------
    inplane_field : np.ndarray
        2D array of the applied in-plane magnetic field.
    out_field : np.ndarray
        2D array of the applied out-of-plane magnetic field.
    dR : np.ndarray
        2D array of the differential resistance.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    cvmax : float, optional
        Colormap vmax value.
    cvmin : float, optional
        Colormap vmin value.
    in_field_limits : np.ndarray, optional
        In-plane field plot limits.
    out_field_limits : np.ndarray, optional
        Out-of-plane field plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    # Use savgol_filter if params are available
    dR = savgol_filter(dR,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dR
    
    pm = m_ax.pcolormesh(out_field*1e3, #field: 1e3 to convert from T to mT,
        inplane_field*1e3, #bias: 1e6 to convert from A to µA
        dR, #dV_dI: 1 to account of gain of amplifier hooked up to DMM
        shading = 'auto',
        vmin = cvmin if cvmin else  0,
        vmax = cvmax if cvmax else  200,
        cmap = 'jy_pink',
        linewidth=0,
        rasterized = True
        )

    if in_field_limits:
         m_ax.set_ylim(in_field_limits) 
    if out_field_limits: 
         m_ax.set_xlim(out_field_limits)

    m_ax.set_ylabel('In-plane Field (mT)', labelpad = 20)
    m_ax.set_xlabel('Out-of-plane Field (mT)')

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02,)
    cb.ax.tick_params(direction='in')
    cb.ax.set_xlabel(r'$\mathbf{\frac{dV}{dI} (\Omega)}$', labelpad = 10)
    
def plot_inplane_vs_Ic_Rn(
    inplane_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    ic_voltage_threshold: float,
    high_bias_threshold: float,
    ic_extraction_method: Optional[str] = 'iv_analysis',
    switching_bias_threshold: Optional[float] = 2e-5,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    ic_limits: Optional[np.ndarray] = None,
    rn_limits: Optional[np.ndarray] = None,
    in_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the Ic and Rn as a function of in-plane magnetic field.

    Parameters
    ----------
    inplane_field : np.ndarray
        2D array of the applied in-plane magnetic field.
    bias : np.ndarray
        2D array of the bias current.
    voltage_drop : np.ndarray
        2D array of the measured voltage drop.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    ic_extraction_method: str, Optional
        Choose method to extract critical current. Default is using "iv_analysis" function but sometimes
        "extract_switching_current" works better.
    switching_bias_threshold: float, Optional
        Value of threshold for extract_switching_current function
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    ic_limits : np.ndarray, optional
        ic axis plot limits.
    rn_limits : np.ndarray, optional
        rn axis plot limits.
    in_field_limits : np.ndarray, optional
        In-plane field plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = fig.gca()
    
    # Use savgol_filter if params are available
    voltage_drop = savgol_filter(voltage_drop,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else voltage_drop
    
    x = analyse_vi_curve(bias, voltage_drop, ic_voltage_threshold,high_bias_threshold)
    
    if ic_extraction_method == 'extract_switching_current':
        ic = extract_switching_current(bias,voltage_drop,switching_bias_threshold)
    elif ic_extraction_method == 'iv_analysis':
        ic = x[2]

    m_ax.grid()
    ax2 = ax.twinx()
    ax2.plot(inplane_field*1000, x[0], color = 'red', linewidth = 5, marker = 'x', markersize = 10)
    ax.plot(inplane_field*1000, ic*1e6,color = 'blue', linewidth = 5, marker = 'o', markersize = 10)
    ax.set_xlabel('In-plane Field (mT)', labelpad = 20)
    ax2.set_ylabel(r'R$_{n}$(Ω)', labelpad = 20, color = 'red')
    ax.set_ylabel(r'I$_{c}$(µA)', labelpad = 20, color = 'blue')
    if in_field_limits:
        ax.set_xlim(in_field_limits)
    if ic_limits:
        ax.set_ylim(ic_limits)
    if rn_limits:
        ax2.set_ylim(rn_limits)

def plot_inplane_vs_IcRn(
    inplane_field: np.ndarray,
    bias: np.ndarray,
    voltage_drop: np.ndarray,
    ic_voltage_threshold: float,
    high_bias_threshold: float,
    ic_extraction_method: Optional[str] = 'iv_analysis',
    switching_bias_threshold: Optional[float] = 2e-5,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    icrn_limits: Optional[np.ndarray] = None,
    in_field_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot Ic*Rn as a function of in-plane magnetic field.

    Parameters
    ----------
    inplane_field : np.ndarray
        2D array of the applied in-plane magnetic field.
    bias : np.ndarray
        2D array of the bias current.
    voltage_drop : np.ndarray
        2D array of the measured voltage drop.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    ic_extraction_method: str, Optional
        Choose method to extract critical current. Default is using "iv_analysis" function but sometimes
        "extract_switching_current" works better.
    switching_bias_threshold: float, Optional
        Value of threshold for extract_switching_current function    
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    icrn_limits : np.ndarray, optional
        icrn axis plot limits.
    in_field_limits : np.ndarray, optional
        In-plane field plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    # Use savgol_filter if params are available
    voltage_drop = savgol_filter(voltage_drop,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else voltage_drop
    
    x = analyse_vi_curve(bias, voltage_drop, ic_voltage_threshold,high_bias_threshold)
    if ic_extraction_method == 'extract_switching_current':
        ic = extract_switching_current(bias,voltage_drop,switching_bias_threshold)
    elif ic_extraction_method == 'iv_analysis':
        ic = x[2]

    m_ax.grid()
    m_ax.plot(inplane_field[:,0]*1000, x[0]*ic*1e6, color = 'green', linewidth = 5, marker = 'x', markersize = 10)
    m_ax.set_xlabel('In-plane Field (mT)', labelpad = 20)
    m_ax.set_ylabel(r'I$_{c}$R$_{n}$(Ω)', labelpad = 20, color = 'green')
    if in_field_limits:
        m_ax.set_xlim(in_field_limits)
    if icrn_limits:
        m_ax.set_ylim(icrn_limits)

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
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()

    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI

    pm = m_ax.pcolormesh(vg,bias*1e6, #bias: 1e6 to convert from A to µA
    dV_dI, #dV_dI: 1/1e2 to account of gain of amplifier hooked up to DMM
    vmin = cvmin if cvmin else  0,
    vmax = cvmax if cvmax else  200,
    cmap = 'jy_pink',
    shading = 'auto',
    linewidth=0,
    rasterized = True
     )

    if vg_limits:
         m_ax.set_xlim(vg_limits) 
    if bias_limits: 
         m_ax.set_ylim(bias_limits)

    m_ax.set_xlabel(r'$\mathbf{V_{g}}$(V)', labelpad = 20)
    m_ax.set_ylabel('Bias (µA)')
    
    if vg_limits:
        m_ax.set_xlim(vg_limits)
    if bias_limits:
        m_ax.set_ylim(bias_limits)

    cb = f.colorbar(pm, ax = m_ax,pad = 0.02, extend = 'max')
    cb.ax.tick_params(direction='in')
    cb.ax.set_xlabel(r'$\mathbf{\frac{dV}{dI} (\Omega)}$', labelpad = 10)
    

def plot_vg_vs_Ic_Rn(
    vg: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    ic_voltage_threshold: float,
    high_bias_threshold: float,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    ic_limits: Optional[np.ndarray] = None,
    rn_limits: Optional[np.ndarray] = None,
    vg_limits: Optional[np.ndarray] = None,
    log_rn: Optional[bool] = False,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot the Ic and Rn as a function of gate voltage.

    Parameters
    ----------
    vg : np.ndarray
        2D array of the applied gate voltage in volts.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    ic_limits : np.ndarray, optional
        ic axis plot limits.
    rn_limits : np.ndarray, optional
        rn axis plot limits.
    vg_limits : np.ndarray, optional
        Vg plot limits.
    log_rn : bool, optional
        Plot rn axis in log.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = fig.gca()
    
    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI
    
    x = analyse_vi_curve(bias, dV_dI, ic_voltage_threshold,high_bias_threshold)

    m_ax.grid()
    ax2 = ax.twinx()
    ax2.plot(vg[:,0], x[0], color = 'red', linewidth = 5, marker = 'x', markersize = 10)
    ax.plot(vg[:,0], x[2]*1e6, color = 'blue', linewidth = 5, marker = 'o', markersize = 10)
    ax.set_xlabel(r'$V_{g}$(V)', labelpad = 20)
    if log_rn:
        ax2.set_yscale('log')
    if vg_limits:
        ax.set_xlim(vg_limits)
    if ic_limits:
        ax.set_ylim(ic_limits)
    if rn_limits:
        ax2.set_ylim(rn_limits)
    ax2.set_ylabel(r'Log(R$_{n}$)(Ω)', labelpad = 20, color = 'red')
    ax.set_ylabel(r'I$_{c}$(µA)', labelpad = 20, color = 'blue')

def plot_vg_vs_IcRn(
    vg: np.ndarray,
    bias: np.ndarray,
    dV_dI: np.ndarray,
    ic_voltage_threshold: float,
    high_bias_threshold: float,
    savgol_windowl: Optional[int] = None,
    savgol_polyorder: Optional[int] = None,
    icrn_limits: Optional[np.ndarray] = None,
    vg_limits: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:
    """Plot Ic*Rn as a function of gate voltage.

    Parameters
    ----------
    vg : np.ndarray
        2D array of the applied gate voltage in volts.
    bias : np.ndarray
        2D array of the bias current.
    dV_dI : np.ndarray
        2D array of the differential resistance.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    savgol_windowl : int, optional
        Window length of savgol_filter.
    savgol_polyorder: int, optional
        Polyorder of savgol_filter.
    icrn_limits : np.ndarray, optional
        icrn axis plot limits.
    vg_limits : np.ndarray, optional
        In-plane field plot limits.
    fig_size : np.ndarray, optional
        Figure size of plot.
    debug : bool, optional
        Should debug information be provided, by default False.

    """
    f = plt.figure(constrained_layout=True, figsize = fig_size if fig_size else mpl.rcParams['figure.figsize'])
    m_ax = f.gca()
    
    # Use savgol_filter if params are available
    dV_dI = savgol_filter(dV_dI,savgol_windowl,savgol_polyorder) if savgol_windowl and savgol_polyorder else dV_dI
    
    x = analyse_vi_curve(bias, dV_dI, ic_voltage_threshold,high_bias_threshold)

    m_ax.grid()
    m_ax.plot(vg[:,0]*1000, x[0]*x[2]*1e6, color = 'red', linewidth = 5, marker = 'x', markersize = 10)
    m_ax.set_xlabel(r'$V_{g}$(V)', labelpad = 20)
    m_ax.set_ylabel(r'I$_{c}$R$_{n}$(Ω)', labelpad = 20, color = 'red')
    if vg_limits:
        m_ax.set_xlim(vg_limits)
    if icrn_limits:
        m_ax.set_ylim(icrn_limits)
