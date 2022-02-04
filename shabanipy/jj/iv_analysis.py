# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Utility functions to analyse V-I characteristic.

"""
from typing import Tuple

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from lmfit.models import LinearModel

from .utils import correct_voltage_offset


def analyse_vi_curve(
    current_bias: np.ndarray,
    measured_voltage: np.ndarray,
    ic_voltage_threshold: float = 1e-4,
    high_bias_threshold: float = 10e-6,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the critical and excess current along with the normal resistance.

    All values are extracted for cold and hot electrons. The cold side is the
    one on which the bias is ramped from 0 to large value, the hot one the one
    on which bias is ramped towards 0.

    Parameters
    ----------
    current_bias : np.ndarray
        N+1D array of the current bias applied on the junction in A.
    measured_voltage : np.ndarray
        N+1D array of the voltage accross the junction in V.
    ic_voltage_threshold : float, optional
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float, optional
        Positive bias value above which the data can be used to extract the
        normal resistance.
    debug : bool, optional
        Generate summary plots of the fitting.

    Returns
    -------
    rn_c : np.ndarray
        Normal resistance evaluated on the cold electron side. ND array
    rn_h : np.ndarray
        Normal resistance evaluated on the hot electron side. ND array
    ic_c : np.ndarray
        Critical current evaluated on the cold electron side. ND array
    ic_h : np.ndarray
        Critical current evaluated on the hot electron side. ND array
    iexe_c : np.ndarray
        Excess current evaluated on the cold electron side. ND array
    iexe_h : np.ndarray
        Excess current evaluated on the hot electron side. ND array

    """
    ic_c = np.empty(current_bias.shape[:-1])
    ic_h = np.empty(current_bias.shape[:-1])
    rn_c = np.empty(current_bias.shape[:-1])
    rn_h = np.empty(current_bias.shape[:-1])
    ie_c = np.empty(current_bias.shape[:-1])
    ie_h = np.empty(current_bias.shape[:-1])

    # Iterate on additional dimensions
    it = np.nditer(current_bias[..., 0], ["multi_index"])

    for b in it:
        m_index = it.multi_index

        # Extract the relevant sweeps.
        cb = current_bias[m_index]
        mv = measured_voltage[m_index]

        # Determine the hot and cold electron side
        if cb[0] < 0.0:
            cold_value = lambda p, n: abs(p)
            hot_value = lambda p, n: abs(n)
        else:
            cold_value = lambda p, n: abs(n)
            hot_value = lambda p, n: abs(p)
            
        # Sort the data so that the bias always go from negative to positive
        sorting_index = np.argsort(cb)
        cb = cb[sorting_index]
        mv = mv[sorting_index]

        # Index at which the bias current is zero
        index = np.argmin(np.abs(cb))

        masked_mv_l = np.where(np.greater(mv[:index],-ic_voltage_threshold))[0]
        masked_mv_r = np.where(np.less(mv[index:],ic_voltage_threshold))[0] +index

        # Extract the critical current on the positive and negative branch
        ldidv = np.diff(mv[masked_mv_l])/np.diff(cb[masked_mv_l])
        rdidv = np.diff(mv[masked_mv_r])/np.diff(cb[masked_mv_r])

        lpeak, _ = find_peaks(ldidv, max(ldidv)*0.5 )
        rpeak, _ = find_peaks(rdidv, max(rdidv)*0.5 )
        
        if lpeak.size == 0:
            ic_n = 0.0
        else:
            ic_n = abs(cb[masked_mv_l[0]+lpeak[-1]])
        
        if rpeak.size == 0:
            ic_p = 0.0
        else:
            ic_p = abs(cb[rpeak[0]+index])

        # # Index at which the bias current is zero
        # index = np.argmin(np.abs(cb))
        # # Extract the critical current on the positive and negative branch
        # ic_n = cb[np.max(np.where(np.less(mv[:index], -ic_voltage_threshold))[0])]
        # ic_p = cb[ eshold))[0]) + index
        # ]
        # # print(ic_n, ic_p)
        # print(np.less(mv[:index], -ic_voltage_threshold)[0])

        # Fit the high positive/negative bias to extract the normal resistance
        # excess current and their product
        index_pos = np.argmin(np.abs(cb - high_bias_threshold))
        index_neg = np.argmin(np.abs(cb + high_bias_threshold))

        model = LinearModel()
        pars = model.guess(mv[index_pos:], x=cb[index_pos:])
        pos_results = model.fit(mv[index_pos:], pars, x=cb[index_pos:])

        pars = model.guess(mv[index_neg:], x=cb[index_neg:])
        neg_results = model.fit(mv[:index_neg], pars, x=cb[:index_neg])

        rn_p = pos_results.best_values["slope"]
        # Iexe p
        iexe_p = -pos_results.best_values["intercept"] / rn_p

        rn_n = neg_results.best_values["slope"]
        # Iexe n
        iexe_n = neg_results.best_values["intercept"] / rn_n

        if debug:
            # Prepare a summary plot: full scale
            fig = plt.figure(constrained_layout=True)
            ax = fig.gca()
            ax.plot(cb * 1e6, mv * 1e3)
            ax.plot(
                cb[index:] * 1e6,
                model.eval(pos_results.params, x=cb[index:]) * 1e3,
                "--k",
            )
            ax.plot(
                cb[: index + 1] * 1e6,
                model.eval(neg_results.params, x=cb[: index + 1]) * 1e3,
                "--k",
            )
            ax.set_xlabel("Bias current (µA)")
            ax.set_ylabel("Voltage drop (mV)")

            # Prepare a summary plot: zoomed in
            mask = np.logical_and(np.greater(cb, -3 * ic_p), np.less(cb, 3 * ic_p))
            if np.any(mask):
                fig = plt.figure(constrained_layout=True)
                ax = fig.gca()
                ax.plot(cb * 1e6, mv * 1e3)
                aux = model.eval(pos_results.params, x=cb[index:]) * 1e3
                ax.plot(
                    cb[index:] * 1e6,
                    model.eval(pos_results.params, x=cb[index:]) * 1e3,
                    "--",
                )
                ax.plot(
                    cb[: index + 1] * 1e6,
                    model.eval(neg_results.params, x=cb[: index + 1]) * 1e3,
                    "--",
                )
                ax.set_xlim(
                    (
                        -3 * cold_value(ic_p, ic_n) * 1e6,
                        3 * cold_value(ic_p, ic_n) * 1e6,
                    )
                )
                aux = mv[mask]
                ax.set_ylim((np.min(mv[mask]) * 1e3, np.max(mv[mask]) * 1e3,))
                ax.set_xlabel("Bias current (µA)")
                ax.set_ylabel("Voltage drop (mV)")

            plt.show()

        rn_c[m_index] = cold_value(rn_p, rn_n)
        rn_h[m_index] = hot_value(rn_p, rn_n)
        ic_c[m_index] = cold_value(ic_p, ic_n)
        ic_h[m_index] = hot_value(ic_p, ic_n)
        ie_c[m_index] = cold_value(iexe_p, iexe_n)
        ie_h[m_index] = hot_value(iexe_p, iexe_n)
    return (rn_c, rn_h, ic_c, ic_h, ie_c, ie_h)


def extract_critical_current(    
    current_bias: np.ndarray,
    measured_voltage: np.ndarray,
    points_mask: int = 10,
    peak_height: float = 0.8,
    debug: bool = False,
)-> Tuple[np.ndarray]:
    """Extract the critical current

    All values are extracted for cold and hot electrons. The cold side is the
    one on which the bias is ramped from 0 to large value, the hot one the one
    on which bias is ramped towards 0.

    Parameters
    ----------
    current_bias : np.ndarray
        N+1D array of the current bias applied on the junction in A.
    measured_voltage : np.ndarray
        N+1D array of the voltage accross the junction in V.
    points_mask : int
        Number of points to ignore on the sides of the VI curve when calculating derivative to find peaks 
        because sometimes there's abnormal peaks 
    peak_heights : float
        Relative peak heights to use to find peaks

    Returns
    -------
    Ic_c : np.ndarray
        Critical current evaluated on the cold electron side. ND array

    """
    ic_c = np.empty(current_bias.shape[:-1])

    # Iterate on additional dimensions
    it = np.nditer(current_bias[..., 0], ["multi_index"])

    for b in it:
        m_index = it.multi_index

        # Extract the relevant sweeps.
        cb = current_bias[m_index]
        mv = measured_voltage[m_index]

        # Determine the hot and cold electron side
        if cb[0] < 0.0:
            cold_value = lambda p, n: abs(p)
            hot_value = lambda p, n: abs(n)
        else:
            cold_value = lambda p, n: abs(n)
            hot_value = lambda p, n: abs(p)

        # Sort the data so that the bias always go from negative to positive
        sorting_index = np.argsort(cb)
        cb = cb[sorting_index]
        mv = mv[sorting_index]

        # Index at which the bias current is zero
        index = np.argmin(np.abs(cb))

        # Extract the critical current on the positive and negative branch
        
        didv = np.diff(mv)/np.diff(cb)
        # didv = gaussian_filter(didv,1)
        
        # l_limit = 10
        # r_limit = 180
        # ldidv = didv[l_limit:index]
        # rdidv = didv[index:index+r_limit]



        ldidv = didv[:index]
        rdidv = didv[index:]

        lpeak, _ = find_peaks(ldidv, height = math.floor(max(ldidv[:-points_mask]))*peak_height) 
        rpeak, _ = find_peaks(rdidv, height = math.floor(max(rdidv[:-points_mask]))*peak_height)
        
        if lpeak.size != 0:
            ic_n = abs(cb[lpeak[-1]])
        else:
             ic_n = abs(cb[np.argmax(didv[:index])])
        
        if rpeak.size != 0:
            ic_p = abs(cb[rpeak[0]+index])
        else:
            ic_p = abs(cb[np.argmax(didv[index:])+index])




        # if lpeak.size != 0:
        #     ic_n = abs(cb[lpeak[-1]+l_limit])
        # else:
        #     ic_n = abs(cb[np.argmax(didv[:index])+l_limit])
        # if rpeak.size != 0:
        #     ic_p = abs(cb[rpeak[0]+index])
        # else:
        #     ic_p = abs(cb[np.argmax(didv[index:])+index])

        ic_c[m_index] = cold_value(ic_p,ic_n)
        
    return ic_c

