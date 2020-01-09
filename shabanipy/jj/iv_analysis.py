"""Utility functions to analyse V-I characteristic.

"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

def analyse_vi_curve(current_bias, measured_voltage, voltage_offset_correction,
                     ic_voltage_threshold, high_bias_threshold, plots=True,
                     plot_title=""):
    """Extract the critical and excess current along with the normal resistance.

    All values are extracted for cold and hot electrons. The cold side is the
    one on which the bias is ramped from 0 to large value, the hot one the one
    on which bias is ramped towards 0.

    Parameters
    ----------
    current_bias : np.ndarray
        Current bias applied on the junction in A.
    measured_voltage : np.ndarray
        Voltage accross the junction in V.
    voltage_offset_correction : int | float
        Number of points around 0 bias on which to average to correct for the
        offset in the DC measurement, or actual offset to substract.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    plots : bool, optional
        Generate summary plots of the fitting.

    Returns
    -------
    voltage_offset_correction : float
        Offset used to correct the measured voltage.
    rn_c : float
        Normal resistance evaluated on the cold electron side.
    rn_h : float
        Normal resistance evaluated on the hot electron side.
    ic_c : float
        Critical current evaluated on the cold electron side.
    ic_h : float
        Critical current evaluated on the hot electron side.
    iexe_c : float
        Excess current evaluated on the cold electron side.
    iexe_h : float
        Excess current evaluated on the hot electron side.

    """
    # Determine the hot and cold electron side
    if current_bias[0] < 0.0:
        cold_value = lambda p, n: abs(p)
        hot_value = lambda p, n: abs(n)
    else:
        cold_value = lambda p, n: abs(n)
        hot_value = lambda p, n: abs(p)


    # Sort the data so that the bias always go from negative to positive
    sorting_index = np.argsort(current_bias)
    current_bias = current_bias[sorting_index]
    measured_voltage = measured_voltage[sorting_index]

    # Index at which the bias current is zero
    index = np.argmin(np.abs(current_bias))

    # Correct the offset in the voltage data
    if voltage_offset_correction and isinstance(voltage_offset_correction, int):
        avg_sl = slice(index-voltage_offset_correction+1,
                       index+voltage_offset_correction)
        voltage_offset_correction = np.average(measured_voltage[avg_sl])

    measured_voltage -= voltage_offset_correction

    # Extract the critical current on the positive and negative branch
    # Express them in µA
    ic_n = current_bias[np.max(
                            np.where(
                                np.less(measured_voltage[:index],
                                        -ic_voltage_threshold
                                        )
                                )[0]
                            )
                        ]
    ic_p = current_bias[np.min(
                            np.where(
                                np.greater(measured_voltage[index:],
                                           ic_voltage_threshold
                                           )
                                )[0]
                            ) + index
                        ]

    # Fit the high positive/negative bias to extract the normal resistance
    # excess current and their product
    index_pos = np.argmin(np.abs(current_bias - high_bias_threshold))
    index_neg = np.argmin(np.abs(current_bias + high_bias_threshold))

    model = LinearModel()
    pars = model.guess(measured_voltage[index_pos:],
                        x=current_bias[index_pos:])
    pos_results = model.fit(measured_voltage[index_pos:], pars,
                            x=current_bias[index_pos:])

    pars = model.guess(measured_voltage[index_neg:],
                       x=current_bias[index_neg:])
    neg_results = model.fit(measured_voltage[:index_neg], pars,
                            x=current_bias[:index_neg])

    rn_p = pos_results.best_values["slope"]
    # Iexe p
    iexe_p = -pos_results.best_values["intercept"]/rn_p

    rn_n = neg_results.best_values["slope"]
    # Iexe n
    iexe_n = -neg_results.best_values["intercept"]/rn_n

    # XXX Fix layout + title plus positive bias plot
    if plots:
        # Prepare a summary plot: full scale
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(plot_title)
        ax = fig.gca()
        ax.plot(current_bias*1e6, measured_voltage*1e3)
        ax.plot(current_bias[index:]*1e6,
                model.eval(pos_results.params, x=current_bias[index:])*1e3,
                "--k")
        ax.plot(current_bias[:index+1]*1e6,
                model.eval(neg_results.params, x=current_bias[:index+1])*1e3,
                "--k")
        ax.set_xlabel("Bias current (µA)")
        ax.set_ylabel("Voltage drop (mV)")

        # Prepare a summary plot: zoomed in
        mask = np.logical_and(np.greater(current_bias*1e6, -3*ic_p),
                              np.less(current_bias*1e6, 3*ic_p))
        if np.any(mask):
            fig = plt.figure(constrained_layout=True)
            fig.suptitle(plot_title + ": zoom")
            ax = fig.gca()
            ax.plot(current_bias*1e6, measured_voltage*1e3)
            aux = model.eval(pos_results.params, x=current_bias[index:])*1e3
            ax.plot(current_bias[index:]*1e6,
                    model.eval(pos_results.params, x=current_bias[index:])*1e3,
                    "--")
            ax.plot(current_bias[:index+1]*1e6,
                    model.eval(neg_results.params, x=current_bias[:index+1])*1e3,
                    "--")
            ax.set_xlim((-3*cold_value(ic_p, ic_n), 3*cold_value(ic_p, ic_n)))
            ax.set_ylim((np.min(measured_voltage[mask]*1e3),
                        np.max(measured_voltage[mask]*1e3)))
            ax.set_xlabel("Bias current (µA)")
            ax.set_ylabel("Voltage drop (mV)")

    return (
        voltage_offset_correction,
        cold_value(rn_p, rn_n),
        hot_value(rn_p, rn_n),
        cold_value(ic_p, ic_n),
        hot_value(ic_p, ic_n),
        cold_value(iexe_p, iexe_n),
        hot_value(iexe_p, iexe_n),
    )
