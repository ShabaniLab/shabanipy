# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tools to analyse cavity in the notch geometry.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.resonators.resonator_tools import circuit
from shabanipy.resonators.utils import center_data

def lorentzian(x, x_center, width, amplitude):
    """Standard lorentzian function.

    """
    return amplitude/(1 + 2j*((x - x_center) / width))


def perfect_notch(freq, f_c, width, amplitude, phase, baseline, phase_offset):
    """Transmission of a line with a single notch resonator.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies at which to evaluate the transmission
    f_c : float
        Central frequency of the resonator.
    width : float
        Width of the resonance in unit of the frequency.
    amplitude : float
        Depth of the resonance corresponds to the loaded quality factor divided
        by the external quality factor.
    phase : float
        Phase of the external quality factor.
    baseline : float
        Amplitude of the baseline.
    phase_offset : float
        Phase offset of the calculation.

    """
    return (baseline*np.exp(1j*phase_offset) *
            (1 - np.exp(1j*phase)*lorentzian(freq, f_c, width, amplitude)))


def fano_notch(freq, f_c, width, amplitude, phase, baseline, phase_offset,
               fano_amp, fano_phase):
    """Transmission of a line with a notch resonator and a parasitic mode.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies at which to evaluate the transmission
    f_c : float
        Central frequency of the resonator.
    width : float
        Width of the resonance in unit of the frequency.
    amplitude : float
        Depth of the resonance corresponds to the loaded quality factor divided
        by the external quality factor.
    phase : float
        Phase of the external quality factor.
    baseline : float
        Amplitude of the baseline.
    phase_offset : float
        Phase offset of the calculation.
    fano_amp : float
        Transmission amplitude of the parasitic mode.
    fano_phase : [type]
        Phase of the parasitic mode.

    """
    fano = fano_amp*np.exp(1j*fano_phase)
    shape = (1 + fano - np.exp(1j*phase) *
             lorentzian(freq, f_c, width,  amplitude))
    return (baseline*np.exp(1j*phase_offset) / np.abs(1 + fano) * shape)

def fit_complex(frequency,
                complex_data,
                powers=None,
                delay=None,
                gui_fit=False,
                return_fit_data=False,
                delay_range=(-.1,.1),
                save_gui_fits=False,
                save_gui_fits_path=None,
                save_gui_fits_prefix=None,
                save_gui_fits_suffix=None,
                save_gui_fits_title=None,
                save_gui_fits_filetype='.png',
                live_result=""):
    """Fits the complex transmission data from a resonator
    
    Fit is performed using modified resonator_tools library:
        https://github.com/ShabaniLab/resonator_tools
    Forked originally from:
        https://github.com/sebastianprobst/resonator_tools

    Parameters
    ----------
    frequency : np.ndarray
        The frequencies corresponding to the transmission value at that point
    complex_data : np.ndarray
        Complex transmission data
    gui_fit (= False) : boolean
        This variable sets a flag to open up the automatic fitting done through
        a matplotlib GUI
    
    Returns
    -------
    fitting_parameters : float | np.ndarray
        Parameters obtained from the fit
    """
    if len(frequency.shape) >= 2:
        original_shape = frequency.shape[:-1]
        trace_number = np.prod(original_shape)
        frequency = frequency.reshape((trace_number, -1))
        complex_data = complex_data.reshape((trace_number, -1))
        if powers is None:
            powers = np.array([None]*trace_number)
        else:
            powers = powers.reshape((trace_number, -1))
    else:
        frequency = np.array((frequency,))
        complex_data = np.array((complex_data,))
        powers = np.array((powers,))
    data_columns = [
        'Qi_dia_corr',        # 0
        'Qi_no_corr',         # 1
        'absQc',              # 2
        'Qc_dia_corr',        # 3
        'Ql',                 # 4
        'fr',                 # 5
        'theta0',             # 6
        'phi0',               # 7
        'phi0_err',           # 8
        'Ql_err',             # 9
        'absQc_err',          # 10
        'fr_err',             # 11
        'chi_square',         # 12
        'Qi_no_corr_err',     # 13
        'Qi_dia_corr_err',    # 14
        'prefactor_a',        # 15
        'prefactor_alpha',    # 16
        'baseline_slope',     # 17
        'baseline_intercept', # 18
        # delay               # 19
        # Power
        # Photon
    ]
    if return_fit_data:
        fit_data = []
    if powers is not None:
        fit_results = np.empty((frequency.shape[0],len(data_columns) + 2))
    else:
        fit_results = np.empty((frequency.shape[0],len(data_columns)))
    for i in range(frequency.shape[0]):
        try:
            centered_frequency, centered_complex_data = center_data(
                frequency[i],
                complex_data[i]
            )

            port = circuit.notch_port(
                f_data = centered_frequency,
                z_data_raw = centered_complex_data
            )
            if gui_fit:
                port.GUIfit(refine_results=True,sl_delay_margin=delay_range, live_result=live_result)
                f = plt.figure(figsize=(8,8))
                port.plotall(show = not save_gui_fits)
                results = port.fitresults
                if save_gui_fits:
                    if save_gui_fits_title is not None:
                        if isinstance(save_gui_fits_title,str):
                            f.suptitle(
                                (
                                    f'{save_gui_fits_title}' + 
                                    ' $Q_{ext}$ = ' +
                                    f'{results["Qc_dia_corr"]:.1e}' +
                                    ' $Q_{int}$ = ' +
                                    f'{results["Qi_dia_corr"]:.1e}'
                                ),
                                fontsize=20
                            )
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        else:
                            f.suptitle(
                                (
                                    f'{save_gui_fits_title[i]}' +
                                    ' $Q_{ext}$ = ' + 
                                    f'{results["Qc_dia_corr"]:.1e}' +
                                    ' $Q_{int}$ = ' + 
                                    f'{results["Qi_dia_corr"]:.1e}'
                                ),
                                fontsize=20
                            )
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    save_file_name = ''
                    if save_gui_fits_prefix is not None:
                        if isinstance(save_gui_fits_prefix,str):
                            save_file_name += save_gui_fits_prefix + "_"
                        else:
                            save_file_name += save_gui_fits_prefix[i] + "_"
                    if save_gui_fits_suffix is not None:
                        if isinstance(save_gui_fits_suffix,str):
                            save_file_name += save_gui_fits_suffix
                        else:
                            save_file_name += save_gui_fits_suffix[i]
                    else:
                        save_file_name += f"trace_{i}"
                    if not save_gui_fits_filetype.startswith('.'):
                        save_file_name += '.' + save_gui_fits_filetype
                    else:
                        save_file_name += save_gui_fits_filetype
                    plt.savefig(os.path.join(save_gui_fits_path,save_file_name))
                    plt.close(fig=f)
            else:
                if delay is None:
                    port.autofit(refine_results=True)
                else:
                    port.autofit(electric_delay=delay,refine_results=True)
                results = port.fitresults
            for j,column in enumerate(data_columns):
                fit_results[i,j] = results[column]
            fit_results[i,-3] = port._delay
            if powers[i] is not None:
                fit_results[i,-2] = powers[i]
                fit_results[i,-1] = port.get_photons_in_resonator(
                    powers[i],'dBm'
                )
            if return_fit_data:
                fit_data.append(port.z_data_sim)
        except ZeroDivisionError:
            print("WARNING ZeroDivisionError")
            for j,column in enumerate(data_columns):
                fit_results[i,j] = np.nan
        except TypeError:
            print("WARNING TypeError")
            for j,column in enumerate(data_columns):
                fit_results[i,j] = np.nan
    
    if return_fit_data:
        return fit_results, fit_data
    else:
        return fit_results

def notch_from_results(f,results):
    return circuit.notch_port()._S21_notch(f,
        fr=results[5],
        Ql=results[4],
        Qc=results[3],
        phi=results[7],
        a=1.,
        alpha=0.,
        delay=0)

