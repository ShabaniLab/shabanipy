# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tools to analyse cavity in the notch geometry.

"""
import numpy as np

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
