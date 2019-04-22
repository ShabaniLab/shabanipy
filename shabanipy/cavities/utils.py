# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Utility function to analyse cavity resonances.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from lmfit.models import LinearModel


def extract_baseline(data, asymmetry, smoothing, niter=10, plot=False):
    """Extract the baseline of a dataset using asymetric smoothing.

    see "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
    http://stackoverflow.com/questions/29156532/python-baseline-correction-library

    Parameters
    ----------
    data : np.ndarray
        Real data from which a baseline should be extracted.

    asymmetry : float
        Asymmetry parameter which should chosen between 0 and 1. When we
        expect positive peaks use a value close to 0, for dips use a value
        close to 1. Use a value of 0.5 for a fully symmetric fit.

    smoothing : float
        Smoothing factor. This should be a large number (1e2, 1e6).

    niter : int, optional
        Number of iteration to perform to obtain the baseline.

    plot : bool, optional
        Plot the provided data and the fitted baseline for debugging purposes.

    Returns
    -------
    z : np.array
        Baseline values

    """
    l = len(data)
    d = sparse.diags([1,-2,1],[0,-1,-2], shape=(l,l-2))
    w = np.ones(l)
    for i in range(niter):
        W = sparse.spdiags(w, 0, l, l)
        Z = W + smoothing * d.dot(d.transpose())
        z = spsolve(Z, w*data)
        w = asymmetry * np.greater(data, z) + (1-asymmetry) * np.less(data, z)

    if plot:
        plt.figure()
        plt.plot(data, '+')
        plt.plot(z)

    return z


def _find_extrema_index(amplitude, kind='min'):
    """Find the position of an extrema

    """
    exarg = np.argmin if kind == 'min' else np.argmax
    return exarg(amplitude)


def estimate_central_frequency(frequency, amplitude, kind='min'):
    """Estimate the position of the central frequency.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency data.
    amplitude : np.ndarray
        Amplitude of the signal.
    kind : {'min', 'max'}
        Should we look for a minimum or a maximum in the data.

    Returns
    -------
    central_frequency : float
        Estimated central frequency of the resonance.

    """
    return frequency[_find_extrema_index(amplitude, kind)]


def estimate_peak_amplitude(amplitude, kind='min'):
    """Estimate the position of the central frequency.

    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude of the signal.
    kind : {'min', 'max'}
        Should we look for a minimum or a maximum in the data.

    Returns
    -------
    central_frequency : float
        Estimated central frequency of the resonance.

    """
    return np.abs(amplitude[_find_extrema_index(amplitude, kind)] -
                  amplitude[0])/amplitude[0]


def estimate_width(frequency, amplitude, kind='min'):
    """Estimate the width of the resonance.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency data.
    amplitude : np.ndarray
        Amplitude of the signal.
    kind : {'min', 'max'}
        Should we look for a minimum or a maximum in the data.

    Returns
    -------
    width : float
        Estimated width of the resonance.

    """
    # Determine the resonance contrast
    baseline = amplitude[0]
    center_index = _find_extrema_index(amplitude, kind)
    target = baseline + (amplitude[center_index] - baseline)/2

    # Find the half-width on the left side of the resonance
    left_index = np.argmin(np.abs(amplitude[:center_index] - target))
    left_freq = frequency[:center_index][left_index]

    # Find the half-width on the right side of the resonance
    right_index = np.argmin(np.abs(amplitude[center_index:] - target))
    right_freq = frequency[center_index:][right_index]

    return right_freq - left_freq


def estimate_time_delay(frequency, phase, npoints=50, plot=False):
    """Correct for the cable delay.

    Parameters
    ----------
    frequency : np.ndarray
        1D array of frequency data points.
    phase : np.ndarray
        1D array of phase data points.
    npoints : int | None, optional
        Number of points on which to perform the linear fit. If None all points
        are used.
    plot : bool, optional
        Should we plot the fitted linear dependence of the phase and the
        corrected phase.

    Returns
    ----------
    corrected_phase : np.ndarray
        Phase from which the cqble delay impact has been substracted.

    """
    phase = np.unwrap(phase)
    model_freq = frequency[:npoints] if npoints is not None else frequency
    model_phase = phase[:npoints] if npoints is not None else phase
    params = LinearModel().guess(model_phase, x=model_freq)
    res = LinearModel().fit(model_phase, params, x=model_freq)
    if plot:
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(frequency, phase, '+')
        axes[0].plot(model_freq, res.best_fit)
        axes[1].plot(frequency,
                     correct_for_time_delay(frequency, phase,
                                            res.best_values['slope']))
    return res.best_values['slope']


def correct_for_time_delay(frequency, phase, slope):
    """Correct the phase accumulation due to the time delay.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency at which the phase has been measured.
    phase : np.ndarray
        Phase of the measured signal.
    slope : float
        Phase slope vs frequency as extimated using estimate_time_delay

    """
    phase = np.unwrap(phase)
    phase -= frequency*slope
    phase -= phase[0]
    return ( phase + np.pi) % (2 * np.pi ) - np.pi

