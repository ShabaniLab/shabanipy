"""Functions used to estimate SQUID model parameters."""
import warnings
from typing import Tuple

import numpy as np
from scipy.signal import find_peaks


def estimate_frequency(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Estimate the frequency of oscillations in y(x) by discrete Fourier transform.

    Parameters
    ----------
    x : 1d array
        The x-values for the data points (x[i], y[i]).
        The sample spacing is assumed uniform; a warning is issued otherwise.
    y : 1d array
        The y-values (assumed real) for the data points (x[i], y[i]).

    Returns
    -------
    freq : float
        The strongest nonzero frequency component.
    fftfreqs : np.ndarray
        The frequencies at which the Fourier transform of y(x) was computed.
    fft : np.ndarray
        The (complex) Fourier transform of y(x).
    """
    dxs = np.unique(np.diff(x))
    try:
        (dx,) = dxs
    except ValueError:
        dx = np.mean(dxs)
        if not np.allclose(dxs, dxs[0], atol=0):
            warnings.warn(
                "Samples are not uniformly spaced in the domain; "
                "frequency estimation might be poor"
            )
    fft = np.fft.rfft(y)
    abs_fft = np.abs(fft)
    fftfreqs = np.fft.rfftfreq(len(y), d=dx)
    max_freq = fftfreqs[np.argmax(abs_fft[1:]) + 1]  # ignore dc component
    return max_freq, (fftfreqs, fft)


def estimate_boffset(
    bfield: np.ndarray, ic_p: np.ndarray, ic_n: np.ndarray = None
) -> Tuple[float, Tuple[np.ndarray, ...]]:
    """Estimate the coil field at which the true flux is integral.

    Parameters
    ----------
    bfield
        Magnetic field values.
    ic_p : optional
        Positive branch of the SQUID critical current.
    ic_n : optional
        Negative branch of the SQUID critical current.

    Returns
    -------
    boffset
        The position of a peak in `ic_p` or a valley in `ic_n` near the center of the
        `bfield` range.  If both `ic_p` and `ic_n` are given, the point midway between
        the peak in `ic_p` and the valley in `ic_n` is returned.
    (peak_idxs, valley_idxs)
        Indices of the peaks (valleys) in `ic_p` (`ic_n`).  If either branch was not
        given, the corresponding value will be None.
    """
    # neither branch given
    assert (
        ic_p is not None or ic_n is not None
    ), "A positive or negative critical current branch (or both) must be provided."
    # one branch or the other given
    if ic_p is not None:
        peak_idxs, _ = find_peaks(ic_p, prominence=(np.max(ic_p) - np.min(ic_p)) / 2)
        boffset_p = bfield[peak_idxs[len(peak_idxs) // 2]]
        if ic_n is None:
            return boffset_p, (peak_idxs, None)
    if ic_n is not None:
        valley_idxs, _ = find_peaks(-ic_n, prominence=(np.max(ic_n) - np.min(ic_n)) / 2)
        boffset_n = bfield[valley_idxs[len(valley_idxs) // 2]]
        if ic_p is None:
            return boffset_n, (None, valley_idxs)
    # both branches given
    return (boffset_p + boffset_n) / 2, (peak_idxs, valley_idxs)
