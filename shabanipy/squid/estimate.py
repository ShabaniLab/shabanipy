"""Functions used to estimate SQUID model parameters."""
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
    abs_fft : np.ndarray
        The magnitude of the Fourier transform of y(x).
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
    abs_fft = np.abs(np.fft.rfft(y))
    fftfreqs = np.fft.fftfreq(len(y), d=dx)[: len(x) // 2 + 1]
    max_freq = fftfreqs[np.argmax(abs_fft[1:]) + 1]  # ignore dc component
    return max_freq, (fftfreqs, abs_fft)


def estimate_boffset(
    bfield: np.ndarray, ic_p: np.ndarray, ic_n: np.ndarray = None
) -> Tuple[float, Tuple[np.ndarray, ...]]:
    """Estimate the coil field at which the true flux is integral.

    Parameters
    ----------
    bfield
        Magnetic field values.
    ic_p
        Positive branch of the SQUID critical current.
    ic_n : optional
        Negative branch of the SQUID critical current.

    Returns
    -------
    boffset
        The position of a peak near the center of the `bfield` range.
        If `ic_n` is given, the point midway between the peak in `ic_p` and the valley in
        `ic_n` near the center is returned.
    (peak_idxs,) or (peak_idxs, peak_idxs_n)
        Indices of the peaks in `ic_p` (and of the valleys in `ic_n` if given).
    """
    peak_idxs, _ = find_peaks(ic_p, prominence=(np.max(ic_p) - np.min(ic_p)) / 2)
    boffset = bfield[peak_idxs[len(peak_idxs) // 2]]
    if ic_n is not None:
        peak_idxs_n, _ = find_peaks(-ic_n, prominence=(np.max(ic_n) - np.min(ic_n)) / 2)
        boffset_n = bfield[peak_idxs_n[len(peak_idxs_n) // 2]]
        boffset = (boffset + boffset_n) / 2
        return boffset, (peak_idxs, peak_idxs_n)
    return boffset, (peak_idxs,)
