"""Functions used to estimate SQUID model parameters."""
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def estimate_frequency(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Estimate the frequency of oscillations in y(x) by discrete Fourier transform.

    Parameters
    ----------
    x : shape (N,)
        The domain values at which `y` was sampled.
        The sample spacing is assumed uniform; a warning is issued otherwise.
    y : shape (..., N)
        Samples of the waveform y(x).  Analysis is performed on the last axis.

    Returns
    -------
    freq
        The strongest nonzero frequency in y(x).
    fftfreqs
        The frequencies at which the Fourier transform of y(x) was computed.
    fft
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
    fftfreqs = np.fft.rfftfreq(len(x), d=dx)
    fft = np.fft.rfft(y, axis=-1)
    idx = np.argmax(np.abs(fft)[..., 1:], axis=-1) + 1  # ignore dc component
    freq = fftfreqs[idx]
    return freq, (fftfreqs, fft)


def estimate_bfield_offset(
    bfield: np.ndarray,
    ic_p: Optional[np.ndarray] = None,
    ic_n: Optional[np.ndarray] = None,
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
    bfield_offset
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
    ), "At least one critical current branch must be provided."
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


def estimate_critical_current(ic: np.ndarray, smaller_ic_jj: Optional[bool] = None):
    """Estimate the critical current of a junction in the SQUID.

    Parameters
    ----------
    ic
        SQUID critical current.  Both positive and negative branches may be passed in a
        shape (2, _) array.
    smaller_ic_jj
        If True (False), estimate the critical current of the junction with the smaller
        (larger) Ic.  If None, assume junction Ic's are roughly equal.

    Returns
    -------
    The estimated critical current of the junction specified by `smaller_ic_jj`.
    """
    if smaller_ic_jj:
        return np.mean(np.abs(np.max(ic, axis=-1) - np.min(ic, axis=-1)) / 2)
    elif smaller_ic_jj is not None:
        return np.mean(np.abs(ic))
    else:
        return np.mean(np.max(np.abs(ic), axis=-1)) / 2
