"""Functions used to estimate SQUID model parameters."""
from typing import Tuple

import numpy as np


def estimate_frequency(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
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
    return max_freq, fftfreqs, abs_fft
