# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Functions relating the critical current density to the Fraunhofer pattern of a JJ.

The functions implemented here are based on:
    [1] Dynes, R. C. & Fulton, T. A.  Supercurrent Density Distribution in Josephson
    Junctions.  Phys. Rev. B 3, 3015–3023 (1971).
"""
import warnings
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad, romb
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import hilbert
from typing_extensions import Literal
from numba import cfunc
from numba.types import CPointer, float64, intc

from shabanipy.utils.integrate import can_romberg, resample_evenly


def fraunhofer(
    current_density,
    position,
    *,
    bfields,
    jj_length = None,
    return_fourier = False,
) -> np.ndarray:
    """Compute the Fraunhofer pattern of a given critical current denisty.

    The Fraunhofer pattern is the critical current interference pattern I_c(B).

    Parameters
    ----------
    current_density: ndarray
        Critical current density J(x) as a function of position along the junction width.
    position: ndarray
        Position x along the junction width corresponding to the current_density J(x).
    bfields: ndarray
        Out-of-plane magnetic field values at which to compute the Fraunhofer pattern.
    jj_length: float, optional
        Effective junction length, which determines the B-field-to-wavenumber conversion
        factor as 2*pi*jj_length/PHI0 where PHI0 is the (superconducting) magnetic flux
        quantum.  If left unspecified, the B-to-k conversion factor defaults to 1.
    return_fourier: bool, optional
        Whether to return the Fourier transform or the Fraunhofer pattern (useful for
        studying the phase of the Fourier transform).

    """
    fourier = np.empty_like(bfields, dtype=complex)
    if not can_romberg(position):
        position, current_density = resample_evenly(position, current_density)

    dx = abs(position[0] - position[1])
    b_to_k = 2 * np.pi * jj_length / PHI0 if jj_length else 1
    for i, b in enumerate(bfields):
        fourier[i] = romb(current_density * np.exp(1j * b_to_k * b * position), dx)

    return fourier if return_fourier else np.abs(fourier)


def _fraunhofer_dft(
    j: np.ndarray, dx: float = 1, b_to_k: float = 1, return_fourier: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Fraunhofer from current density using discrete Fourier transform."""
    j_fourier = fftshift(fft(j))
    b = 2 * np.pi / b_to_k * fftshift(fftfreq(len(j), dx))
    return (b, j_fourier) if return_fourier else (b, np.abs(j_fourier))
def fourier_phase(
    fields: np.ndarray,
    ics: np.ndarray,
    b_to_k: float,
    jj_width: float,
    method: Optional[Literal["romb", "quad", "hilbert"]] = "hilbert",
) -> np.ndarray:
    """Compute the phase as the Hilbert transform of ln(I_c).

    Note the integral expressions for θ(β) implemented here differ from Eq. (5)
    [1] by a factor of 2, but give the correct output. (The source of this
    discrepancy is as yet unknown. But note the Hilbert transform is usually
    defined, as below, with a prefactor 1/π.)

    Parameters
    ----------
    fields : np.ndarray
        Magnetic field at which the critical current was measured. For ND input
        the sweep should occur on the last axis.
    ics : np.ndarray
        Measured value of the critical current. For ND input the sweep should
        occur on the last axis.
    b_to_k : float
        Field-to-wavenumber conversion factor (i.e. β / B).
    jj_width: float
        Width of the junction transverse to the field and current.
    method: str, optional
        Method used to compute the phase; `romb` and `quad` specify
        numerical integration methods (note `quad` is particularly slow), while
        `hilbert` uses `scipy.signal.hilbert` to compute the discrete Hilbert
        transform.

    Returns
    -------
    np.ndarray
        Phase θ, the Hilbert transform of ln(I_c), to be used when rebuilding
        the current distribution. The phases are shifted by a factor
        `field * jj_width / 2` to give a reconstructed current density centered
        about the origin.

    """
    # scale B to beta first; then forget about it
    fields = fields * b_to_k

    if method == "romb":
        theta = _fourier_phase_romb(fields, ics)
    elif method == "quad":
        theta = _fourier_phase_quad(fields, ics)
    elif method == "hilbert":
        theta = _fourier_phase_hilbert(ics)
    else:
        raise ValueError(f"Method '{method}' unsupported")

    return theta - fields * jj_width / 2


def _fourier_phase_romb(fields: np.ndarray, ics: np.ndarray) -> np.ndarray:
    """Compute Eq. (5) of [1] using Romberg integration."""
    if not can_romberg(fields):
        fine_fields, fine_ics = resample_evenly(fields, ics)
    else:
        fine_fields, fine_ics = fields, ics
    step = abs(fine_fields[0] - fine_fields[1])

    theta = np.empty_like(fields)
    for i, (field, ic) in enumerate(zip(fields, ics)):
        # don't divide by zero
        denom = field ** 2 - fine_fields ** 2
        denom[denom == 0] = 1e-9

        theta[i] = field / np.pi * romb((np.log(fine_ics) - np.log(ic)) / denom, step)
    return theta


def _fourier_phase_quad(fields: np.ndarray, ics: np.ndarray) -> np.ndarray:
    """Compute Eq. (5) of [1] using scipy.integrate.quad."""
    ics_interp = interp1d(fields, ics, "cubic")

    def integrand(b, beta, ic):
        # quad will provide b when calling this
        return (np.log(ics_interp(b)) - np.log(ic)) / (beta ** 2 - b ** 2)

    theta = np.empty_like(fields)
    for i, (field, ic) in enumerate(zip(fields, ics)):
        theta[i] = (
            field
            / np.pi
            * quad(
                integrand,
                np.min(fields),
                np.max(fields),
                args=(field, ic),
                points=[field],
            )[0]
        )
    return theta


def _fourier_phase_hilbert(ics: np.ndarray) -> np.ndarray:
    """Compute Eq. (5) of [1] using a discrete Hilbert transform."""
    return hilbert(np.log(ics)).imag


def critical_current_density(
    fields: np.ndarray,
    ics: np.ndarray,
    b_to_k: float,
    jj_width: float,
    jj_points: int,
    debug: bool = False,
    theta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the critical current density J_c(x) from a Fraunhofer pattern I_c(B).

    Parameters
    ----------
    fields : np.ndarray
        1D array of the magnetic field at which the critical current was measured.
    ics : np.ndarray
        1D array of the measured value of the critical current.
    b_to_k : float
        Field to wave-vector conversion factor. This can be estimated from the
        Fraunhofer periodicity.
    jj_width : float
        Size of the junction. The current distribution will be reconstructed on
        a larger region (2 * jj_width)
    jj_points : int
        Number of points used to describe the junction inside jj_width.
    theta : np.ndarray, optional
        Phase distribution to use in the current reconstruction. If None, it
        will be extracted from the given Fraunhofer pattern.

    Returns
    -------
    np.ndarray
        Positions at which the current density was calculated.
    np.ndarray
        Current density.

    """
    if not can_romberg(fields):
        fine_fields, fine_ics = resample_evenly(fields, ics)
        if theta is not None:
            _, fine_theta = resample_evenly(fields, theta)
        else:
            fine_theta = theta
    else:
        fine_fields, fine_ics, fine_theta = fields, ics, theta

    fine_ics[np.less_equal(fine_ics, 1e-10)] = 1e-10

    if fine_theta is None:
        fine_theta = fourier_phase(fine_fields, fine_ics, b_to_k, jj_width)

    # scale from B to beta
    fine_fields = b_to_k * fine_fields
    step = abs(fine_fields[0] - fine_fields[1])

    xs = np.linspace(-jj_width, jj_width, int(2 * jj_points))
    j = np.empty(xs.shape, dtype=complex)
    for i, x in enumerate(xs):
        j[i] = (
            1
            / (2 * np.pi)
            * romb(fine_ics * np.exp(1j * (fine_theta - fine_fields * x)), step)
        )

    if debug:
        f, axes = plt.subplots(2, 1)
        axes[0].plot(b_to_k * fields, ics)
        axes[1].plot(xs, j.real)
        plt.show()

    return xs, j.real
