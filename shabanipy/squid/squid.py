# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Functions used to model SQUID behavior."""

from typing import Callable, Iterable, List, Literal, Union
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d


def critical_behavior(
    phase: Union[float, np.ndarray],
    cpr1: Callable,
    params1: Iterable,
    cpr2: Callable,
    params2: Iterable,
    inductance: float = 0,
    branch: Literal["+", "-"] = "+",
    *,
    nbrute: int = 101,
    return_jjs: bool = False,
) -> List[np.ndarray]:
    """Compute the critical behavior of a dc SQUID as a function of total phase.

    This computes the phases of the junctions along the positive or negative branch of
    the SQUID critical current, as a function of the SQUID phase 2πΦ/Φ0.  To compute the
    behavior as a function of the externally applied phase, see `critical_control`.

    Sign conventions follow those of Tinkham Fig. 6.8 with positive B-field out of the
    page:
                                       I₁(γ₁)
                                        -->
                                      ---X---
                                  I  |       |
                                -->--|  B ⊙  |-->--
                                     |       |
                                      ---X---
                                        -->
                                       I₂(γ₂)

    The junction phases γ₁, γ₂ are fully constrained by two equations.  The first
    requires that the phase of the Ginzburg-Landau order parameter is single-valued and
    is often called the "flux quantization" condition (though the flux threading the
    loop is not quantized in general):

                                γ₁ - γ₂ = 2πΦ/Φ0                                (1)

    The second defines the SQUID critical current which is the maximum supercurrent the
    SQUID can pass for a given Φ:

                       Ic = max_{γ₁,γ₂} [ I₁(γ₁) + I₂(γ₂) ]                     (2)

    The flux Φ threading the loop in (1) comprises the externally applied flux Φ_ext and
    the self-induced flux arising from the current flowing in the loop of inductance
    L > 0:

                      Φ = Φ_ext + (L/2) [ I₂(γ₂) - I₁(γ₁) ]                     (3)

    where we assume the inductive contributions from each arm of the SQUID are
    identical.

    See Tinkham ed. 2 §6.4.1 and §6.5 for details, specifically Eq. (6.49).

    Parameters
    ----------
    phase
        Phase 2πΦ/Φ0 due to total flux threading the SQUID loop.  At most 1d.
    cpr1, cpr2
        Functions I(γ) used to compute the supercurrent in the junctions given their
        phases as the first argument.
    params1, params2
        Parameters (γ0, ...) passed to `cpr`s.  The first argument is interpreted as an
        anomalous phase shift such that γ -> γ - γ0.
    inductance
        Inductance of the SQUID loop, in units of Φ0/A where A is the unit of current
        returned by `cpr`s.  E.g. if A is microamperes, inductance=1 corresponds to 2nH.
        Must be nonnegative.
    branch
        Which branch of the SQUID critical current to compute.
    nbrute
        Number of points to use in the brute-force optimization of the SQUID current.
    return_jjs
        Return the phase and supercurrent of each junction as well.

    Return values all correspond to the given `phase` values.

    Returns
    -------
    phase_ext
        Phase 2π(Φ_ext/Φ0) due to externally applied flux.
    squid_ic
        SQUID critical current Ic(Φ).
    phase1
        Phase γ₁(Φ) across junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    current1
        Supercurrent I₁(Φ) through junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    phase2
        Same as `phase1` but for junction 2.  Only returned if `return_jjs=True`.
    current2
        Same as `current1` but for junction 2.  Only returned if `return_jjs=True`.
    """
    phase = np.atleast_1d(phase)
    phase1_offset, *params1 = params1
    phase2_offset, *params2 = params2

    # use (1) to eliminate γ1 in (2) and extremize over γ2 for each Φ
    phase2 = np.linspace(0, 2 * np.pi, nbrute) - phase2_offset
    phase2 = np.tile(phase2, (len(phase), 1)).T
    phase1 = phase2 + phase - phase1_offset
    current1 = cpr1(phase1, *params1)
    current2 = cpr2(phase2, *params2)
    squid_current = current1 + current2
    argopt = np.argmax if branch == "+" else np.argmin
    idxopt = (argopt(squid_current, axis=0), np.arange(len(phase)))
    squid_ic = squid_current[idxopt]

    # use (3) to determine Φ_ext
    current1_opt = current1[idxopt]
    current2_opt = current2[idxopt]
    phase_ext = phase - inductance * (current2_opt - current1_opt) * np.pi

    if return_jjs:
        output = [
            phase_ext,
            squid_ic,
            phase1[idxopt],
            current1_opt,
            phase2[idxopt],
            current2_opt,
        ]
    else:
        output = [phase_ext, squid_ic]
    return [a.squeeze() for a in output]


def critical_control(
    phase_ext: Union[float, np.ndarray],
    *args,
    ninterp: int = 101,
    **kwargs,
) -> np.ndarray:
    """Compute the critical behavior of a dc SQUID as a function of applied phase.

    This computes the phases of the junctions along the positive or negative branch of
    the SQUID critical current, as a function of the externally applied phase
    2π(Φ_ext/Φ0).  To compute the behavior as a function of the total phase, see
    `critical_behavior`.

    The behavior is first obtained as a function of total phase 2πΦ/Φ0 and then
    interpolated to find the values corresponding to `phase_ext`.

    For inductance L = 0, use `critical_behavior` directly as Φ_ext = Φ in this case.

    Parameters
    ----------
    phase_ext
        Phase 2π(Φ_ext/Φ0) due to externally applied flux.
    *args
        Positional arguments of `critical_behavior` (excluding `phase`).
    ninterp
        Number of points in total phase Φ ~ [0, 2π] used to interpolate the SQUID
        behavior as a function of Φ_ext(Φ).
    **kwargs
        Keyword arguments of `critical_behavior`.

    Return values all correspond to the given `phase_ext` values.

    Returns
    -------
    phase
        Phase 2π(Φ/Φ0) due to total flux.
    squid_ic
        SQUID critical current Ic(Φ_ext).
    phase1
        Phase γ₁(Φ_ext) across junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    current1
        Supercurrent I₁(Φ_ext) through junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    phase2
        Same as `phase1` but for junction 2.  Only returned if `return_jjs=True`.
    current2
        Same as `current1` but for junction 2.  Only returned if `return_jjs=True`.
    """
    # Given *discrete* values of Φ spanning [0, 2π], the interpolation range
    # Φ_ext(Φ) (mod 2π) will in general not perfectly span [0, 2π],
    # so we slightly expand the range to Φ ~ [-δ, 2π + δ].
    delta = 2 * np.pi / (ninterp - 1)
    phase = np.linspace(-delta, 2 * np.pi + delta, ninterp + 2)
    behavior = critical_behavior(phase, *args, **kwargs)
    p_ext = behavior[0]  # domain Φ_ext(Φ) over which to interpolate
    behavior[0] = phase
    if np.any(np.diff(p_ext) < 0):
        warn("Φ_ext(Φ) is not monotonically increasing.")
    # The resulting Φ_ext(Φ) spans a range of size 2π + ε.
    # Map the requested domain (spanning at most 2π) into this interpolation range.
    interp_min = np.min(p_ext)
    quotient = interp_min // (2 * np.pi)
    phase_ext = phase_ext % (2 * np.pi) + 2 * np.pi * quotient
    phase_ext = np.where(phase_ext < interp_min, phase_ext + 2 * np.pi, phase_ext)
    # phases need to be unwrapped to avoid interpolating through discontinuities
    for idx in [0, 2, 4]:
        behavior[idx] = np.unwrap(behavior[idx], period=2 * np.pi)
    return interp1d(p_ext, behavior, axis=-1, kind="cubic", copy=False)(phase_ext)
