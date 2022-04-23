# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Functions used to model SQUID behavior."""

from typing import Callable, Iterable, Literal, Tuple, Union

import numpy as np


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
) -> Tuple[Union[float, np.ndarray]]:
    """Compute the critical behavior of a dc SQUID as a function of total phase.

    This computes the phases of the junctions along the positive or negative branch of
    the SQUID critical current, as a function of the SQUID phase 2πΦ/Φ0.  To compute the
    behavior as a function of the externally applied phase, see
    `critical_control`.

    Sign conventions follow those of Tinkham Fig. 6.8 with positive B-field out of the
    page:
                                       I1(γ1)
                                        --> 
                                      ---X---
                                  I  |       |
                                -->--|  B ⊙  |-->--
                                     |       |
                                      ---X---
                                        --> 
                                       I2(γ2)

    The junction phases γ1, γ2 are fully constrained by "flux quantization" (or more
    accurately single-valuedness of the order parameter's phase),

                                γ1 - γ2 = 2πΦ/Φ0                                (1)

    and supercurrent maximization,

                       Ic = max_{γ1,γ2} [ I1(γ1) + I2(γ2) ]                     (2)

    The flux Φ threading the loop in (1) comprises the externally applied flux Φ_ext and
    the self-induced flux arising from the current flowing in the loop of inductance
    L > 0:

                      Φ = Φ_ext + (L/2) [ I2(γ2) - I1(γ1) ]                     (3)

    See Tinkham ed. 2 §6.4.1 and §6.5 for details.

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
    squid_ic
        SQUID critical current Ic.
    phase_ext
        Phase 2π(Φ_ext/Φ0) due to externally applied flux.
    current1
        Supercurrent I1 through junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    phase1
        Phase γ1 across junction 1 at the SQUID critical current.
        Only returned if `return_jjs=True`.
    current2
        Same as `current1` but for junction 2.  Only returned if `return_jjs=True`.
    phase2
        Same as `phase1` but for junction 2.  Only returned if `return_jjs=True`.
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
            squid_ic,
            phase_ext,
            current1_opt,
            phase1[idxopt],
            current2_opt,
            phase2[idxopt],
        ]
    else:
        output = [squid_ic, phase_ext]
    return [a.squeeze() for a in output]
