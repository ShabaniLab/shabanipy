"""Resistively and capacitively shunted junction (RCSJ) model.

For details, see Tinkham §6.3.

All units are SI unless explicitly stated otherwise.

The following notation is used:
    Ic  critical_current
    R   resistance
    C   capacitance
"""
import numpy as np
from scipy.constants import e, hbar


def plasma_frequency(Ic, C):
    return np.sqrt(2 * e * Ic / (hbar * C))


def quality_factor(Ic, R, C):
    """Compute the quality factor of an RCSJ.

    The quality factor distinguishes overdamped (Q < 1) from underdamped (Q > 1) junctions.
    """
    return plasma_frequency(Ic=Ic, C=C) * R * C


def retrapping_current(Ic, R, C):
    """Estimate the retrapping current of an underdamped (hysteretic) RCSJ."""
    return 4 * Ic / (np.pi * quality_factor(Ic=Ic, R=R, C=C))
