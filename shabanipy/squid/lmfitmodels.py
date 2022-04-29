"""SQUID model functions to fit against experimental data.

These functions are used to construct an lmfit.Model."""
from typing import Literal

import numpy as np
from scipy.constants import elementary_charge as e
from scipy.constants import physical_constants

from shabanipy.jj import transparent_cpr as tcpr

from .squid import critical_control

PHI0 = physical_constants["mag. flux quantum"][0]


def squid_model(
    bfield: np.ndarray,
    bfield_offset: float,
    radians_per_tesla: float,
    anomalous_phase1: float,
    anomalous_phase2: float,
    critical_current1: float,
    critical_current2: float,
    transparency1: float,
    transparency2: float,
    inductance: float,
    temperature: float,
    gap: float,
    branch: Literal["+", "-", "+-"],
    nbrute: int = 101,
    ninterp: int = 101,
):
    """Model of the critical current of a dc SQUID with transparent junctions.

    Parameters
    ----------
    bfield
        External magnetic field (T) at which the critical current was measured.
    bfield_offset
        Magnetic field offset (T) due to hysteresis in the vector magnet coils.
    radians_per_tesla
        Externally applied phase per unit field, 2πA/Φ0, where A is the loop area.
    anomalous_phase1, anomalous_phase2
        Anomalous shifts in the junctions' phases (rad).
    critical_current1, critical_current2
        Junctions' critical currents (A).
    transparency1, transparency2
        Junctions' transparencies, on the interval [0, 1).
    inductance
        SQUID loop inductance (H).
    temperature
        Mixing chamber temperature (K), or electron temperature if known.
    gap
        Superconducting gap Δ (eV).
    branch
        Which branch of the SQUID critical current to model: positive (+), negative (-),
        or both (+-).
    nbrute
        Number of points to use in the brute-force optimization of the SQUID current.
    ninterp
        Number of points in total phase Φ ~ [0, 2π] used to interpolate the SQUID
        behavior as a function of Φ_ext(Φ).

    Returns
    -------
    The SQUID critical current for the requested `branch`es.
    """
    phase_ext = (bfield - bfield_offset) * radians_per_tesla
    squid_ic = [
        critical_control(
            phase_ext,
            tcpr,
            (anomalous_phase1, critical_current1, transparency1, temperature, gap * e),
            tcpr,
            (anomalous_phase2, critical_current2, transparency2, temperature, gap * e),
            inductance=inductance / PHI0,
            branch=b,
            nbrute=nbrute,
            ninterp=ninterp,
        )[1]
        for b in branch
    ]
    return np.squeeze(squid_ic)
