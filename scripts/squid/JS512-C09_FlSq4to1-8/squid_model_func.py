"""SQUID model function used to construct an lmfit Model."""
import numpy as np

from shabanipy.squid.cpr import finite_transparency_jj_current as cpr
from shabanipy.squid.squid_model import compute_squid_current


def squid_model_func(
    bfield: np.ndarray,
    transparency1: float,
    transparency2: float,
    switching_current1: float,
    switching_current2: float,
    bfield_offset: float,
    radians_per_tesla: float,
    anom_phase1: float,
    anom_phase2: float,
    temperature: float,
    gap: float,
    inductance: float,
    both_branches: bool = False,
):
    """The model function to fit against the data."""
    bfield = bfield - bfield_offset

    i_squid = [
        compute_squid_current(
            bfield * radians_per_tesla,
            cpr,
            (anom_phase1, switching_current1, transparency1),
            cpr,
            (anom_phase2, switching_current2, transparency2),
            positive=positive,
            inductance=inductance,
        )
        for positive in ((True, False) if both_branches else (True,))
    ]
    return np.array(i_squid).flatten()
