"""SQUID model function used to construct an lmfit Model."""
from typing import Tuple

import numpy as np

from shabanipy.jj import transparent_cpr as cpr
from shabanipy.squid import critical_behavior


# TODO this needs to be updated given squid code refactor
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
    positive: Tuple[bool],
):
    """The model function to fit against the data."""
    bfield = bfield - bfield_offset

    i_squid = [
        critical_behavior(
            bfield * radians_per_tesla,
            cpr,
            (anom_phase1, switching_current1, transparency1),
            cpr,
            (anom_phase2, switching_current2, transparency2),
            positive=p,
            inductance=inductance,
        )[1]
        for p in positive
    ]
    return np.array(i_squid).flatten()
