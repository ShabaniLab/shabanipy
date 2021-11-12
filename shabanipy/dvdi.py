# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Analysis of differential resistance curves of superconducting devices."""
from typing import Literal, Tuple, Union

import numpy as np


def extract_switching_current(
    bias: np.ndarray,
    dvdi: np.ndarray,
    *,
    side: Literal["positive", "negative", "both"] = "positive",
    threshold: float
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Extract the switching currents from a set of differential resistance curves.

    Parameters
    ----------
    bias
        N-dimensional array of bias current, assumed to be swept along the last axis.
    dvdi
        N-dimensional array of differential resistance at the corresponding `bias`
        values; same shape as `bias`.
    side : optional
        Which branch of the switching current to extract (positive, negative, or both).
        If "both", a tuple of (negative, positive) switching currents is returned.
    threshold
        The switching current is determined as the first `bias` value for which `dvdi`
        rises above `threshold`.

    Returns
    -------
    ndarray or (ndarray, ndarray)
        The positive or negative branch of the switching current, or (negative,
        positive) if `side` is "both".  The returned array(s) have the same shape as the
        input arrays without the last axis.
    """
    if side not in ("positive", "negative", "both"):
        raise ValueError("`side` should be one of: 'positive', 'negative', 'both'")

    if side != "negative":
        ic_p = _find_rising_edge(bias, np.where(bias >= 0, dvdi, np.nan), threshold)
    if side != "positive":
        ic_n = _find_rising_edge(
            bias[..., ::-1], np.where(bias <= 0, dvdi, np.nan)[..., ::-1], threshold,
        )

    return ic_p if side == "positive" else ic_n if side == "negative" else (ic_n, ic_p)


def _find_rising_edge(bias, dvdi, threshold):
    """Find the first `bias` where `dvdi` exceeds `threshold` (along the last axis)."""
    index = np.argmax(dvdi > threshold, axis=-1)
    return np.take_along_axis(bias, np.expand_dims(index, axis=-1), axis=-1).squeeze(
        axis=-1
    )
