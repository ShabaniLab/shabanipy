# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Analysis of differential resistance and IV curves of superconducting devices."""
from typing import Literal, Optional, Tuple

import numpy as np


def extract_switching_current(
    bias: np.ndarray,
    dvdi: np.ndarray,
    *,
    side: Literal["positive", "negative", "both"] = "positive",
    threshold: Optional[float] = None,
    global_threshold: bool = False,
    interp: bool = False,
    offset: float = 0,
    offset_npoints: Optional[int] = None,
    ignore_npoints: Optional[int] = None,
) -> np.ndarray:
    """Extract the switching currents from a set of differential resistance curves.

    This function will also work for V(I) curves if `offset` and an explicit `threshold`
    are given.

    Parameters
    ----------
    bias
        N-dimensional array of bias current, assumed to be swept along the last axis.
    dvdi
        N-dimensional array of differential resistance at the corresponding `bias`
        values; same shape as `bias`.
    side : optional
        Which branch of the switching current to extract (positive, negative, or both).
    threshold : optional
        The switching current is determined as the first `bias` value for which `dvdi`
        rises above `threshold`.
        If None, the threshold is inferred as half the rise from the minimum to the
        maximum `dvdi` value in the direction of `side`.  A different threshold is
        computed for each bias sweep in each direction, unless `global_threshold=True`.
    global_threshold : optional
        If true when `threshold` is None, a single threshold is computed for all bias
        sweeps as half the difference from the global minimum to the global maxmium of
        `dvdi`.
    interp : optional
        If true, linearly interpolate `dvdi` vs `bias` to more accurately detect the
        switching current.
    offset : optional
        A constant value to subtract from `dvdi`.  Overrides `offset_npoints`.
    offset_npoints : optional
        The number of points around 0 bias that are averaged to compute the offset.
        Ignored if nonzero `offset` is given.
    ignore_npoints : optional
        The number of points around 0 bias that are ignored, e.g. if lock-in wasn't
        settled when starting a sweep from zero bias.
        This option currently assumes the bias is swept symmetrically about zero.

    Returns
    -------
    ic: ndarray
        The positive or negative branch of the switching current.  If `side` is "both",
        ic[0] = ic- and ic[1] = ic+.  The returned arrays have the same shape as the
        input arrays without the last axis.  `np.nan` is returned where no switching
        current could be found.
    """
    if side not in ("positive", "negative", "both"):
        raise ValueError("`side` should be one of: 'positive', 'negative', 'both'")

    if ignore_npoints:
        bias, dvdi = _ignore_points(bias, dvdi, ignore_npoints)

    if offset:
        pass
    elif offset_npoints:
        offset = _compute_offset(bias, dvdi, offset_npoints)
    dvdi -= offset

    if threshold is None and global_threshold:
        threshold = (dvdi.max() - dvdi.min()) / 2

    if side != "negative":
        ic_p = find_rising_edge(
            bias,
            np.where(bias >= 0, np.abs(dvdi), np.nan),
            threshold=threshold,
            interp=interp,
        )
    if side != "positive":
        ic_n = find_rising_edge(
            bias[..., ::-1],
            # np.abs is used to support V(I) curves as well as dV/dI
            np.where(bias <= 0, np.abs(dvdi), np.nan)[..., ::-1],
            threshold=threshold,
            interp=interp,
        )

    return (
        ic_p
        if side == "positive"
        else ic_n
        if side == "negative"
        else np.array((ic_n, ic_p))
    )


def find_rising_edge(x, y, *, threshold=None, interp=False):
    """Find the first `x` where `y` exceeds `threshold` (along the last axis).

    `x` and `y` must have the same shape.

    If `threshold` is None, it will be inferred as half the rise from min(y) to max(y).

    If `interp` is True, linearly interpolate between the two points below and above
    `threshold` to get a more accurate value of `x` at the crossing.

    If all values of `y` are below or above the threshold (i.e. there is no crossing),
    return `np.nan`.
    """
    if threshold is None:
        threshold = ((np.nanmin(y, axis=-1) + np.nanmax(y, axis=-1)) / 2)[
            ..., np.newaxis
        ]
    index = np.argmax(y > threshold, axis=-1)
    x1 = np.take_along_axis(x, index[..., np.newaxis], axis=-1).squeeze(axis=-1)
    if interp:
        x0 = np.take_along_axis(x, index[..., np.newaxis] - 1, axis=-1).squeeze(axis=-1)
        y0 = np.take_along_axis(y, index[..., np.newaxis] - 1, axis=-1).squeeze(axis=-1)
        y1 = np.take_along_axis(y, index[..., np.newaxis], axis=-1).squeeze(axis=-1)
        dydx = (y1 - y0) / (x1 - x0)
        rising_edge = x0 + (np.squeeze(threshold) - y0) / dydx
    else:
        rising_edge = x1
    rising_edge = np.atleast_1d(rising_edge)
    rising_edge[..., index == 0] = np.nan  # all values were above or below threshold
    return rising_edge.squeeze()


def extract_iexrn(
    bias: np.ndarray,
    volt: np.ndarray,
    bias_min: float,
    *,
    side: Literal["positive", "negative", "both"] = "positive",
    offset: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract excess current and normal resistance from a V(I) curve.

    Parameters
    ----------
    bias
        N-dimensional array of d.c. bias current, assumed to be swept along the last
        axis.
    volt
        N-dimensional array of d.c. voltage measured at the corresponding `bias` values;
        same shape as `bias`.
    bias_min
        Current bias threshold above which the V(I) curve is considered ohmic/linear.
        The linear fit is limited to where |bias| > bias_min.
    side : optional
        Which side of the V(I) curve to analyze (positive, negative, or both).

    Returns
    -------
    (iex, rn)
        The excess current and normal resistance obtained from the positive or negative
        side of the V(I) curve. If `side` is "both", iex[0] = iex- and iex[1] = iex+
        (likewise for rn).  The returned arrays have the same shape as the input arrays
        without the last axis.
    """
    if side == "both":
        dim0 = (2,)
    else:
        dim0 = ()
    iex = np.empty(dim0 + bias.shape[:-1])
    rn = np.empty(dim0 + bias.shape[:-1])
    iex[:] = rn[:] = np.nan

    it = np.nditer(bias[..., 0], flags=["multi_index"])
    for _ in it:
        index = it.multi_index
        i, v = bias[index], volt[index]
        if side != "negative":
            mask = i >= bias_min
            iex_p, rn_p = _fit_extrap(i[mask], v[mask])
        if side != "positive":
            mask = i <= -bias_min
            iex_n, rn_n = _fit_extrap(i[mask], v[mask])

        if side == "negative":
            iex[index] = iex_n
            rn[index] = rn_n
        elif side == "positive":
            iex[index] = iex_p
            rn[index] = rn_p
        else:  # side == "both"
            iex[:, index] = np.array([[iex_n, iex_p]]).T
            rn[:, index] = np.array([[rn_n, rn_p]]).T

    return iex.squeeze(), rn.squeeze()


def _fit_extrap(x: np.ndarray, y: np.ndarray) -> (float, float):
    """Fit a line and extrapolate to y=0.

    Inputs must be 1d.  Returns (x_intercept, slope), i.e. (excess_current,
    normal_resistance) if x is current bias and y is d.c. voltage.
    """
    poly = np.polynomial.Polynomial.fit(x, y, 1)
    y_int, slope = poly.convert().coef
    x_int = -y_int / slope
    return x_int, slope


def _compute_offset(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Compute the average of the n points in y closest to x=0.

    Points along the last axis in x are assumed to be sorted.  The point closest to x=0
    and the n//2 points on either side are used in the average.  Hence n+1 points are
    actually used if n is even, and the points are "closest to x=0" in terms of index
    distance.

    If there are less than n//2 points on either side of x=0, the remaining points are
    taken from the other side.

    If x and y are N-dimensional (N > 1), the offset is computed for each slice along
    the last axis.  The returned array has the same shape as x and y except the
    dimension of the last axis is collapsed to 1.
    """
    x0_index = np.atleast_1d(np.argmin(np.abs(x), axis=-1))
    indexes_to_average = np.concatenate(
        [
            np.arange(i0 - n // 2, i0 + n // 2 + 1)
            - min(i0 - n // 2, 0)
            - max(i0 + n // 2 + 1 - x.shape[-1], 0)
            for i0 in x0_index.flatten()
        ]
    ).reshape(y.shape[:-1] + (-1,))
    yvalues_to_average = np.take_along_axis(y, indexes_to_average, axis=-1)
    return np.mean(yvalues_to_average, axis=-1, keepdims=True)


def _ignore_points(
    x: np.ndarray, y: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove n points at the center of x and y.

    If x and y are N-dimensional (N > 1), the n points at the center w.r.t. the last
    axis are removed.
    """
    indexes = np.arange(n) + (x.shape[-1] - n) // 2
    return tuple(np.delete(a, indexes, axis=-1) for a in (x, y))
