"""Some sample current density profiles, J(x), useful for testing.

Common parameters
-----------------
    ic0         :   zero-field critical current
    jj_width    :   Josephson junction width
"""
import numpy as np
import scipy


def j_uniform(x, ic0=1, jj_width=1):
    """Uniform current density with which to compare output."""
    return (
        ic0
        / jj_width
        * np.piecewise(x, [np.abs(x) < jj_width / 2, np.abs(x) >= jj_width / 2], [1, 0])
    )


def j_gaussian(x, ic0=1, jj_width=1):
    """Gaussian current density."""
    return ic0 * scipy.stats.norm.pdf(x, loc=0, scale=jj_width / 4)


def j_gennorm(x, ic0=1, jj_width=1):
    """Generalized normal distributed current density."""
    return ic0 * scipy.stats.gennorm.pdf(x, 8, loc=0, scale=jj_width / 2)


def j_multigate(x, distr, ic0=1, jj_width=1):
    """Five-gate multigate distribution.

    Parameters
    ----------
    distr : list or np.ndarray
        1D list or array, of length 5, describing the relative distribution of
        current density in the regions below each minigate. Will be
        automatically normalized.
    """
    gate_width = jj_width / 5
    left_edge = -jj_width / 2
    distr = np.asarray(distr) / np.sum(distr)
    return np.piecewise(
        x,
        [
            np.logical_and(x >= left_edge, x < left_edge + gate_width),
            np.logical_and(x >= left_edge + gate_width, x < left_edge + 2 * gate_width),
            np.logical_and(
                x >= left_edge + 2 * gate_width, x < left_edge + 3 * gate_width
            ),
            np.logical_and(
                x >= left_edge + 3 * gate_width, x < left_edge + 4 * gate_width
            ),
            np.logical_and(x >= left_edge + 4 * gate_width, x < jj_width / 2),
        ],
        distr * ic0 / gate_width,
    )
