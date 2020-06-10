# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Reconstruction routines based on Bayesian sampling.

"""
from math import prod
from typing import List, Union

import matplotlib.pyplot as plt
from dynesty import DynamicNestedSampler, NestedSampler, utils
from typing_extensions import Literal

from .generate_pattern import produce_fraunhofer_fast


def rebuild_current_distribution(
    fields: np.ndarray,
    ics: np.ndarray,
    jj_size: float,
    current_pattern: List[Union[Literal["f"], str]],
    sweep_invariants: List[Union[Literal["offset"], Literal["field_to_k"]]] = [
        "offset",
        "field_to_k",
    ],
    precision: float = 100,
    n_points: int = 2 ** 10 + 1,
) -> dict:
    """Rebuild a current distribution from a Fraunhofer pattern.

    This assumes a uniform field focusing since allowing a non uniform focusing
    would lead to a much larger space to explore.

    Parameters
    ----------
    fields : np.ndarray
        Out of plane field for which the critical current was measured.
    ics : np.ndarray
        Critical current of the junction.
    jj_size : float
        Size of the junction.
    current_pattern : List[Union[Literal["f"], str]]
        Describe in how many pieces to use to represent the junction. If the
        input arrays are more than 1D, "f" means that value is the same across
        all outer dimension, "v" means that the slice takes different value
        for all outer dimension (ie. one value per sweep).
    sweep_invariants : Tuple[Union[Literal["offset", "field_to_k"]]]
        Indicate what quantities are invariants across sweep for more the 1D
        inputs.
    precision : float, optional
        pass
    n_points : int, optional

    Returns
    -------
    dict


    """
    # Get the offset and estimated amplitude used in the prior
    # We do not use the estimated current and phase distribution to give the
    # more space to the algorithm.
    offsets, first_node_locs, _, _, _ = guess_current_distribution(
        field, fraunhofer, site_number, jj_size
    )
    # Gives a Fraunhofer pattern at the first node for v[1] = 1
    field_to_ks = 2 * np.pi / jj_size / np.abs(first_node_locs - offsets)

    # Determine the dimensionality of the problem based on the invariants and
    # the shape of the inputs.
    if len(sweep_invariants) > 2:
        raise ValueError("There are at most 2 invariants.")
    if any(k for k in sweep_invariants if k not in ("offset", "field_to_k")):
        raise ValueError(
            f"Invalid invariant specified {sweep_invariants}, "
            "valid values are 'offset', 'field_to_k'."
        )

    shape = fields.shape[:-1]
    shape_product = prod(shape) if shape else 0

    if shape_product == 0 and any(p.startswith("v") for p in current_pattern):
        raise ValueError(
            "Found variable current in the distribution but the measurements are 1D."
        )

    dim = len(sweep_invariants) + current_pattern.count("f")
    dim += shape_product * (current_pattern.count("v") + 2 - len(sweep_invariants))

    # Pre-compute slices to access elements in the prior and log-like
    offset_access = slice(
        0, 1 if "offset" in sweep_invariants else (shape_product or 1)
    )
    field_to_k_access = slice(
        offset_access.stop,
        offset_access.stop + 1
        if "field_to_k" in sweep_invariants
        else (shape_product or 1),
    )

    stop = field_to_k_access.stop
    current_density_accesses = []
    for p in current_pattern:
        if p == "f":
            current_density_accesses.append(slice(stop, stop + 1))
            stop += 1
        elif p == "v":
            current_density_accesses.append(slice(stop, stop + (shape_product or 1)))
            stop += current_density_accesses[-1].stop
        else:
            raise ValueError(
                f"Valid values in current_pattern are 'f' and 'v', found '{p}'"
            )

    def prior(u):
        """Map the sampled in 0-1 to the relevant values range.

        For all values we consider the values in the prior to be the log of the
        values we are looking for.

        """
        v = np.empty_like(u)
        v[offset_access] = 4 * u[offset_access] - 2
        v[field_to_k_access] = 4 * u[field_to_k_access] - 2
        stop += step

        # For all the amplitude we map the value between 0 and -X since the
        # amplitude of a single segment cannot be larger than the total current
        # X is determined based on the number of segments
        ampl = -np.log10(len(current_pattern))
        for sl in current_density_accesses:
            v[sl] = u[sl] * ampl

        return v

    def loglike(v):
        """Compute the distance to the data"""

        # We turn invariant input into their variant form (from 1 occurence in v
        # to n repetition in w) to ease a systematic writing of the loglike.
        stop = step = shape_product or 1

        w = np.empty((2 + len(current_pattern)) * (shape_product or 1))
        stop = step = shape_product or 1
        w[0:stop] = w_offset = v[offset_access]
        w[stop : stop + step] = w_f2k = v[field_to_k_access]
        stop += step
        for sl in current_density_accesses:
            w[stop : stop + step] = v[sl]

        # Pack the current distribution so that each line corresponds to different
        # conditions
        c_density = w[stop + step :].reshape((len(current_pattern), -1)).T

        err = np.empty_like(ics)

        it = np.nditer((offsets, first_node_locs, field_to_ks), ["multi_index"])
        for i, (off, fnloc, f2k) in enumerate(it):
            # Compute the offset
            f_off = off + np.sign(w_off[i]) * 10 ** -abs(w_off[i]) * fnloc

            # Compute the Fraunhofer pattern
            f = produce_fraunhofer_fast(
                (fields[it.multi_index] - f_off[i]),
                f2k * 10 ** w_f2k[i],
                jj_size,
                c_density[i],
                2 ** 10 + 1,
            )

            # Compute and store the error
            err[it.multi_index] = np.sum(
                (100 * (ics[it.multi_index] - f) / amplitude) ** 2
            )

        return -np.ravel(err)

    # XXX do that nasty part later
    sampler = NestedSampler(loglike, prior, dim)
    sampler.run_nested(dlogz=precision)
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    mu, cov = utils.mean_and_cov(res["samples"], weights)

    res["fraunhofer_params"] = {
        "offset": offset + np.sign(mu[0]) * 10 ** -abs(mu[0]) * first_node_loc,
        "field_to_k": 2 * np.pi / jj_size / abs(first_node_loc - offset) * 10 ** mu[1],
        "amplitude": amplitude * 10 ** mu[2],
        "current_distribution": np.array(
            [1 - np.sum(mu[3 : 3 + site_number - 1])]
            + list(mu[3 : 3 + site_number - 1])
        ),
        "phase_distribution": np.array(
            [0] + list(mu[3 + site_number - 1 : 3 + 2 * site_number - 2])
        ),
    }

    return res
