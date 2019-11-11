# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Rebuild the current distribution from a Fraunhofer like pattern.

"""
import numpy as np
import matplotlib.pyplot as plt
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty import utils

from .util import produce_fraunhofer_fast


def guess_current_distribution(field: np.ndarray,
                               fraunhofer: np.ndarray,
                               site_number: int,
                               jj_size: float,
                               ) -> (float, float, np.ndarray, np.ndarray):
    """Determine reasonable starting points for reconstructing the current.

    """
    pattern_max_loc = np.argmax(fraunhofer)
    field_offset = field[pattern_max_loc]
    amplitude = fraunhofer[pattern_max_loc]

    # Determine the position of the first minima and 2nd maximum to start with a
    # Fraunhofer like distribution or a SQUID like distribution
    i = pattern_max_loc
    while fraunhofer[i] > amplitude/2 or fraunhofer[i+1] < fraunhofer[i]:
        i += 1
    minimum = i

    while fraunhofer[i+1] > fraunhofer[i]:
        i += 1
    maximum = i

    # If the first node is close to double in size to second we start from a
    # a uniform current distribution otherwise we start from a squid like one.
    half_first_node = minimum - pattern_max_loc
    half_second_node = maximum - minimum

    # Fraunhofer like
    if (abs((2*half_second_node - half_first_node)) <
            abs(half_first_node - half_second_node)):
        c_dis = np.ones(site_number)/site_number/jj_size
        p_dis = np.zeros(site_number)
    # SQUID like
    else:
        c_dis = np.zeros(site_number)
        c_dis[0] = c_dis[-1] = 0.5*site_number/jj_size
        phase_slope = np.pi/jj_size/abs(field[minimum] - field_offset)
        p_dis = np.linspace(0, phase_slope, site_number)

    return (field_offset, field[minimum], amplitude, c_dis, p_dis)


def rebuild_current_distribution(field, fraunhofer, jj_size, site_number,
                                 precision=100, dimension=11,
                                 current_distribution=None,
                                 phase_distribution=None):
    """Rebuild a current distribution from a Fraunhofer pattern.

    Parameters
    ----------
    field : [type]
        [description]
    fraunhofer : [type]
        [description]
    jj_size : [type]
        [description]

    Returns
    -------

    """
    # Get the offset and estimated amplitude used in the prior
    # We do not use the estimated current and phase distribution to give the
    # more space to the algorithm.
    offset, first_node_loc, amplitude, _, _ = guess_current_distribution(field,
                                                                         fraunhofer,
                                                                         site_number,
                                                                         jj_size)

    def prior(u):
        """Map the sampled in 0-1 to the relevant values.

        For the offset and amplitude since the bounds are well known we use a linear
        mapping. For the field to k ratio, the current and phase distribution we will
        consider the values in the prior to be the log of the values we are looking for.

        """
        # The first site of the phase distribution is set to zero to have a phase
        # reference.
        sn = site_number - 1

        v = np.empty_like(u)
        v[0] = 2*u[0] - 2  # field offset
        v[1] = 4*u[1] - 2  # field to k conversion factor
        v[2] = 2*u[2] - 1  # amplitude

        # Current distribution such that the integral is always 1
        if dimension > 3:
            initial = u[3:3+sn]
            dividers = sorted(initial)
            v[3:3+sn] = np.array([a - b for a, b in zip(dividers + [1],
                                                        [0] + dividers)])[:-1]

        # Phase distribution before normalization allowed between 1e-1 and 1e1
        if dimension > 7:
            v[3+sn:3+2*sn] = 4*u[3+sn:3+2*sn] - 2

        return v

    def loglike(v):
        """Compute the distance to the data.

        """
        # The first site of the phase distribution is set to zero to have a phase
        # reference.
        sn = site_number - 1

        f_off = offset + np.sign(v[0])*10**-abs(v[0])*first_node_loc
        # Gives a Fraunhofer pattern at the first node for v[1] = 1
        field_to_k = 2*np.pi/jj_size/abs(first_node_loc - offset)*10**v[1]
        amp = amplitude*10**v[2]

        # Compute the current distribution
        c_dis = np.ones(site_number)
        if current_distribution is not None:
            c_dis = current_distribution
        if dimension > 3:
            c_dis[1:] = v[3:3+sn]
            c_dis[0] = 1 - np.sum(c_dis[:-1])

        # Slope leading to a SQUID like pattern with the periodicity extracted from the
        # first node.
        phase_slope = np.pi/jj_size/abs(first_node_loc - offset)

        p_dis = np.zeros(site_number)
        if phase_distribution is not None:
            p_dis = phase_distribution
        if dimension > 7:
            p_dis[0] = 0
            p_dis[1:] = (np.sign(v[3+sn:3+2*sn]) *
                         10**np.abs(v[3+sn:3+2*sn]) * 10 * phase_slope)

        f = amp*produce_fraunhofer_fast((field - f_off), field_to_k, jj_size,
                                        c_dis, p_dis, 2**10+1)

        err = np.sum((100*(fraunhofer - f)/amplitude)**2)

        return -err

    sampler = NestedSampler(loglike, prior, dimension)
                            # bound="balls", sample="rstagger")
    sampler.run_nested(dlogz=precision)
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    mu, cov = utils.mean_and_cov(res["samples"], weights)

    res["fraunhofer_params"] = \
        {"offset": offset + np.sign(mu[0])*10**-abs(mu[0])*first_node_loc,
         "field_to_k": 2*np.pi/jj_size/abs(first_node_loc - offset)*10**mu[1],
         "amplitude": amplitude*10**mu[2],
         "current_distribution": np.array([1 - np.sum(mu[3:3+site_number-1])] +
                                          list(mu[3:3+site_number-1])),
         "phase_distribution": np.array([0] +
                                        list(mu[3+site_number-1:3+2*site_number-2])),
        }

    return res
