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
from typing import Tuple, Union

import numpy as np

from shabanipy.utils.np_utils import scalar_if_0d


def guess_current_distribution(
    fields: np.ndarray, ics: np.ndarray, site_number: int, jj_size: float,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    np.ndarray,
]:
    """Determine reasonable starting points for reconstructing the current.

    Parameters
    ----------
    fields : np.ndarray
        Out of plane magnetic fields at which the JJ critical current was measured.
        If more than 1D the magnetic field sweep is expected to happen on the
        last axis.
    ics : np.ndarray
        Critical current of the JJ. If more than 1D the magnetic field sweep is
        expected to happen on the last axis.
    site_number : int
        Number of sites to use in the description of the JJ.
    jj_size : float
        Size of the JJ.

    Returns
    -------
    field_offset : float | np.ndarray
        Offset in field (assuming the max of the pattern is always at 0).
    field_first_node : float | np.ndarray
        Field at which we observe the first minimum in the pattern.
    amplitude : np.ndarray
        Maximal amplitude of the current in the pattern
    current_distribution : np.ndarray
        Array of the current distribution.

    """
    pattern_max_locs = np.argmax(ics, axis=-1)
    if fields.ndim > 1:
        field_offsets = np.take_along_axis(
            fields, pattern_max_locs[..., None], axis=-1
        ).reshape(pattern_max_locs.shape)
        amplitudes = np.take_along_axis(
            ics, pattern_max_locs[..., None], axis=-1
        ).reshape(pattern_max_locs.shape)
    else:
        field_offsets = fields[pattern_max_locs]
        amplitudes = ics[pattern_max_locs]

    it = np.nditer((pattern_max_locs, field_offsets, amplitudes), ["multi_index"])

    # Determine the position of the first minima and 2nd maximum to start with a
    # Fraunhofer like distribution or a SQUID like distribution

    c_dis = np.empty(field_offsets.shape + (site_number,))
    first_nodes = np.empty_like(field_offsets)
    for pattern_max_loc, field_offset, amplitude in it:
        i = int(pattern_max_loc)
        ic = ics[it.multi_index]
        len_ic = len(ic)

        # Go in the direction in which we have more points
        increment = 1 if i < len_ic // 2 else -1

        while 0 <= i + increment < len_ic and (
            ic[i] > amplitude / 2 or ic[i + increment] < ic[i]
        ):
            i += increment
        minimum = i
        first_nodes[it.multi_index] = fields[it.multi_index][minimum]

        while 0 <= i + increment < len_ic and ic[i + increment] > ic[i]:
            i += 1
        second_maximum = ic[i]
        while 0 <= i + increment < len_ic and ic[i + increment] < ic[i]:
            i += 1
        second_minimum = i

        # To choose between squid and Fraunhofer and SQUID we use the periodicity
        # of the structure and the relative amplitude of the first and second node.

        # If the first node is close to double in size to the second and the
        # ratio of amplitude is close to 0.2, we select a uniform current
        # distribution otherwise we select from a squid like one.
        half_first_node = abs(minimum - pattern_max_loc)
        second_node = second_minimum - minimum

        # Fraunhofer like
        if (
            abs((second_node - half_first_node))
            < abs(half_first_node - second_node / 2)
            and second_maximum / amplitude < 0.6
        ):
            aux = np.ones(site_number) / site_number / jj_size

        # SQUID like
        else:
            aux = np.zeros(site_number)
            aux[0] = aux[-1] = 0.5 * site_number / jj_size

        c_dis[it.multi_index] = aux

    return (
        scalar_if_0d(field_offsets),
        scalar_if_0d(first_nodes),
        scalar_if_0d(amplitudes),
        c_dis,
    )
