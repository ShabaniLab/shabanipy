# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Test Fraunhofer estimation.

"""
import numpy as np
from shabanipy.jj.fraunhofer.estimation import guess_current_distribution


def create_fraunhofer_like():
    fields = np.linspace(-1, 1, 1001)
    return fields, np.abs(np.sinc(8 * (fields - 0.1)))


def create_squid_like():
    fields = np.linspace(-1, 1, 1001)
    return (
        fields,
        2 + np.cos(8 * np.pi * (fields + 0.1)) * np.sinc(0.1 * (fields + 0.1)),
    )


def validate_fraunhofer(offset, first_node, amplitude, c_dis):
    np.testing.assert_almost_equal(offset, 0.1)
    assert abs(first_node + 0.025) < 0.05
    np.testing.assert_almost_equal(amplitude, 1.0)
    np.testing.assert_array_equal(c_dis, np.ones(5) / 20)


def validate_squid(offset, first_node, amplitude, c_dis):
    np.testing.assert_almost_equal(offset, -0.1)
    assert abs(first_node - 0.025) < 0.05
    np.testing.assert_almost_equal(amplitude, 3.0)
    np.testing.assert_array_equal(c_dis, np.array([0.625, 0, 0, 0, 0.625]))


def test_guess_current_distribution_fraunhofer():
    """Test identifying a fraunhofer like pattern.

    """
    fields, fraunhofer_like_ics = create_fraunhofer_like()

    offsets, first_nodes, amplitudes, c_dis = guess_current_distribution(
        fields, fraunhofer_like_ics, 5, 4
    )

    validate_fraunhofer(offsets, first_nodes, amplitudes, c_dis)


def test_guess_current_distribution_squid():
    """Test identifying a SQUID like pattern.

    """
    fields, squid_like_ics = create_squid_like()

    offsets, first_nodes, amplitudes, c_dis = guess_current_distribution(
        fields, squid_like_ics, 5, 4
    )

    validate_squid(offsets, first_nodes, amplitudes, c_dis)


def test_guess_current_distribution_too_small_data():
    """Test handling data which do not comport enough points.

    """
    fields = np.linspace(-1, 1, 201)
    fraunhofer_like_ics = np.abs(np.sinc(2 * (fields - 0.1)))

    offsets, first_nodes, amplitudes, c_dis = guess_current_distribution(
        fields, fraunhofer_like_ics, 5, 4
    )
    np.testing.assert_almost_equal(offsets, 0.1)
    assert amplitudes == 1.0


def test_2D_inputs():
    """Test that we can handle properly 2D inputs."""
    fields_f, fraunhofer_like_ics = create_fraunhofer_like()
    fields_s, squid_like_ics = create_squid_like()

    # 2D inputs
    fields = np.empty((2, len(fields_f)))
    fields[0] = fields_f
    fields[1] = fields_s

    ics = np.empty_like(fields)
    ics[0] = fraunhofer_like_ics
    ics[1] = squid_like_ics

    offsets, first_nodes, amplitudes, c_dis = guess_current_distribution(
        fields, ics, 5, 4
    )

    for o, f, a, cd, validator in zip(
        offsets, first_nodes, amplitudes, c_dis, (validate_fraunhofer, validate_squid)
    ):
        validator(o, f, a, cd)


def test_3D_inputs():
    """Test that we can handle properly 3D inputs."""
    fields_f, fraunhofer_like_ics = create_fraunhofer_like()
    fields_s, squid_like_ics = create_squid_like()

    # 3D inputs
    fields = np.empty((2, 2, len(fields_f)))
    fields[0, :] = fields_f
    fields[1, :] = fields_s

    ics = np.empty_like(fields)
    ics[0, :] = fraunhofer_like_ics
    ics[1, :] = squid_like_ics

    offsets, first_nodes, amplitudes, c_dis = guess_current_distribution(
        fields, ics, 5, 4
    )

    for o, f, a, cd, validator in zip(
        offsets, first_nodes, amplitudes, c_dis, (validate_fraunhofer, validate_squid)
    ):
        validator(o[0], f[0], a[0], cd[0])
        validator(o[1], f[1], a[1], cd[1])
