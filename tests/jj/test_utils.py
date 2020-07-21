# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Test utility functions.

"""
import numpy as np
from shabanipy.jj.utils import compute_voltage_offset, extract_switching_current


def test_voltage_offset_symmetric():
    """Test extracting the voltage offset for 1D, 2D and 3D data.

    """
    current = np.linspace(-1, 1, 101)
    voltage = np.concatenate(
        (np.linspace(-1, 0, 40), np.zeros(31), np.linspace(0, 1, 30))
    )

    assert compute_voltage_offset(current, voltage, 10) == 0
    assert compute_voltage_offset(current, voltage, 51) == np.average(voltage)

    c2d = np.empty((2, 101))
    c2d[..., :] = current
    v2d = np.empty((2, 101))
    v2d[0] = voltage
    v2d[1] = voltage + 1

    np.testing.assert_equal(compute_voltage_offset(c2d, v2d, 10), np.array([0, 1]))

    c3d = np.empty((2, 2, 101))
    c3d[..., :] = current
    v3d = np.empty((2, 2, 101))
    v3d[0, 0] = voltage
    v3d[0, 1] = voltage + 1
    v3d[1, 0] = voltage + 2
    v3d[1, 1] = voltage - 1

    np.testing.assert_equal(
        compute_voltage_offset(c3d, v3d, 10), np.array([[0, 1], [2, -1]])
    )


def test_voltage_offset_asymmetric():
    """Test extracting the voltage offset for 1D, 2D and 3D data.

    """
    current = np.linspace(0, 1, 101)
    voltage = np.concatenate((np.zeros(71), np.linspace(0, 1, 30)))

    assert compute_voltage_offset(current, voltage, 10) == 0
    assert compute_voltage_offset(current, voltage, 101) == np.average(voltage)

    c2d = np.empty((2, 101))
    c2d[..., :] = current
    v2d = np.empty((2, 101))
    v2d[0] = voltage
    v2d[1] = voltage + 1

    np.testing.assert_equal(compute_voltage_offset(c2d, v2d, 10), np.array([0, 1]))

    c3d = np.empty((2, 2, 101))
    c3d[..., :] = current
    v3d = np.empty((2, 2, 101))
    v3d[0, 0] = voltage
    v3d[0, 1] = voltage + 1
    v3d[1, 0] = voltage + 2
    v3d[1, 1] = voltage - 1

    np.testing.assert_equal(
        compute_voltage_offset(c3d, v3d, 10), np.array([[0, 1], [2, -1]])
    )


def test_extract_switching():
    """Test extracting the switching current offset for 1D, 2D and 3D data.

    """
    current = np.linspace(-1, 1, 101)
    voltage = np.concatenate(
        (np.linspace(-1, -0.1, 40), np.zeros(31), np.linspace(0.1, 1, 30))
    )

    assert extract_switching_current(current, voltage, 0.04) == current[70]
    assert extract_switching_current(current, voltage, 0.04, "negative") == current[40]
    assert (
        extract_switching_current(current, voltage + 0.1, 0.04, offset_correction=10)
        == current[70]
    )

    c2d = np.empty((2, 101))
    c2d[0] = current
    c2d[1] = -current
    v2d = np.empty((2, 101))
    v2d[0] = voltage
    v2d[1] = -voltage + 1

    np.testing.assert_equal(
        extract_switching_current(c2d, v2d, 0.04, offset_correction=10),
        np.array([current[70], -current[40]]),
    )

    c3d = np.empty((2, 2, 101))
    c3d[..., :] = current
    v3d = np.empty((2, 2, 101))
    v3d[0, 0] = voltage
    v3d[0, 1] = -voltage[::-1]
    v3d[1, 0] = -voltage[::-1] + 0.1
    v3d[1, 1] = voltage - 0.1

    np.testing.assert_equal(
        extract_switching_current(c3d, v3d, 0.04, offset_correction=10),
        np.array([[current[70], -current[40]], [-current[40], current[70]]]),
    )
