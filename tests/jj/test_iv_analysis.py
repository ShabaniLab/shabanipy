# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Test iv analysis functions.

"""
import numpy as np
from shabanipy.jj.iv_analysis import analyse_vi_curve


def test_voltage_offset_symmetric():
    """Test extracting the voltage offset for 1D, 2D and 3D data.

    """
    current = np.linspace(-1, 1, 101)
    voltage = np.concatenate(
        (np.linspace(-1, -0.2, 40), np.zeros(31), np.linspace(0.2, 1, 30))
    )

    r1 = (voltage[-1] - voltage[-2]) / (current[-1] - current[-2])
    r2 = (voltage[0] - voltage[1]) / (current[0] - current[1])

    # 1D tests
    rn_c, rn_h, ic_c, ic_h, ie_c, ie_h = analyse_vi_curve(
        current, voltage, 0.05, 0.5, True,
    )
    assert abs(rn_c - r1) < 1e-6
    assert abs(rn_h - r2) < 1e-6
    assert ic_c == current[70]
    assert ic_h == -current[40]
    assert abs(ie_c - 0.275) < 1e-6
    assert abs(ie_h - 0.025) < 1e-6

    rn_c, rn_h, ic_c, ic_h, ie_c, ie_h = analyse_vi_curve(
        current[::-1], voltage[::-1] + 0.1, 0.05, 0.5, True,
    )
    assert abs(rn_c - r2) < 1e-6
    assert abs(rn_h - r1) < 1e-6
    assert ic_c == -current[40]
    assert ic_h == current[70]
    assert abs(ie_c - 0.025) < 1e-6
    assert abs(ie_h - 0.275) < 1e-6

    # 2D test
    c2d = np.empty((2, 101))
    c2d[0] = current
    c2d[1] = current[::-1]
    v2d = np.empty((2, 101))
    v2d[0] = voltage
    v2d[1] = voltage[::-1] + 0.1

    rn_c, rn_h, ic_c, ic_h, ie_c, ie_h = analyse_vi_curve(
        c2d, v2d, 0.05, 0.5, True,
    )

    np.testing.assert_array_almost_equal(rn_c, np.array([r1, r2]))
    np.testing.assert_array_almost_equal(rn_h, np.array([r2, r1]))
    np.testing.assert_array_equal(ic_c, np.array([current[70], -current[40]]))
    np.testing.assert_array_equal(ic_h, np.array([-current[40], current[70]]))
    np.testing.assert_array_almost_equal(ie_c, np.array([0.275, 0.025]))
    np.testing.assert_array_almost_equal(ie_h, np.array([0.025, 0.275]))

    # 3D test
    c3d = np.empty((2, 2, 101))
    c3d[0, ..., :] = current
    c3d[1, ..., :] = current[::-1]
    v3d = np.empty((2, 2, 101))
    v3d[0, ..., :] = voltage
    v3d[1, ..., :] = voltage[::-1] + 0.1

    rn_c, rn_h, ic_c, ic_h, ie_c, ie_h = analyse_vi_curve(
        c3d, v3d, 0.05, 0.5, True,
    )

    np.testing.assert_array_almost_equal(rn_c, np.array([[r1, r1], [r2, r2]]))
    np.testing.assert_array_almost_equal(rn_h, np.array([[r2, r2], [r1, r1]]))
    np.testing.assert_array_equal(
        ic_c, np.array([[current[70], current[70]], [-current[40], -current[40]]])
    )
    np.testing.assert_array_equal(
        ic_h, np.array([[-current[40], -current[40]], [current[70], current[70]]])
    )
    np.testing.assert_array_almost_equal(
        ie_c, np.array([[0.275, 0.275], [0.025, 0.025]])
    )
    np.testing.assert_array_almost_equal(
        ie_h, np.array([[0.025, 0.025], [0.275, 0.275]])
    )
