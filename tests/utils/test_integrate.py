# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Unit tests for shabanipy.utils.integrate module."""
import unittest

import numpy as np

from shabanipy.utils.integrate import can_romberg, resample_evenly


class TestCanRomberg(unittest.TestCase):
    """Unit tests for can_romberg function."""

    def test_is_1_plus_power_of_2(self):
        """Numbers like 2**n + 1 return true."""
        result = [n for n in range(1026) if can_romberg(n)]
        expected = [2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]
        self.assertEqual(result, expected)

    def test_is_greater_than_1(self):
        """Numbers less than 2 return false."""
        result = [n for n in range(-1025, 2) if can_romberg(n)]
        self.assertFalse(result)


class TestResampleData(unittest.TestCase):
    """Unit tests for resample_evenly function."""

    def setUp(self):
        """Set up test data."""
        self.x = np.linspace(0, 2*np.pi, 101)
        self.y = self._signal(self.x)

    def _signal(self, x):
        """Use y(x) = sin(x) + 2 to generate test data."""
        return np.sin(x) + 2

    def test_same_n_points(self):
        """Samples are (roughly) unchanged if sampled at same rate."""
        n_points = len(self.x)
        x, y = resample_evenly(self.x, self.y, n_points)
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_allclose(y, self.y)

    def test_half_n_points(self):
        """Every other sample is dropped if (odd) n_points is ~halved."""
        n_points = len(self.x) // 2 + 1
        x, y = resample_evenly(self.x, self.y, n_points)
        np.testing.assert_array_equal(x, self.x[::2])
        np.testing.assert_allclose(y, self.y[::2])

    def test_double_n_points(self):
        """Samples are correctly interpolated when n_points is ~doubled."""
        n_points = len(self.x) * 2 - 1
        x, y = resample_evenly(self.x, self.y, n_points)
        x_expected = np.linspace(self.x[0], self.x[-1], n_points)
        y_expected = self._signal(x_expected)
        np.testing.assert_array_equal(x, x_expected)
        np.testing.assert_allclose(y, y_expected)

    def test_evenly_spaced(self):
        """Unevenly spaced input generates evenly spaced output."""
        x_uneven = np.logspace(1, 2, 100)
        x_even, _ = resample_evenly(x_uneven, np.zeros_like(x_uneven),
                                  len(x_uneven))
        dx = np.diff(x_even)
        np.testing.assert_allclose(dx, dx[0])


if __name__ == '__main__':
    unittest.main()
