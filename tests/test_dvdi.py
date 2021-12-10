# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Unit tests for shabanipy.dvdi module."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from shabanipy.dvdi import extract_switching_current


class TestExtractSwitchingCurrent(unittest.TestCase):
    """Unit tests for extract_switching_current function."""

    def test_threshold_with_1d_arrays(self):
        bias = np.array([-2, -1, 0, 1, 2])
        dvdi = np.array([1, 0, 0, 0, 1])
        self.assertEqual(extract_switching_current(bias, dvdi, threshold=0.5), 2)
        self.assertEqual(
            extract_switching_current(bias, dvdi, threshold=0.5, side="negative"), -2
        )
        self.assertEqual(
            extract_switching_current(bias, dvdi, threshold=0.5, side="both"), (-2, 2)
        )

    def test_threshold_with_2d_arrays(self):
        bias = np.array([[-2, -1, 0, 1, 2], [-20, -10, 0, 10, 20]])
        dvdi = np.array([[1, 0, 0, 0, 1], [1, 1, 0, 1, 1]])
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5), [2, 10]
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5, side="negative"),
            [-2, -10],
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5, side="both"),
            ([-2, -10], [2, 10]),
        )

    def test_threshold_with_3d_arrays(self):
        # 2x3 bias sweeps of [-2, -1, 0, 1, 2]
        bias = np.tile(np.linspace(-2, 2, 5), (2, 3, 1))
        dvdi = np.array(
            [
                [[1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1]],
                [[1, 1, 0, 1, 1], [1, 1, 0, 0, 1], [1, 0, 0, 0, 1]],
            ]
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5), [[2, 1, 2], [1, 2, 2]]
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5, side="negative"),
            [[-2, -2, -1], [-1, -1, -2]],
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5, side="both"),
            ([[-2, -2, -1], [-1, -1, -2]], [[2, 1, 2], [1, 2, 2]]),
        )


if __name__ == "__main__":
    unittest.main()
