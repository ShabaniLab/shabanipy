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

from shabanipy.dvdi import _compute_offset, extract_switching_current


class TestExtractSwitchingCurrent(unittest.TestCase):
    """Unit tests for extract_switching_current function."""

    def test_threshold_with_1d_arrays(self):
        bias = np.array([-2, -1, 0, 1, 2])
        dvdi = np.array([1, 0, 0, 0, 1])
        self.assertEqual(extract_switching_current(bias, dvdi, threshold=0.5), 2)
        self.assertEqual(
            extract_switching_current(bias, dvdi, threshold=0.5, side="negative"), -2
        )
        assert_array_equal(
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

    def test_interpolation(self):
        # 2x3 bias sweeps of [-2, -1, 0, 1, 2]
        bias = np.tile(np.linspace(-2, 2, 5), (2, 3, 1))
        dvdi = np.array(
            [
                [[1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1]],
                [[1, 1, 0, 1, 1], [1, 1, 0, 0, 1], [1, 0, 0, 0, 1]],
            ]
        )
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=0.5, interp=True),
            [[1.5, 0.5, 1.5], [0.5, 1.5, 1.5]],
        )

    def test_auto_threshold_1d(self):
        bias = np.array([-2, -1, 0, 1, 2])
        dvdi = np.array([2, 1, 0, 4, 8])
        self.assertEqual(
            extract_switching_current(bias, dvdi, threshold=None, interp=True), 1
        )
        self.assertEqual(
            extract_switching_current(
                bias, dvdi, threshold=None, side="negative", interp=True
            ),
            -1,
        )
        assert_array_equal(
            extract_switching_current(
                bias, dvdi, threshold=None, side="both", interp=True
            ),
            (-1, 1),
        )

    def test_auto_threshold_2d(self):
        bias = np.array([[-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4]])
        dvdi = np.array([[2, 1, 0, 2, 4], [10, 5, 0, 3, 6]])
        assert_array_equal(
            extract_switching_current(bias, dvdi, threshold=None, interp=True), [1, 2]
        )
        assert_array_equal(
            extract_switching_current(
                bias, dvdi, threshold=None, side="negative", interp=True
            ),
            [-1, -2],
        )
        assert_array_equal(
            extract_switching_current(
                bias, dvdi, threshold=None, side="both", interp=True
            ),
            [[-1, -2], [1, 2]],
        )


class TestComputeOffset(unittest.TestCase):
    """Unit tests for _compute_offset function."""

    def test_compute_offset_1d(self):
        x = np.array([-2, -1, 0, 1, 2])
        y = np.array([0, 1, 2, 1, 0])
        assert_array_equal(_compute_offset(x, y, 1), [2])
        assert_array_equal(_compute_offset(x, y, 2), [4 / 3])
        assert_array_equal(_compute_offset(x, y, 3), [4 / 3])
        assert_array_equal(_compute_offset(x, y, 4), [4 / 5])
        assert_array_equal(_compute_offset(x, y, 5), [4 / 5])

    def test_compute_offset_2d(self):
        # x=0 at different positions in each slice
        x = np.array([[-3, -2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3, 4]])
        y = np.array([[0, 1, 2, 3, 2, 1, 0], [-1, 5, 10, 20, 15, 4, 0]])
        assert_array_equal(_compute_offset(x, y, 1), [[3], [10]])
        assert_array_equal(_compute_offset(x, y, 2), [[7 / 3], [35 / 3]])
        assert_array_equal(_compute_offset(x, y, 3), [[7 / 3], [35 / 3]])
        assert_array_equal(_compute_offset(x, y, 4), [[9 / 5], [49 / 5]])
        assert_array_equal(_compute_offset(x, y, 5), [[9 / 5], [49 / 5]])

    def test_compute_offset_3d(self):
        x = np.broadcast_to([-2, -1, 0, 1, 2], (2, 2, 5))
        y = np.abs(x)
        assert_array_equal(_compute_offset(x, y, 1), [[[0], [0]], [[0], [0]]])
        assert_array_equal(
            _compute_offset(x, y, 2), [[[2 / 3], [2 / 3]], [[2 / 3], [2 / 3]]]
        )
        assert_array_equal(
            _compute_offset(x, y, 3), [[[2 / 3], [2 / 3]], [[2 / 3], [2 / 3]]]
        )
        assert_array_equal(
            _compute_offset(x, y, 4), [[[6 / 5], [6 / 5]], [[6 / 5], [6 / 5]]]
        )
        assert_array_equal(
            _compute_offset(x, y, 5), [[[6 / 5], [6 / 5]], [[6 / 5], [6 / 5]]]
        )


if __name__ == "__main__":
    unittest.main()
