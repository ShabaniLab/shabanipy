"""Unit tests for shabanipy.labber.shablabber module."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from shabanipy.labber.shablabber import _pad_and_reshape


class TestPadAndReshape(unittest.TestCase):
    """Unit tests for _pad_and_reshape function."""

    def setUp(self):
        self.a = np.array([0.0])

    def tearDown(self):
        del self.a

    def test_pad_and_reshape_1d(self):
        actual = _pad_and_reshape(self.a, (2,))
        expected = self.a
        assert_array_equal(actual, expected, err_msg="Shouldn't pad at all")

    def test_pad_and_reshape_2d(self):
        actual = _pad_and_reshape(self.a, (2, 2))
        expected = np.array([0, np.nan]).reshape((2, 1), order="F")
        assert_array_equal(actual, expected, err_msg="Should pad 1st but not last axis")

    def test_pad_and_reshape_3d(self):
        actual = _pad_and_reshape(self.a, (2, 2, 2))
        expected = np.array([0, np.nan, np.nan, np.nan]).reshape((2, 2, 1), order="F")
        assert_array_equal(
            actual, expected, err_msg="Should pad 1st and 2nd but not last axis"
        )

    def test_pad_and_reshape_3d_1st_axis_full(self):
        a = np.array([0, 1], dtype=float)
        actual = _pad_and_reshape(a, (2, 2, 2))
        expected = np.array([0, 1, np.nan, np.nan]).reshape((2, 2, 1), order="F")
        assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
