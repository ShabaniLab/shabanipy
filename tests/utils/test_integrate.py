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

from shabanipy.utils.integrate import can_romberg


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


if __name__ == '__main__':
    unittest.main()
