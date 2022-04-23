# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Unit tests for shabanipy.squid.squid_model module."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from shabanipy.jj import josephson_cpr as jcpr
from shabanipy.jj import transparent_cpr as tcpr
from shabanipy.squid import critical_behavior


class TestComputeSquidCriticalBehavior(unittest.TestCase):
    """Unit tests for critical_behavior function."""

    def test_scalar_phase(self):
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            0, jcpr, (0, 1), jcpr, (0, 1), return_jjs=True
        )
        self.assertEqual(p_ext, 0)
        self.assertAlmostEqual(p1, np.pi / 2)
        self.assertAlmostEqual(p2, np.pi / 2)
        self.assertEqual(ic, 2)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 1)

    def test_vector_phase(self):
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            [0, 2 * np.pi], jcpr, (0, 1), jcpr, (0, 1), return_jjs=True
        )
        assert_array_equal(p_ext, [0, 2 * np.pi])
        assert_allclose(p1 % (2 * np.pi), [np.pi / 2] * 2)
        assert_allclose(p2 % (2 * np.pi), [np.pi / 2] * 2)
        assert_array_equal(ic, [2, 2])
        assert_array_equal(c1, [1, 1])
        assert_array_equal(c2, [1, 1])

    def test_flux_quantization(self):
        p = np.linspace(0, 2 * np.pi, 101)
        *_, p1, _, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 1), return_jjs=True
        )
        assert_allclose(p1 - p2, p)  # symmetric squid
        *_, p1, _, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 100), return_jjs=True
        )
        assert_allclose(p1 - p2, p)  # asymmetric squid
        *_, p1, _, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 3), inductance=1, return_jjs=True
        )
        assert_allclose(p1 - p2, p)  # nonzero inductance

    def test_symmetric_squid(self):
        # remove ill-behaved point at Φ=π phase-slip
        p = np.delete(np.linspace(0, 2 * np.pi, 101), 50)
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 1), nbrute=201, return_jjs=True
        )
        assert_array_equal(p_ext, p)
        assert_allclose(ic, 2 * np.abs(np.cos(p / 2)))
        p1_expected = np.linspace(np.pi / 2, 3 * np.pi / 2, 101) % np.pi
        p1_expected = np.delete(p1_expected, 50)
        p2_expected = (p1_expected - p) % np.pi
        assert_allclose(p1 % (2 * np.pi), p1_expected)
        assert_allclose(p2 % (2 * np.pi), p2_expected)
        assert_allclose(c1, np.abs(np.cos(p / 2)))
        assert_allclose(c2, np.abs(np.cos(p / 2)))

    def test_asymmetric_squid(self):
        p = np.linspace(0, 2 * np.pi, 101)
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 100), return_jjs=True
        )
        assert_array_equal(p_ext, p)
        assert_allclose(p1, np.pi / 2 + np.linspace(0, 2 * np.pi, 101))
        assert_allclose(p2, [np.pi / 2] * 101)
        assert_array_equal(ic, 100 + np.cos(p))
        assert_allclose(c1, np.cos(p), atol=1e-15)  # c1 ~ 0 -> large relative error
        assert_array_equal(c2, [100] * 101)

    def test_inductance_symmetric_squid(self):
        # remove ill-behaved point at Φ=π phase-slip
        p = np.delete(np.linspace(0, 2 * np.pi, 101), 50)
        # symmetric SQUID has c1=c2 critical behavior -> inductance has no effect
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 1), inductance=0.1, return_jjs=True, nbrute=201
        )
        ic0, p_ext0, c10, p10, c20, p20 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 1), inductance=0, return_jjs=True, nbrute=201
        )
        assert_allclose(p_ext, p_ext0)
        # outputs are a function of total phase -> inductance has no effect
        assert_array_equal(ic, ic0)
        assert_array_equal(p1, p10)
        assert_array_equal(c1, c10)
        assert_array_equal(p2, p20)
        assert_array_equal(c2, c20)

    def test_inductance_asymmetric_squid(self):
        p = np.linspace(0, 2 * np.pi, 101)
        ic0, p_ext0, c10, p10, c20, p20 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 10), inductance=0, return_jjs=True
        )
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, jcpr, (0, 1), jcpr, (0, 10), inductance=0.2, return_jjs=True
        )
        assert_array_equal(p_ext[[0, -1]], np.array([0, 2 * np.pi]) - 0.2 * 9 * np.pi)
        # outputs are a function of total phase -> inductance has no effect
        assert_array_equal(ic, ic0)
        assert_array_equal(p1, p10)
        assert_array_equal(c1, c10)
        assert_array_equal(p2, p20)
        assert_array_equal(c2, c20)

    def test_finite_transparency_symmetric_squid(self):
        p = np.linspace(0, 2 * np.pi, 101)
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, tcpr, (0, 1, 0.9), tcpr, (0, 1, 0.9), inductance=0.1, return_jjs=True
        )
        assert_allclose(p1 - p2, p)
        assert_array_equal(c1 + c2, ic)
        assert_allclose(p_ext + 0.1 * (c2 - c1) * np.pi, p)

    def test_finite_transparency_asymmetric_squid(self):
        p = np.linspace(0, 2 * np.pi, 101)
        ic, p_ext, c1, p1, c2, p2 = critical_behavior(
            p, tcpr, (0, 1, 0.9), tcpr, (0, 10, 0.9), inductance=0.1, return_jjs=True
        )
        assert_allclose(p1 - p2, p)
        assert_array_equal(c1 + c2, ic)
        assert_allclose(p, p_ext + 0.1 * (c2 - c1) * np.pi)


if __name__ == "__main__":
    unittest.main()
