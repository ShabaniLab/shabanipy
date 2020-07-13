# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Numerical integration routines and helpers."""


def can_romberg(n_points: int) -> bool:
    """Determine if n_points is of the form 2**n + 1.

    Romberg integration requires 2**n + 1 samples.
    """
    return n_points > 1 and not (n_points - 1) & (n_points - 2)
