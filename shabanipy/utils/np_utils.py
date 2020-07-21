# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Numpy related utility function.

"""


def scalar_if_0d(array):
    """Turn a 0d array in scalar."""
    return array.item() if array.ndim == 0 else array
