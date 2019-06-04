# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
""" Common tools to improve plots.

"""
import numpy as np


def format_phase(value, tick_number):
    """The value are expected in unit of π

    """
    if value == 0:
        return '0'
    elif value == 1.0:
        return 'π'
    elif value == -1.0:
        return '-π'
    else:
        return f'{value:g}π'
