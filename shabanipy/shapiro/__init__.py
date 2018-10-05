# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines used in the analysis of shapiro steps experiments.

"""


def shapiro_step(frequency):
    """ Compute the amplitude of a Shapiro step at a given frequency.

    """
    return 6.626e-34*frequency/(2*1.6e-19)
