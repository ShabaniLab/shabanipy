# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Superconducting quantum interference devices."""
from .estimate import (
    estimate_bfield_offset,
    estimate_critical_current,
    estimate_frequency,
)
from .lmfitmodels import SquidModel
from .squid import critical_behavior, critical_control
