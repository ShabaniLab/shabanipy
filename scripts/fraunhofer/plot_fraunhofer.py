# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Generate Fraunhofer patterns for different current distributions.

The first node of a Fraunhoffer will occur at 2 for a conversion factor of
π / the junction size.

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.fraunhofer.util import produce_fraunhofer, produce_fraunhofer_fast

# Current and phase distribution to use.
DISTRIBUTIONS = [
    ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),
    # ([2.5, 0, 0, 0, 2.5], [0, 0, 0, 0, np.pi/4]),
]


for d, p in DISTRIBUTIONS:
    width = 4
    d = np.array(d)/width
    d, p = np.array(d), np.array(p)
    b = np.linspace(-5, 5, 51)
    tic = time.perf_counter()
    f = produce_fraunhofer(b, np.pi/4, width, d, p, method="quad")
    print(time.perf_counter() - tic)
    plt.plot(b, f)
    tic = time.perf_counter()
    f = produce_fraunhofer_fast(b, np.pi/4, width, d, p, 2**10+1)
    print(time.perf_counter() - tic)
    plt.plot(b, f)

plt.legend()
plt.show()
