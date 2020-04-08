# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Plot the magneto conductance minimum position and amplitude.

"""
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.quantum_hall.wal.universal import (
    compute_trajectory_traces_no_zeeman,
    wal_magneto_conductance,
)
from shabanipy.quantum_hall.wal.universal.trajectories import get_trajectory_data
from shabanipy.quantum_hall.wal.universal.utils import (
    cubic_soi_to_cubic_theta,
    linear_soi_to_linear_theta,
)

# --- Parameters -----------------------------------------------------------------------

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

#: Spin coherence length in unit of the mean free path
L_PHI = 1.0

#: The next three parameters expect (value, (min, max, points))
#: Value is used to generate a plot in which the other two SOI parameters are ramped,
#: min and max are the extrema values for the ramps using logarithmiccally spaced points

#: Rasha spin orbit term in angle per mean free path.
#: linear_soi_to_linear_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
RASHA_SOI = (pi / 4, (pi / 100, pi, 3))

#: Linear Dresselhaus spin orbit term in angle per mean free path.
#: linear_soi_to_linear_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
LIN_DRESSELHAUS_SOI = (pi / 4, (pi / 100, pi, 3))

#: Cubic Dresselhaus spin orbit term in angle per mean free path.
#: cubic_soi_to_cubic_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
CUBIC_DRESSELHAUS_SOI = (pi / 4, (pi / 100, pi, 3))

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

# --- Calculation ----------------------------------------------------------------------

# Plot between 1/100 and 10 B_tr which sets the validity of the diffusion approximation
fields = 10 ** np.linspace(-2, 1, 301)
index, l, c_phi, c_3phi, s_phi, s_3phi = get_trajectory_data("trace", TRAJECTORY_NUMBER)
surfaces, lengths, cosj = get_trajectory_data("mag", TRAJECTORY_NUMBER)

for i, (x, y, v, x_lab, y_lab) in enumerate(
    zip(
        (RASHA_SOI[1], RASHA_SOI[1], LIN_DRESSELHAUS_SOI[1]),
        (LIN_DRESSELHAUS_SOI[1], CUBIC_DRESSELHAUS_SOI[1], CUBIC_DRESSELHAUS_SOI[1]),
        (CUBIC_DRESSELHAUS_SOI[0], LIN_DRESSELHAUS_SOI[0], RASHA_SOI[0]),
        ("α", "α", "β$_1$"),
        ("β$_1$", "β$_3$", "β$_3$"),
    )
):

    print(f"Generating plot for {x_lab} and {y_lab}")
    xarr = np.logspace(np.log10(x[0]), np.log10(x[1]), x[2])
    yarr = np.logspace(np.log10(x[0]), np.log10(x[1]), x[2])
    loc = np.empty((len(xarr), len(yarr)))
    ampl = np.empty_like(loc)

    for j, x_ in enumerate(xarr):
        for k, y_ in enumerate(yarr):
            if i == 0:
                alpha, beta1, beta3 = x_, y_, v
            elif i == 1:
                alpha, beta1, beta3 = x_, v, y_
            elif i == 2:
                alpha, beta1, beta3 = v, x_, y_

            T = compute_trajectory_traces_no_zeeman(
                index,
                l,
                c_phi,
                c_3phi,
                s_phi,
                s_3phi,
                alpha,
                beta1,
                beta3,
                TRAJECTORY_NUMBER,
            )

            y = np.empty_like(fields)
            for n, field in enumerate(fields):
                y[n] = wal_magneto_conductance(field, L_PHI, T, surfaces, lengths, cosj)

            min_index = np.argmin(y)
            loc[j, k] = fields[min_index]
            ampl[j, k] = y[0] - y[min_index]

    f, axes = plt.subplots(1, 2, constrained_layout=True)
    im = axes[0].contourf(xarr, yarr, loc)
    cb = f.colorbar(im, ax=axes[0])
    cb.set_label("Field B$_{min}$ at which σ is minimal (B$_{tr}$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(x_lab)
    axes[0].set_ylabel(y_lab)
    im = axes[1].contourf(xarr, yarr, ampl)
    cb = f.colorbar(im, ax=axes[1])
    cb.set_label("σ(0) - σ(B$_{min}$ (e^2/h)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(x_lab)
    axes[1].set_ylabel(y_lab)

plt.show()
