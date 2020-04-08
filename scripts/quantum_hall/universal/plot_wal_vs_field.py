# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# --------------------------------------------------------------------------------------
"""Plot conductance for different values of the spin-orbit length.

This calculation is done in the absence of an in-plane Zeeman field.

"""
from collections.abc import Iterable
from itertools import chain
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from shabanipy.quantum_hall.wal.universal import (
    wal_magneto_conductance,
    compute_trajectory_traces_no_zeeman,
)
from shabanipy.quantum_hall.wal.universal.utils import (
    linear_soi_to_linear_theta,
    cubic_soi_to_cubic_theta,
)
from shabanipy.quantum_hall.wal.universal.trajectories import get_trajectory_data


# --- Parameters -----------------------------------------------------------------------

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

#: The next four parameters can be either a single number or a list. For each
#: parameters which is a list a different subplot will be created and the first
#: parameters of the other listed parameters will be used.

#: Spin coherence length in unit of the mean free path
L_PHI = 1

#: Rasha spin orbit term in angle per mean free path
#: linear_soi_to_linear_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
RASHA_SOI = [pi / 8, pi / 4, pi / 2]

#: Linear Dresselhaus spin orbit term in angle per mean free path
#: linear_soi_to_linear_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
LIN_DRESSELHAUS_SOI = pi / 20

#: Cubic Dresselhaus spin orbit term in angle per mean free path
#: cubic_soi_to_cubic_theta can used to perform the conversion using kf and the mean
#: free time both. All parameters should be in SI units.
CUBIC_DRESSELHAUS_SOI = pi / 10

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

# --- Calculation ----------------------------------------------------------------------

parameters = (L_PHI, RASHA_SOI, LIN_DRESSELHAUS_SOI, CUBIC_DRESSELHAUS_SOI)
swept = [isinstance(p, Iterable) for p in parameters]

number_of_sweep = len([s for s in swept if s])
if number_of_sweep == 4:
    f, axes = plt.subplots(2, 2, constrained_layout=True)
    axes = list(chain(*axes))
elif number_of_sweep in (2, 3):
    f, axes = plt.subplots(1, len(swept), constrained_layout=True)
else:
    f, axes = plt.subplots(1, 1, constrained_layout=True)
    axes = [axes]

for ax in axes:
    ax.set(
        xlabel="$B/B_t$", ylabel=r"$\sigma ($\frac{e^2}{2\,π\,\hbar})$", xscale="log",
    )

# Plot between 1/100 and 10 B_tr which sets the validity of the diffusion approximation
fields = 10 ** np.linspace(-2, 1, 301)
index, l, c_phi, c_3phi, s_phi, s_3phi = get_trajectory_data("trace", TRAJECTORY_NUMBER)
surfaces, lengths, cosj = get_trajectory_data("mag", TRAJECTORY_NUMBER)

i = 0
labels = ["L$_\phi$", "α", "$\beta_1$", "$\beta_3$"]
print("Starting calculations")
for j, (p, s) in enumerate(zip(parameters, swept)):
    if s:
        print(f"Generating plot for multiple {labels[j]}")
        for v in p:
            l_phi = v if j == 0 else (parameters[0][0] if swept[0] else parameters[0])
            alpha = v if j == 1 else (parameters[1][0] if swept[1] else parameters[1])
            beta1 = v if j == 2 else (parameters[2][0] if swept[2] else parameters[2])
            beta3 = v if j == 3 else (parameters[3][0] if swept[3] else parameters[3])

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
            for k, field in enumerate(fields):
                y[k] = wal_magneto_conductance(field, l_phi, T, surfaces, lengths, cosj)

            axes[i].plot(fields, y, label=labels[j] + f"= {v:g}")

        axes[i].legend()
        i += 1

plt.show()
