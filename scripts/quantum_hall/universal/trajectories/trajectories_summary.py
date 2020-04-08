# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Summarize which trajectories agree with the paper and which disagree.

"""
import matplotlib.pyplot as plt

from shabanipy.quantum_hall.wal.universal.trajectories import get_trajectory_data


with get_trajectory_data("all") as f:

    # Number of trajectories
    n_valid = len(f["valid"])
    n_invalid = len(f["invalid"])
    n_tot = n_valid + n_invalid

    print(
        f"Among {n_tot} trajectories reported in Sawada et al., our calculations "
        f"match for {n_valid}, but disagree with {n_invalid}.\n"
    )

    # Aggregate total length and cosj to plot distance to expected value
    # Also identify how many trajectories have a different number of scattering
    n = []
    l = []
    cosj = []
    nscat = 0
    for g in f["invalid"]:
        group = f["invalid"][g]
        if group.attrs["n_scat"] != group.attrs["calculated_n_scat"]:
            nscat += 1
            continue

        n.append(int(group.name[group.name.index("=") + 1 :]))
        l.append(g.attrs["length"] - group.attrs["calculated_length"])
        cosj.append(group.attrs["length"] - group.attrs["calculated_length"])

if l:

    f, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    ax1.bar(n, l)
    ax1.axhline(0.005, color="k", linestyle="--")
    ax1.axhline(-0.005, color="k", linestyle="--")
    ax1.set_xlabel("Trajectory number")
    ax1.set_ylabel("L$_{paper}$ - L$_{calculated}$")

    ax2.bar(n, cosj)
    ax2.axhline(5e-6, color="k", linestyle="--")
    ax2.axhline(-5e-6, color="k", linestyle="--")
    ax2.set_xlabel("Trajectory number")
    ax2.set_ylabel("cosj$_{paper}$ - cosj$_{calculated}$")

    plt.show()

else:
    print(
        "No trajectory with the right number of scattering disagree based on "
        "the total length or final angle."
    )
