# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Use our implementation to replot the figure presented in:

A. Sawada, T. Koga, Universal modeling of weak antilocalization corrections in
quasi-two-dimensional electron systems using predetermined return orbitals.
Phys. Rev. E. 95, 023309 (2017).

"""
# Allow to enable disable the plotting of a figure
PLOT_FIGURE_2 = True
PLOT_FIGURE_3 = True
PLOT_FIGURE_6 = True
PLOT_FIGURE_7 = True

# Use a subset of the trajectories (Use None to use all trajectories).
N_TRAJECTORIES = 40000

# Number of points to use for field axis
FIELD_POINTS = 81

# Number of points to use for rashba parameter axis
RASHBA_POINTS = 41

import matplotlib.pyplot as plt
import numpy as np

from shabanipy.quantum_hall.wal.universal import (
    compute_trajectory_traces_no_zeeman,
    wal_magneto_conductance,
)
from shabanipy.quantum_hall.wal.universal.trajectories import (
    get_detailed_trajectory_data,
    get_summary_trajectory_data,
)

# Get the trajectory parameters
print("Loading trajectory parameters")
index, lengths, c_phi, c_3phi, s_phi, s_3phi = get_detailed_trajectory_data(
    trajectory_number=N_TRAJECTORIES
)
total_lengths, surfaces, cosjs = get_summary_trajectory_data(
    trajectory_number=N_TRAJECTORIES
)

# Figure 2: Rashba only compared to Golub
if PLOT_FIGURE_2:
    print("Generating Figure 2")
    fig, axes = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=True)
    l_phi = 10
    fields = np.logspace(-2, 1, FIELD_POINTS)
    rashbas = b = np.logspace(-2, 1, RASHBA_POINTS)
    pr, pf = np.meshgrid(fields, rashbas)
    dsigma = np.empty_like(pf)
    dsigma_0 = np.empty_like(rashbas)
    for i, r in enumerate(rashbas):
        trajs = compute_trajectory_traces_no_zeeman(
            index,
            lengths,
            c_phi,
            c_3phi,
            s_phi,
            s_3phi,
            r * np.pi,
            0,
            0,
            index.shape[0],
        )
        # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
        dsigma[i] = wal_magneto_conductance(
            fields, l_phi, trajs, total_lengths, surfaces, cosjs
        ) / (4 * np.pi)
        dsigma_0[i] = wal_magneto_conductance(
            0.0, l_phi, trajs, total_lengths, surfaces, cosjs
        ) / (4 * np.pi)

    pc = axes[0].pcolormesh(pr, pf, dsigma, shading="nearest")
    cb = fig.colorbar(pc, ax=axes[0])
    # Corresponds to the white line at B_SO
    axes[0].plot(fields, np.sqrt(2 * fields) / np.pi, "w--")
    for v in (2, 1, 0.5, 0.25):
        axes[0].axhline(v, color="w", linestyle="--", linewidth=0.5)
    plt.clabel(axes[0].contour(pr, pf, dsigma, [-0.2, -0.1, 0.0], colors="k"))
    axes[0].set_xlabel("B/B$_{tr}$")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("θ$_R$/π")
    axes[0].set_yscale("log")
    cb.set_label("Δσ/(2e²/h)")

    for v in (2, 1, 0.5, 0.25, 0):
        t = compute_trajectory_traces_no_zeeman(
            index,
            lengths,
            c_phi,
            c_3phi,
            s_phi,
            s_3phi,
            v * np.pi,
            0,
            0,
            index.shape[0],
        )
        # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
        axes[1].plot(
            fields,
            wal_magneto_conductance(fields, l_phi, t, total_lengths, surfaces, cosjs)
            / (4 * np.pi),
            label=f"{v}π",
        )
    axes[1].set_xlabel("B/B$_{tr}$")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.35, 0.0)
    axes[1].set_ylabel("Δσ/(2e²/h)")
    axes[1].legend()

    axes[2].plot(rashbas, dsigma_0)
    axes[2].set_xlabel("θ$_R$/π")
    axes[2].set_xscale("log")
    axes[2].set_ylabel("Δσ/(2e²/h)")


# Figure 3: Persistent spin helix (Rashba = linear Dresselhaus)
if PLOT_FIGURE_3:
    print("Generating Figure 3")
    fig, axes = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=True)
    l_phi = 100
    fig.suptitle(r"L$_\phi$ =" + f" {l_phi}")
    fields = np.logspace(-2, 1, FIELD_POINTS)
    rashbas = b = np.logspace(-2, 1, RASHBA_POINTS)
    dressel3 = (0, 64, 16)
    pr, pf = np.meshgrid(fields, rashbas)
    b_min = np.empty_like(rashbas)
    for i, d in enumerate(dressel3):
        dsigma = np.empty_like(pf)
        for j, r in enumerate(rashbas):
            trajs = compute_trajectory_traces_no_zeeman(
                index,
                lengths,
                c_phi,
                c_3phi,
                s_phi,
                s_3phi,
                r * np.pi,
                r * np.pi,
                np.pi / d if d else d,
                index.shape[0],
            )
            # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
            dsigma[j] = wal_magneto_conductance(
                fields, l_phi, trajs, total_lengths, surfaces, cosjs
            ) / (4 * np.pi)
            if i == 2:
                pos = np.argmin(dsigma[j])
                b_min[j] = fields[pos] if pos else np.nan

        pc = axes[i].pcolormesh(pr, pf, dsigma, shading="nearest")
        cb = fig.colorbar(pc, ax=axes[i])
        plt.clabel(
            axes[i].contour(
                pr, pf, dsigma, sorted([-0.2, -0.3, -0.4, -0.5, -0.6]), colors="k"
            )
        )
        if i == 2:
            axes[i].set_xlabel("B/B$_{tr}$")
            axes[i].plot(b_min, rashbas, "w--")
        axes[i].set_xscale("log")
        axes[i].set_ylabel("θ$_R$/π")
        axes[i].set_yscale("log")
        axes[i].set_title(r"$\Theta$ = " + (f"π/{d}" if d else "0"))
        cb.set_label("Δσ/(2e²/h)")


# Figure 6: Rashba only (we do not plot the error estimate)
if PLOT_FIGURE_6:
    print("Generating Figure 6")
    fig, axes = plt.subplots(2, 1, figsize=(5, 9), constrained_layout=True)
    l_phis = (100, 1000)
    fields = np.logspace(-2, 1, FIELD_POINTS)
    rashbas = b = np.logspace(-2, 1, RASHBA_POINTS)
    pr, pf = np.meshgrid(fields, rashbas)
    dsigma = np.empty_like(pf)
    for i, l_phi in enumerate(l_phis):
        print(f"    Generating data with l_phi = {l_phi}")
        for j, r in enumerate(rashbas):
            trajs = compute_trajectory_traces_no_zeeman(
                index,
                lengths,
                c_phi,
                c_3phi,
                s_phi,
                s_3phi,
                r * np.pi,
                0,
                0,
                index.shape[0],
            )
            # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
            dsigma[j] = wal_magneto_conductance(
                fields, l_phi, trajs, total_lengths, surfaces, cosjs
            ) / (4 * np.pi)

        pc = axes[i].pcolormesh(pr, pf, dsigma, shading="nearest")
        cb = fig.colorbar(pc, ax=axes[i])
        # Corresponds to the white line at B_SO
        axes[i].plot(fields, np.sqrt(2 * fields) / np.pi, "w--")
        axes[i].set_title(r"L$_\phi$ = " + f"{l_phi}")
        plt.clabel(
            axes[i].contour(
                pr,
                pf,
                dsigma,
                [-0.6, -0.4, -0.2, 0.0] if i == 0 else [-0.6, -0.3, 0],
                colors="k",
            )
        )
        if i == 1:
            axes[i].set_xlabel("B/B$_{tr}$")
        axes[i].set_xscale("log")
        axes[i].set_ylabel("θ$_R$/π")
        axes[i].set_yscale("log")
        cb.set_label("Δσ/(2e²/h)")

# Figure 7: Cuts of figure 6
if PLOT_FIGURE_7:
    print("Generating Figure 7")
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
    l_phis = (100, 1000)
    fields = np.logspace(-2, 1, FIELD_POINTS)
    rashbas = b = np.logspace(-2, 1, RASHBA_POINTS)
    pr, pf = np.meshgrid(fields, rashbas)
    dsigma = np.empty_like(pf)
    dsigma_0 = np.empty_like(rashbas)
    for i, l_phi in enumerate(l_phis):
        print(f"    Generating data with l_phi = {l_phi}")
        for j, r in enumerate(rashbas):
            trajs = compute_trajectory_traces_no_zeeman(
                index,
                lengths,
                c_phi,
                c_3phi,
                s_phi,
                s_3phi,
                r * np.pi,
                0,
                0,
                index.shape[0],
            )
            # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
            dsigma_0[j] = wal_magneto_conductance(
                0.0, l_phi, trajs, total_lengths, surfaces, cosjs
            ) / (4 * np.pi)

        for v in (2, 1, 0.5, 0.25, 0):
            t = compute_trajectory_traces_no_zeeman(
                index,
                lengths,
                c_phi,
                c_3phi,
                s_phi,
                s_3phi,
                v * np.pi,
                0,
                0,
                index.shape[0],
            )
            # Need to convert to Sawada unit 2e^2/h (not Knap e^2/(2πh))
            axes[i, 0].plot(
                fields,
                wal_magneto_conductance(
                    fields, l_phi, t, total_lengths, surfaces, cosjs
                )
                / (4 * np.pi),
                label=f"{v}π",
            )

        if i == 1:
            axes[i, 0].set_xlabel("B/B$_{tr}$")
        axes[i, 0].set_xscale("log")
        axes[i, 0].set_ylim(-0.8 if i == 0 else -1.0, 0.4 if i == 0 else 0.5)
        axes[i, 0].set_ylabel("Δσ/(2e²/h)")
        axes[i, 0].legend()

        axes[i, 1].plot(rashbas, dsigma_0)
        if i == 1:
            axes[i, 1].set_xlabel("θ$_R$/π")
        axes[i, 1].set_xscale("log")
        axes[i, 1].set_ylim(-0.8 if i == 0 else -1.0, 0.4 if i == 0 else 0.5)


plt.show()
