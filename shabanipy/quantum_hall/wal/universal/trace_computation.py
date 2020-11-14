# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to compute the trace of the evolution matrix along a trajectory.

"""
from math import cos, exp, pi

import numpy as np
from numba import njit, prange, generated_jit, types


@generated_jit(nopython=True, fastmath=True)
def _compute_single_trajectory_trace_no_zeeman(
    l: np.ndarray,
    c_phi: np.ndarray,
    c_3phi: np.ndarray,
    s_phi: np.ndarray,
    s_3phi: np.ndarray,
    theta_alpha: float,
    theta_beta1: float,
    theta_beta3: float,
) -> float:
    """Find the trace of the matrix R_tot^2 for a single trajectory

    Parameters
    ----------
    l : np.ndarray
        (n_scat) array, length of each segment
    c_phi : np.ndarray
        Cosinus of the angle of the trajectory on each segment.
    c_3phi : np.ndarray
        Cosinus of the triple angle of the trajectory on each segment.
    s_phi : np.ndarray
        Sinus of the angle of the trajectory on each segment.
    s_3phi : np.ndarray
        Sinus of the triple angle of the trajectory on each segment.
    theta_alpha : float
        Rashba SOI induced rotation per unit length
    theta_beta3 : float
        Cubic Dresselhaus induced rotation per unit length
    theta_beta1 : float
        Linear Dresselhaus induced rotation per unit length

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    if l.dtype == types.float32:
        fcast = np.float32
        ccast = np.complex64
        inner_cdtype = np.dtype("complex64")
    else:
        fcast = np.float64
        ccast = np.complex128
        inner_cdtype = np.dtype("complex128")

    def _inner(
        l: np.ndarray,
        c_phi: np.ndarray,
        c_3phi: np.ndarray,
        s_phi: np.ndarray,
        s_3phi: np.ndarray,
        theta_alpha: float,
        theta_beta1: float,
        theta_beta3: float,
    ):
        rotations = np.empty((len(l), 2, 2), dtype=inner_cdtype)

        # Necessary cast to avoid upcasting to 64 bits
        theta_alpha = fcast(theta_alpha)
        theta_beta1 = fcast(theta_beta1)
        theta_beta3 = fcast(theta_beta3)
        B_x = theta_alpha * s_phi + theta_beta3 * c_3phi + theta_beta1 * c_phi
        B_y = -theta_alpha * c_phi + theta_beta3 * s_3phi - theta_beta1 * s_phi
        B = np.sqrt(B_x ** 2 + B_y ** 2)
        theta = B * l
        # Necessary cast to avoid upcasting to 64 bits
        c_theta = np.cos(fcast(0.5) * theta)
        s_theta = np.sin(fcast(0.5) * theta)

        psi1 = np.empty(len(l), dtype=inner_cdtype)
        psi2 = np.empty(len(l), dtype=inner_cdtype)
        for i, (b, bx, by) in enumerate(zip(B, B_x, B_y)):
            if b != 0:
                # Necessary cast to avoid upcasting to 128 bits
                psi1[i] = -ccast(1j) * (bx / b + ccast(1j) * by / b)
                psi2[i] = -ccast(1j) * (bx / b - ccast(1j) * by / b)
            else:
                psi1[i] = psi2[i] = 0

        rotations[:, 0, 0] = c_theta
        rotations[:, 0, 1] = psi1 * s_theta
        rotations[:, 1, 0] = psi2 * s_theta
        rotations[:, 1, 1] = c_theta

        # For 2x2 matrices calling BLAS matrix multiplication has a large overhead
        # and the need to allocate the output matrix is likely to cause issue with
        # parallelization of the code.
        cw_rot = np.array([[1, 0], [0, 1]], dtype=inner_cdtype)
        for i in range(0, len(l)):
            # equivalent to cw_rot = r @ cw_rot
            r = rotations[i]
            a = r[0, 0] * cw_rot[0, 0] + r[0, 1] * cw_rot[1, 0]
            b = r[0, 0] * cw_rot[0, 1] + r[0, 1] * cw_rot[1, 1]
            c = r[1, 0] * cw_rot[0, 0] + r[1, 1] * cw_rot[1, 0]
            d = r[1, 0] * cw_rot[0, 1] + r[1, 1] * cw_rot[1, 1]
            cw_rot[0, 0] = a
            cw_rot[0, 1] = b
            cw_rot[1, 0] = c
            cw_rot[1, 1] = d

        return (
            cw_rot[0, 0] * cw_rot[0, 0]
            + cw_rot[0, 1] * cw_rot[1, 0]
            + cw_rot[1, 0] * cw_rot[0, 1]
            + cw_rot[1, 1] * cw_rot[1, 1]
        ).real

    return _inner  # type: ignore


@njit(fastmath=True, parallel=True)
def compute_trajectory_traces_no_zeeman(
    index: np.ndarray,
    l: np.ndarray,
    c_phi: np.ndarray,
    c_3phi: np.ndarray,
    s_phi: np.ndarray,
    s_3phi: np.ndarray,
    theta_alpha: float,
    theta_beta1: float,
    theta_beta3: float,
    N_orbit: int,
) -> np.ndarray:
    """Compute the trace of the evolution operator for different trajectories.

    This is run in parallel in batches of 1000.

    Parameters
    ----------
    index : np.ndarray
        (n_scat, 2) array, with the beginning and end index for each trajectory
    l : np.ndarray
        (n_scat) array, length of each segment
    c_phi : np.ndarray
        Cosinus of the angle of the trajectory on each segment.
    c_3phi : np.ndarray
        Cosinus of the triple angle of the trajectory on each segment.
    s_phi : np.ndarray
        Sinus of the angle of the trajectory on each segment.
    s_3phi : np.ndarray
        Sinus of the triple angle of the trajectory on each segment.
    theta_alpha : float
        Rashba SOI induced rotation per unit length
        (see utils for conversion functions)
    theta_beta1 : float
        Cubic Dresselhaus induced rotation per unit length
        (see utils for conversion functions)
    theta_beta3 : float
        Linear Dresselhaus induced rotation per unit length
        (see utils for conversion functions)
    N_orbit : int
        Number of orbitals to use in the calculations.

    Returns
    -------
    traces : np.ndarray
        1D array of the trace of each trajectory.

    """
    T = np.empty(N_orbit)
    for n in prange(N_orbit // 1000):
        r = N_orbit % 1000 if n * 1000 + 999 >= N_orbit else 1000
        for i in range(r):
            traj_id = n * 1000 + i
            begin, end = index[traj_id]
            T_a = _compute_single_trajectory_trace_no_zeeman(
                l[begin:end],
                c_phi[begin:end],
                c_3phi[begin:end],
                s_phi[begin:end],
                s_3phi[begin:end],
                theta_alpha,
                theta_beta1,
                theta_beta3,
            )
            T[traj_id] = T_a

    return T


@njit(fastmath=True, parallel=True)
def _compute_single_trajectory_trace_zeeman(
    l: np.ndarray,
    c_phi: np.ndarray,
    c_3phi: np.ndarray,
    s_phi: np.ndarray,
    s_3phi: np.ndarray,
    theta_alpha: float,
    theta_beta1: float,
    theta_beta3: float,
    B_zx: float,
    B_zy: float,
) -> float:
    """Find the trace of the matrix R_tot^2 in the presence of an in-plane Zeeman field.

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    c_phi: np.ndarray
        Cosinus of the angle of the trajectory on each segment.
    c_3phi: np.ndarray
        Cosinus of the triple angle of the trajectory on each segment.
    s_phi: np.ndarray
        Sinus of the angle of the trajectory on each segment.
    s_3phi: np.ndarray
        Sinus of the triple angle of the trajectory on each segment.
    theta_alpha: float
        Rashba SOI induced rotation per unit length
    theta_beta3: float
        Cubic Dresselhaus induced rotation per unit length
    theta_beta1: float
        Linear Dresselhaus induced rotation per unit length
    B_zx : float
        Rotation induced by the Zeeman field in x by unit length
    B_zy : float
        Rotation induced by the Zeeman field in y by unit length

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    if l.dtype == types.float32:
        fcast = np.float32
        ccast = np.complex64
        inner_cdtype = np.dtype("complex64")
    else:
        fcast = np.float64
        ccast = np.complex128
        inner_cdtype = np.dtype("complex128")

    def _inner(
        l: np.ndarray,
        c_phi: np.ndarray,
        c_3phi: np.ndarray,
        s_phi: np.ndarray,
        s_3phi: np.ndarray,
        theta_alpha: float,
        theta_beta3: float,
        theta_beta1: float,
        B_zx: float,
        B_zy: float,
    ):
        # Computation for the clockwise trajectory
        rotations = np.empty((len(l), 2, 2), dtype=inner_cdtype)

        # Necessary cast to avoid upcasting to 64 bits
        theta_alpha = fcast(theta_alpha)
        theta_beta1 = fcast(theta_beta1)
        theta_beta3 = fcast(theta_beta3)
        B_zx = fcast(B_zx)
        B_zy = fcast(B_zy)
        B_x_cw = theta_alpha * s_phi + theta_beta3 * c_3phi + theta_beta1 * c_phi + B_zx
        B_y_cw = (
            -theta_alpha * c_phi + theta_beta3 * s_3phi - theta_beta1 * s_phi + B_zy
        )
        B_cw = np.sqrt(B_x_cw ** 2 + B_y_cw ** 2)
        theta_cw = B_cw * l
        # Necessary cast to avoid upcasting to 64 bits
        c_theta_cw = np.cos(fcast(0.5) * theta_cw)
        s_theta_cw = np.sin(fcast(0.5) * theta_cw)

        psi1_cw = np.empty(len(l), dtype=inner_cdtype)
        psi2_cw = np.empty(len(l), dtype=inner_cdtype)
        for i, (b, bx, by) in enumerate(zip(B_cw, B_x_cw, B_y_cw)):
            if b != 0:
                # Necessary cast to avoid upcasting to 128 bits
                psi1_cw[i] = -ccast(1j) * (bx / b + ccast(1j) * by / b)
                psi2_cw[i] = -ccast(1j) * (bx / b - ccast(1j) * by / b)
            else:
                psi1_cw[i] = psi2_cw[i] = 0

        rotations[:, 0, 0] = c_theta_cw
        rotations[:, 0, 1] = psi1_cw * s_theta_cw
        rotations[:, 1, 0] = psi2_cw * s_theta_cw
        rotations[:, 1, 1] = c_theta_cw

        # For 2x2 matrices calling BLAS matrix multiplication has a large overhead
        # and the need to allocate the output matrix is likely to cause issue with
        # parallelization of the code.
        cw_rot = np.array([[1, 0], [0, 1]], dtype=inner_cdtype)
        for i in range(0, len(l)):
            # equivalent to cw_rot = r @ cw_rot
            r = rotations[i]
            a = r[0, 0] * cw_rot[0, 0] + r[0, 1] * cw_rot[1, 0]
            b = r[0, 0] * cw_rot[0, 1] + r[0, 1] * cw_rot[1, 1]
            c = r[1, 0] * cw_rot[0, 0] + r[1, 1] * cw_rot[1, 0]
            d = r[1, 0] * cw_rot[0, 1] + r[1, 1] * cw_rot[1, 1]
            cw_rot[0, 0] = a
            cw_rot[0, 1] = b
            cw_rot[1, 0] = c
            cw_rot[1, 1] = d

        # Computation for the counter clock wise trajectory

        B_x_ccw = (
            -theta_alpha * s_phi - theta_beta3 * c_3phi - theta_beta1 * c_phi + B_zx
        )
        B_y_ccw = (
            theta_alpha * c_phi - theta_beta3 * s_3phi + theta_beta1 * s_phi + B_zy
        )
        B_ccw = np.sqrt(B_x_ccw ** 2 + B_y_ccw ** 2)
        theta_ccw = B_ccw * l
        # Necessary cast to avoid upcasting to 64 bits
        c_theta_ccw = np.cos(fcast(0.5) * theta_ccw)
        s_theta_ccw = np.sin(fcast(0.5) * theta_ccw)

        psi1_ccw = np.empty(len(l), dtype=inner_cdtype)
        psi2_ccw = np.empty(len(l), dtype=inner_cdtype)
        for i, (b, bx, by) in enumerate(zip(B_ccw, B_x_ccw, B_y_ccw)):
            if b != 0:
                # Necessary cast to avoid upcasting to 128 bits
                psi1_ccw[i] = -ccast(1j) * (bx / b + ccast(1j) * by / b)
                psi2_ccw[i] = -ccast(1j) * (bx / b - ccast(1j) * by / b)
            else:
                psi1_ccw[i] = psi2_ccw[i] = 0

        rotations[:, 0, 0] = c_theta_ccw
        rotations[:, 0, 1] = psi1_ccw * s_theta_ccw
        rotations[:, 1, 0] = psi2_ccw * s_theta_ccw
        rotations[:, 1, 1] = c_theta_ccw
        ccw_rot = np.array([[1, 0], [0, 1]], dtype=inner_cdtype)
        for i in range(len(l) - 1, -1, -1):
            # equivalent to ccw_rot = r @ ccw_rot
            r = rotations[i]
            a = r[0, 0] * ccw_rot[0, 0] + r[0, 1] * ccw_rot[1, 0]
            b = r[0, 0] * ccw_rot[0, 1] + r[0, 1] * ccw_rot[1, 1]
            c = r[1, 0] * ccw_rot[0, 0] + r[1, 1] * ccw_rot[1, 0]
            d = r[1, 0] * ccw_rot[0, 1] + r[1, 1] * ccw_rot[1, 1]
            ccw_rot[0, 0] = a
            ccw_rot[0, 1] = b
            ccw_rot[1, 0] = c
            ccw_rot[1, 1] = d

        return (
            ccw_rot[0, 0].conjugate() * cw_rot[0, 0]
            + ccw_rot[1, 0].conjugate() * cw_rot[1, 0]
            + ccw_rot[0, 1].conjugate() * cw_rot[0, 1]
            + ccw_rot[1, 1].conjugate() * cw_rot[1, 1]
        ).real

    return _inner  # type: ignore


@njit(fastmath=True, parallel=True)
def compute_trajectory_traces_zeeman(
    index,
    l,
    c_phi,
    c_3phi,
    s_phi,
    s_3phi,
    theta_alpha,
    theta_beta1,
    theta_beta3,
    N_orbit,
    B_zx,
    B_zy,
):
    """Compute the trace of the evolution operator for different trajectories.

    This is run in parallel in batches of 1000.

    Parameters
    ----------
    index : np.ndarray
        (n_scat, 2) array, with the beginning and end index for each trajectory
    l : np.ndarray
        (n_scat) array, length of each segment
    c_phi : np.ndarray
        Cosinus of the angle of the trajectory on each segment.
    c_3phi : np.ndarray
        Cosinus of the triple angle of the trajectory on each segment.
    s_phi : np.ndarray
        Sinus of the angle of the trajectory on each segment.
    s_3phi : np.ndarray
        Sinus of the triple angle of the trajectory on each segment.
    theta_alpha : float
        Rashba SOI induced rotation per unit length
    theta_beta3 : float
        Cubic Dresselhaus induced rotation per unit length
    theta_beta1 : float
        Linear Dresselhaus induced rotation per unit length
    N_orbit : int
        Number of orbitals to use in the calculations.
    B_zx : float
        Rotation induced by the Zeeman field in x by unit length
    B_zy : float
        Rotation induced by the Zeeman field in y by unit length

    Returns
    -------
    traces : np.ndarray
        1D array of the trace of each trajectory.

    """
    T = np.empty(N_orbit)
    for n in prange(N_orbit // 1000):
        for i in range(1000):
            traj_id = n * 1000 + i
            if traj_id >= N_orbit:
                break
            begin, end = index[traj_id]
            T_a = find_trace_Zeeman(
                l[begin:end],
                c_phi[begin:end],
                c_3phi[begin:end],
                s_phi[begin:end],
                s_3phi[begin:end],
                theta_alpha,
                theta_beta1,
                theta_beta3,
                B_zx,
                B_zy,
            )
            T[traj_id] = T_a

    return T
