# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak antilocalisation analysis routines.

This routines relies on diagonalizing the full Landau-levels Hamiltonian which
allows to take into account both Dresselhaus and Rashba terms.

"""
import numpy as np
from numba import njit


def compute_wal_conductance_difference(field, dephasing_field,
                                       linear_soi_rashba, linear_soi_dressel,
                                       cubic_soi, low_field_reference,
                                       matrix_truncation=500):
    """Difference in conductance due to SOI compared to a reference field value

    F. G. Pikus, G. E. Pikus, Conduction-band spin splitting and negative
    magnetoresistance in A2B5 heterostructures. Phys. Rev. B. 51, 16928–16935
    (1995).

    Parameters
    ----------
    field : float | np.array
        Magnetic field at which to compute the conductance.
    dephasing_field : float
        Dephasing field used in the model. The unit should the same as the one
        used to express the magnetic field.
    linear_soi_rashba : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling of Rashba origin. The unit should the same as the one used to
        express the magnetic field.
    linear_soi_dress : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling of Dresselhaus. The unit should the same as the one used to
        express the magnetic field.
    cubic_soi : float
        Field describing the contribution of the cubic term arising from
        Dresselhaus SOI. The unit should the same as the one used to express
        the magnetic field.
    low_field_reference : float
        Reference field at which to compute the conductance difference.
    matrix_truncation : int, optional
        Number of Landau level to consider in the system. This should be
        larger than δ τ_1 according to the notation of the cited paper.

    """
    # Add the reference field to the values that should be computed
    field = np.concatenate((np.array([low_field_reference]), np.array(field)))

    # Compute the j=0 term of the sum in eq 27
    index = np.tile(np.arange(1, matrix_truncation + 1), (len(field), 1)).T
    j0_sum = np.sum(1/(index + 0.5 + dephasing_field/field),
                    axis=0)

    # Compute the j=1 term of the sum in eq 27
    h = fill_hamiltonian(field, abs(dephasing_field),
                         abs(linear_soi_rashba), abs(linear_soi_dressel),
                         abs(cubic_soi), matrix_truncation)
    print('starting eigen values solving')
    eigs = np.linalg.eigvalsh(h)
    print('done computing')
    j1_sum = np.sum(1/eigs, axis=1)


    # substract the reference field value and take into account the log term
    # from 27
    return (j0_sum[1:] - j0_sum[0] - j1_sum[1:] + j1_sum[0] +
            np.log(field[1:]/field[0]))


@njit(cache=True, fastmath=True)
def fill_hamiltonian(field, dephasing_field,
                     linear_soi_rashba, linear_soi_dressel,
                     cubic_soi, matrix_truncation=1000):
    """[summary]

    Parameters
    ----------
    field : np.ndarray
        1D array containing teh field at which to compute the Hamiltonian.
    dephasing_field : float
        Dephasing field used in the model. The unit should the same as the one
        used to express the magnetic field.
    linear_soi_rashba : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling of Rashba origin. The unit should the same as the one used to
        express the magnetic field.
    linear_soi_dress : float
        Field describing the contribution of the linear term in the spin-orbit
        coupling of Dresselhaus. The unit should the same as the one used to
        express the magnetic field.
    cubic_soi : float
        Field describing the contribution of the cubic term arising from
        Dresselhaus SOI. The unit should the same as the one used to express
        the magnetic field.
    matrix_truncation : int, optional
        Number of Landau level to consider in the system. This should be
        larger than δ τ_1 according to the notation of the cited paper.

    """
    # To speed up the computation we compute the eigenvalues of the matrix
    # describing the system for all considered fields at once by stacking
    # them. We consider always at least two fields: the point of interest and
    # the reference field.
    shape = (len(field), 3*matrix_truncation, 3*matrix_truncation)
    h = np.zeros(shape, dtype=np.complex128)

    # Useful matrix representation to relate eq 26 to the following operation.
    # Jz^2 = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
    # J+ = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    # J- = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    # J+^2 = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    # J-^2 = [[0, 0, 0], [0, 0, 0], [1, 0, 0]]

    # Term appearing on the diagonal of the Hamiltonian
    z_term = (linear_soi_dressel + linear_soi_rashba + cubic_soi)/field
    # Term proportional to J+^2, J-^2
    j2_term = 1j*2*np.sqrt(linear_soi_rashba*linear_soi_dressel)/field
    # Rashba off-diagonal term
    r_off_term = np.sqrt(2*linear_soi_rashba)/field
    # Dresselhaus off-diagonal term
    d_off_term = 1j*np.sqrt(2*linear_soi_dressel)/field

    for i, (phi, z, j2, r, d) in enumerate(zip(dephasing_field/field,
                                               z_term, j2_term,
                                               r_off_term, d_off_term)):
        for j in range(matrix_truncation):
            h[i, 3*j, 3*j]     = j + 0.5 + phi + z
            h[i, 3*j, 3*j+2]   = -j2
            h[i, 3*j+1, 3*j+1] = j + 0.5 + phi + 2*z
            h[i, 3*j+2, 3*j+2] = j + 0.5 + phi + z
            h[i, 3*j+2, 3*j]   = j2
        for j in range(matrix_truncation-1):
            sqrtn = np.sqrt(j+1)
            r_off_n = r * sqrtn
            d_off_n = d * sqrtn
            # Upper diagonal going from j+1 to j, ie terms in a
            h[i, 3*j, 3*j+4]   = -r_off_n
            h[i, 3*j+1, 3*j+3] = - d_off_n
            h[i, 3*j+1, 3*j+5] = -r_off_n
            h[i, 3*j+2, 3*j+4] = - d_off_n

            # Lower diagonal going from j to j+1, ie terms in a^{dag}
            h[i, 3*j+3, 3*j+1]   = d_off_n
            h[i, 3*j+4, 3*j] = r_off_n
            h[i, 3*j+4, 3*j+2] = d_off_n
            h[i, 3*j+5, 3*j+1] = r_off_n

    return h
