#!python3
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
allows to take into account both linear Dresselhaus and Rashba terms.

"""
import numpy as np
from numba import njit


def compute_wal_conductance_difference(field, dephasing_field,
                                       linear_soi_rashba, linear_soi_dressel,
                                        cubic_soi, low_field_reference,
                                       matrix_truncation=100):
    """Difference in conductance due to SOI compared to a reference field value

    F. G. Pikus, G. E. Pikus, Conduction-band spin splitting and negative
    magnetoresistance in A2B5 heterostructures. Phys. Rev. B. 51, 16928–16935
    (1995).

    Note: There is a typo in Eq. 26 of this paper. A corrected version of this
    formula is available in W. Knap et al. (1996). Weak antilocalization and spin
    precession in quantum wells. Phys. Rev. B. 53(7), 3912.

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
    linear_soi_dressel : float
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
    index = np.tile(np.arange(0, matrix_truncation), (len(field), 1)).T
    j0_sum = np.sum(1.0 / (index + 0.5 + dephasing_field / field) - 1.0 / (index + 1), axis=0)

    # Compute the j=1 term of the sum in eq 27
    h = fill_hamiltonian(field, abs(dephasing_field),
                         abs(linear_soi_rashba), abs(linear_soi_dressel),
                         abs(cubic_soi), matrix_truncation)
    print('starting eigen values solving')
    eigenvalues = np.linalg.eigvalsh(h)
    print('done computing')
    j1_sum = np.sum(1.0 / eigenvalues - 1.0 / (np.tile(index.T, 3) + 1), axis=1)

    # subtract the reference field value from eq 27
    sigma = (j0_sum[1:] - j1_sum[1:]) - (j0_sum[0] - j1_sum[0]) + 2*np.log(field[1:]/field[0])
    return sigma


@njit(cache=True, fastmath=True)
def fill_hamiltonian(field, dephasing_field,
                     linear_soi_rashba, linear_soi_dressel,
                     cubic_soi, matrix_truncation=100):
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
    linear_soi_dressel : float
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
    shape = (len(field), 3 * matrix_truncation, 3 * matrix_truncation)
    h = np.zeros(shape, dtype=np.complex128)

    # Useful matrix representation to relate eq 26 to the following operation.
    # These correspond to the total angular momentum of J = 1 for a spin triplet.
    identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Jz2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    Jp = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # = (J_x + iJ_y)/sqrt(2)
    Jm = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # = (J_x - iJ_y)/sqrt(2)
    Jp2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    Jm2 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

    # Term appearing on the diagonal of the Hamiltonian
    z_term = (linear_soi_rashba + linear_soi_dressel + cubic_soi) / field
    # Term proportional to J+^2, J-^2
    j2_term = -1j * 2 * np.sqrt(linear_soi_rashba * linear_soi_dressel) / field
    # Rashba off-diagonal term
    r_off_term = 1j * np.sqrt(2 * linear_soi_rashba / field)
    # Dresselhaus off-diagonal term
    d_off_term = np.sqrt(2 * linear_soi_dressel / field)

    for field_index, (phi, z, j2, r, d) in enumerate(zip(dephasing_field / field,
                                                         z_term, j2_term,
                                                         r_off_term, d_off_term)):
        for n in range(matrix_truncation):
            h[field_index, 3 * n: 3 * (n + 1), 3 * n: 3 * (n + 1)] = (n + 0.5 + phi) * identity + z * (
                    2 * identity - Jz2) + j2 * (Jp2 - Jm2)
        for n in range(matrix_truncation - 1):
            r_off_n = r * np.sqrt(n + 1)
            d_off_n = d * np.sqrt(n + 1)
            # Upper diagonal going from n+1 to n, ie terms in a
            h[field_index, 3 * n: 3 * (n + 1), 3 * (n + 1): 3 * (n + 2)] = - d_off_n * Jp - r_off_n * Jm
            # Lower diagonal going from n to n+1, ie terms in a^{dag}
            h[field_index, 3 * (n + 1): 3 * (n + 2), 3 * n: 3 * (n + 1)] = - d_off_n * Jm + r_off_n * Jp

    return h

