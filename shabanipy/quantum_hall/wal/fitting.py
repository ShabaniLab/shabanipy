# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak anti-localisation analysis main fitting routine.

"""
import scipy.constants as cs
import matplotlib.pyplot as plt

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_no_dresselhaus \
    import compute_wal_conductance_difference as simple_wal
from .wal_full_diagonalization \
    import compute_wal_conductance_difference as full_wal


def extract_soi_from_wal(field, r, reference_field,
                         dmodel='full', truncation=1000,
                         guess_from_previous=True, guesses=None,
                         plot_fit=False):
    """Extract the SOI parameters from fitting the wal conductance.

    This algorithm assumes that the data are properly centered.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values for which the the resistance was measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    r : np.ndarray
        Resistance values in Ω which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    reference_field :
    model : {'full', 'simplified'}, optional
        Model used to describe the WAL. 'simplified' corresponds to the situation
        in which either the Rashba term or the linear Dresselhaus term can be
        neglected. 'full' corresponds to a more complete model, however each
        evaluation of the fitting function requires to find the eigenvalues
        of a large and as a consequence may be slow.
    truncation : int, optional
        Both models imply a truncation of the number of Landau levels
        considered: in the 'simplified' case this enters the evaluation of a
        series, in the 'full' case the size of the matrix whose eigenvalues
        need to be computed.
    guess_from_previous : bool, optionnal
        When we perform a fit for multiple sweeps should the fitted values of
        the previous sweep be used as guess for the next sweep.
    guesses :
        Guessed fit parameters to use. Those should include the dephasing
        field, both linear fields (for simplified the second one is ignored),
        the cubic term field.
    plot_fit : bool, optional
        Should each fit be plotted to allow manual verification.

    Returns
    -------
    dephasing_field : np.ndarray
        Dephasing field and standard deviation.
    linear_soi : np.ndarray
        Fields and standard deviations corresponding to the linear SOI terms.
        This is returned as a ...2x2 array in which the first line is the
        Rashba term and the second the Dresselhaus.
    cubic_soi_field : np.ndarray
        Field and standard deviation corresponding to the cubic Dresselhaus
        term.

    """
    # Identify the shape of the data and make them suitable for the following
    # treatment.
    if len(field.shape) >= 2:
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxy = rxy.reshape((trace_number, -1))
        if len(guesses) == 4:
            g = np.empty(original_shape + (4,))
            for i in range(4):
                fc[..., i] = guesses[i]
            guesses = g
        guesses = guesses.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        rxy = np.array((rxy,))
        guesses = np.array((guesses,))

    results = np.empty((trace_number, 4, 2))

    # Express the conductance in term of the quantum of conductance.
    sigma = (1/r) / (2*cs.e**2/cs.Planck)

    # Perform a linear fit in the specified field range and extract the slope
    for i in range(trace_number):
        start_field, stop_field = field_cutoffs[i]
        start_ind = np.argmin(np.abs(field[i] - start_field))
        stop_ind = np.argmin(np.abs(field[i] - stop_field))
        f = field[i][start_ind:stop_ind]
        r = rxy[i][start_ind: stop_ind]
        res = model.fit(r,x=f)
        results[i][0] = res.best_values['slope']/cs.e/1e4  # value in cm^-2
        results[i][1] = res.params['slope'].stderr/cs.e/1e4  # value in cm^-2

        # If requested interrupt execution to plot the result.
        if plot_fit:
            plt.plot(f, r, '+')
            plt.plot(f, res.best_fit)
            plt.show()

    if results.shape[0] == 1:
        return results[0]
    else:
        return results.T.reshape((2, ) + original_shape)