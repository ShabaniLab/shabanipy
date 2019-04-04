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
import numpy as np
import scipy.constants as cs
import matplotlib.pyplot as plt
from lmfit.model import Model

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_no_dresselhaus\
    import compute_wal_conductance_difference as simple_wal
from .wal_full_diagonalization\
    import compute_wal_conductance_difference as full_wal
from .utils import weight_wal_data

def estimate_parameters(field, r):
    """Estimate the parameters to use for the fit.

    """
    pass


# XXX add support for weighting the data
def extract_soi_from_wal(field, r, reference_field, max_field,
                         model='full', truncation=1000,
                         guess_from_previous=True, guesses=None,
                         plot_fit=False, method='nelder'):
    """Extract the SOI parameters from fitting the wal conductance.

    This algorithm assumes that the data are properly centered.

    The fitted values are expressed in the unit of the input field (the guesses
    should use the same convention).

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
    reference_field : float
        Field used a reference to eliminate possible experimental offsets.
    max_field : float
        Maximum field to consider when fitting the data since we know that the
        theory breaks down at high field.
    model : {'full', 'simplified'}, optional
        Model used to describe the WAL. 'simplified' corresponds to the
        situation in which either the Rashba term or the linear Dresselhaus
        term can be neglected. 'full' corresponds to a more complete model,
        however each evaluation of the fitting function requires to find the
        eigenvalues of a large and as a consequence may be slow.
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
    method : str, optional
        Algorithm to use to perform the fit. See lmfit.minimize documentation
        for acceptable values.

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
        r = r.reshape((trace_number, -1))
        if guesses is not None and len(guesses) == 4:
            g = np.empty(original_shape + (4,))
            for i in range(4):
                g[..., i] = guesses[i]
            guesses = g
        if guesses is not None:
            guesses = guesses.reshape((trace_number, -1))
        else:
            guesses = np.array([None]*trace_number)
    else:
        trace_number = 1
        field = np.array((field,))
        r = np.array((r,))
        guesses = np.array((guesses,))

    results = np.empty((4, 2, trace_number))

    # Express the conductance in term of the quantum of conductance.
    sigma = (1/r) / (2*cs.e**2/cs.Planck)

    # Create the fitting model
    if model == 'full':
        model_obj = Model(full_wal)
        model_obj.set_param_hint('matrix_truncation',
                                 value=truncation,
                                 vary=False)
    else:
        model_obj = Model(simple_wal)
        model_obj.set_param_hint('series_truncation',
                                 value=truncation,
                                 vary=False)
    model_obj.set_param_hint('low_field_reference',
                             value=reference_field,
                             vary=False)

    names = (('dephasing_field', 'linear_soi_rashba', 'linear_soi_dressel',
              'cubic_soi') if model == 'full' else
             ('dephasing_field', 'linear_soi', '', 'cubic_soi'))
    for name in [n for n in names if n]:
        model_obj.set_param_hint(name, min=0, value=1e-6)
    params = model_obj.make_params()

    # Perform a fit for each magnetic field sweep
    for i in range(trace_number):

        print(f'Treating WAL trace {i+1}/{trace_number}')

        # Conserve only the data for positive field since we symmetrized the
        # data
        mask = np.where(np.logical_and(np.greater(field[i], 0),
                                       np.less(field[i], max_field)))
        f, s = field[i][mask], sigma[i][mask]

        # Find the conductance at the reference field and compute Δσ
        ref_ind = np.argmin(np.abs(f - reference_field))
        reference_field = f[ref_ind]
        dsigma = s - s[ref_ind]

        # Build the weights
        weights = weight_wal_data(f, dsigma)

        # Set the initial values for the parameters
        if i != 0  and guess_from_previous:
            params = res.params
        else:
            if guesses[i] is not None:
                for n, v in zip(names, guesses[i]):
                    if n:
                        params[n].value = v
        params['low_field_reference'].value = reference_field

        # Perform the fit
        res = model_obj.fit(dsigma, params, field=f, method=method,
                            weights=weights)
        for j, n in enumerate(names):
            if not n:
                continue
            results[j, 0, i] = res.best_values[n]
            results[j, 1, i] = res.params[n].stderr

        # If requested plot the result.
        if plot_fit:
            fig, ax = plt.subplots()
            ax.plot(field[i], sigma[i] - s[ref_ind], '+')
            ax.plot(f, res.best_fit)
            ax.set_ylim((None, 1.1*np.max(sigma[i] - s[ref_ind])))
            ax2 = ax.twinx()
            ax2.plot(f, weights)

    if results.shape[0] == 1:
        return results[0]
    else:
        results = results.reshape((4, 2) + original_shape)
        return results[0], results[1:3], results[3]