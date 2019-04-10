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
import os
import pickle
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
                         plot_fit=False, method='least_squares',
                         weigth_method='gauss', weight_stiffness=1.0,
                         htr=None, cubic_soi=None, density=None,
                         plot_path=None):
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
    # XXX

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

    results = np.zeros((4, 2, trace_number))

    # Express the conductance in usual WAL normalization. (e^2/(2πh))
    # W. Knap et al.,
    # Weak Antilocalization and Spin Precession in Quantum Wells.
    # Physical Review B. 53, 3912–3924 (1996).
    sigma = (1/r) / (cs.e**2/(2*np.pi*cs.Planck))

    # Create the fitting model
    if model == 'full':
        raise ValueError('Unsupported model')
        model_obj = Model(full_wal)
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
        if name == 'cubic_soi' and cubic_soi is not None:
            model_obj.set_param_hint(name, value=cubic_soi,
                                     vary=cubic_soi is None)
        elif name == 'dephasing_field':
            model_obj.set_param_hint(name, min=0, value=0.0003)
        else:
            model_obj.set_param_hint(name, min=0, value=0.01)

    if cubic_soi is None:
        model_obj.set_param_hint('soi', min=0, value=0.01)
        model_obj.set_param_hint('rashba_fraction', min=0, max=1, value=1)
        model_obj.set_param_hint('linear_soi', expr='soi*rashba_fraction')
        model_obj.set_param_hint('cubic_soi', expr='soi*(1-rashba_fraction)')

    params = model_obj.make_params()

    # Perform a fit for each magnetic field sweep
    for i in range(trace_number):

        print(f'Treating WAL trace {i+1}/{trace_number}')

        # Conserve only the data for positive field since we symmetrized the
        # data
        mask = np.where(np.logical_and(np.greater(field[i], 0.0002),
                                       np.less(field[i], max_field)))
        f, s = field[i][mask], sigma[i][mask]

        # Find the conductance at the reference field and compute Δσ
        ref_ind = np.argmin(np.abs(f - reference_field))
        reference_field = f[ref_ind]
        dsigma = s - s[ref_ind]

        # Build the weights
        weights = weight_wal_data(f, dsigma, mask=weigth_method,
                                  stiffness=weight_stiffness)

        # Set the initial values for the parameters
        if i != 0  and guess_from_previous:
            params = res.params
        else:
            if guesses[i] is not None:
                for n, v in zip(names, guesses[i]):
                    if n and (n != 'cubic_soi' or cubic_soi is None):
                        params[n].value = v
        params['low_field_reference'].value = reference_field

        # Perform the fit
        res = model_obj.fit(dsigma, params, field=f, method='nelder',
                            weights=weights)
        res = model_obj.fit(dsigma, res.params, field=f, method=method,
                            weights=weights)
        for j, n in enumerate(names):
            if not n:
                continue
            results[j, 0, i] = res.best_values[n]
            results[j, 1, i] = res.params[n].stderr

        # If requested plot the result.
        if plot_fit:
            fig, ax = plt.subplots(constrained_layout=True)
            if density is not None:
                fig.suptitle(f'Density {density[i]/1e4:.1e} (cm$^2$)')
            ax.plot(field[i]*1e3, sigma[i] - s[ref_ind], '+')
            ax.plot(np.concatenate((-f[::-1], f))*1e3,
                    np.concatenate((res.best_fit[::-1], res.best_fit)))
            ax.set_xlabel('Magnetic field B (mT)')
            ax.set_ylabel(r'Δσ(B) - Δσ(0) ($\frac{e^2}{2\,π\,\hbar})$')
            amp = abs(np.max(s - s[ref_ind]) - np.min(s - s[ref_ind]))
            ax.set_ylim((None, np.max(s - s[ref_ind]) + 0.1*amp))
            if htr is None:
                ax.set_xlim((-max_field*1e3, max_field*1e3))
            else:
                # ax.set_xlim((-5*htr[i]*1e3, 5*htr[i]*1e3))
                ax.set_xlim((-50, 50))
            if htr is not None:
                ax.axvline(htr[i]*1e3, color='k', label='H$_{tr}$')
            ax.legend()
            # if plot_path:
            #     path = os.path.join(plot_path,
            #                         f'fit_{i}_n_{density[i]}.pickle')
            #     with open(path, 'wb') as fig_pickle:
            #         pickle.dump(fig, fig_pickle)
            # ax2 = ax.twinx()
            # ax2.plot(f*1e3, weights, color='C2')
            # plt.show()

    if results.shape[0] == 1:
        return results[0]
    else:
        results = results.reshape((4, 2) + original_shape)
        return results[0], results[1:3], results[3]