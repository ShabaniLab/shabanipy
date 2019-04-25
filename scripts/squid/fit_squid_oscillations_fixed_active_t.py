# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit a set of data with a constrained SQUID model.

We pack together different gate voltage data and different parallel field. We
assume the following:
- one junction (idler) is always at the same gate -> its transparency is fixed
- at a given field the critical current and the phase of the idler is fixed
- at a given field the envelope of the squid oscillation is fixed

"""

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'j2_phaseshift_zero_config.py'

#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = ''

#: Dictionary of parallel field, file path.
DATA_PATHS = {}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 2

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 3

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = False

#: Should we fix the transparency of the idler as a function of field.
FIX_IDLER_TRANSPARENCY = False

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 1

#: Fix the anomalous phase to 0.
FIX_PHI_ZERO = False

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ''

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings
import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from lmfit.models import LinearModel

from shabanipy.squid.squid_analysis import extract_switching_current
from shabanipy.squid.squid_model import compute_squid_current
from shabanipy.squid.cpr import (fraunhofer_envelope,
                                 finite_transparency_jj_current)
from shabanipy.utils.labber_io import LabberData

if CONFIG_NAME:
    print(f"Using configuration {CONFIG_NAME}, all scripts constants will be"
          " overwritten.")
    path = os.path.join(os.path.dirname(__file__), 'configs', CONFIG_NAME)
    with open(path) as f:
        exec(f.read())

if not DATA_ROOT_FOLDER:
    raise ValueError('No root path for the date was specified.')

gates_number = {}
datasets = {}

# Load and filter all the datasets
for f, ppath in DATA_PATHS.items():

    datasets[f] = {}

    with LabberData(os.path.join(DATA_ROOT_FOLDER, ppath)) as data:

        shape = data.compute_shape((BIAS_COLUMN, FIELD_COLUMN))
        # Often we do not do all perp field so allow to reshape as needed
        shape = (shape[0], -1)
        frange = FIELD_RANGES[f]
        gates = [g for g in np.unique(data.get_data(GATE_COLUMN))
                 if g not in EXCLUDED_GATES]
        gates_number[f] = len(gates)

        if PLOT_EXTRACTED_SWITCHING_CURRENT:
            fig, axes = plt.subplots(gates_number[f], sharex=True,
                                     figsize=(10, 15),
                                     constrained_layout=True)
            fig.suptitle(f'Parallel field {f} mT')

        for i, gate in enumerate(gates):
            filt = {GATE_COLUMN: gate}
            field = data.get_data(FIELD_COLUMN, filters=filt).reshape(shape).T
            bias = data.get_data(BIAS_COLUMN, filters=filt).reshape(shape).T
            li_index = data.name_or_index_to_index(RESISTANCE_COLUMN)
            diff = (data.get_data(li_index, filters=filt) +
                    1j*data.get_data(li_index + 1, filters=filt))
            diff = np.abs(diff).reshape(shape).T
            rfield, curr = extract_switching_current(field, bias, diff,
                                                     RESISTANCE_THRESHOLD)

            # Filter out Nan
            mask = np.logical_not(np.isnan(curr))
            rfield, curr = rfield[mask], curr[mask]

            if any(frange):
                masks = []
                if frange[0]:
                    masks.append(np.greater(rfield, frange[0]))
                if frange[1]:
                    masks.append(np.less(rfield, frange[1]))
                index = (np.nonzero(np.logical_and(*masks))
                         if len(masks) == 2 else np.nonzero(masks[0]))
                datasets[f][gate] = (rfield[index], curr[index])
            else:
                datasets[f][gate] = (rfield, curr)

            if PLOT_EXTRACTED_SWITCHING_CURRENT:
                axes[i].imshow(diff.T,
                               extent=(rfield[0], rfield[-1],
                                       bias[0, 0], bias[0, -1]),
                               origin='lower',
                               aspect='auto',
                               vmin=0,
                               vmax=np.max(diff[0, -1]))
                axes[i].plot(*datasets[f][gate], color='C1')
                axes[i].set_title(f'Gate voltage {gate} V')

if PLOT_EXTRACTED_SWITCHING_CURRENT:
    plt.show()

# Setup the fit
params = Parameters()
params.add(f'phase_conversion')
params.add(f'fraun_scale', value=0)
if FIX_IDLER_TRANSPARENCY:
    params.add('t_idler', min=0, max=1)

for i, f in enumerate(datasets):
    params.add(f'I_idler_{i}')
    params.add(f'phi_idler_{i}', value=0, vary=False)
    params.add(f'fraun_offset_{i}', value=0.0)
    if not FIX_IDLER_TRANSPARENCY:
        params.add(f't_idler_{i}', min=0.0, max=0.999, value=0.9, vary=False)

    params.add(f't_active_{i}', min=0.0, max=0.999, value=0.9)
    if FIX_PHI_ZERO:
        params.add(f'phi_active_{i}', value=0.0, min=0, max=2*np.pi)
    for j, gate in enumerate(datasets[f]):
        params.add(f'I_active_{i}_{j}')
        if not FIX_PHI_ZERO:
            params.add(f'phi_active_{i}_{j}', value=0.0, min=0, max=2*np.pi)


def eval_squid_current(pfield, i, j, params):
    """Compute the squid flowing as a function of the perpendicular field.

    """
    t_id = (params['t_idler'] if 't_idler' in params else
            params[f't_idler_{i}'])
    idler_params = (params[f'phi_idler_{i}'],
                    params[f'I_idler_{i}'],
                    t_id)
    phi_id = (params[f'phi_active_{i}_{j}']
              if f'phi_active_{i}_{j}' in params else
              params[f'phi_active_{i}'])
    active_params = (phi_id,
                     params[f'I_active_{i}_{j}'],
                     params[f't_active_{i}'])
    fraun_phase = pfield*params[f'fraun_scale'] + params[f'fraun_offset_{i}']
    fe = fraunhofer_envelope(fraun_phase)
    sq = compute_squid_current(HANDEDNESS*pfield*params['phase_conversion'],
                               finite_transparency_jj_current,
                               idler_params,
                               finite_transparency_jj_current,
                               active_params)
    return fe*sq


def target_function(params, datasets):
    """Target function used to fit multiple SQUID traces at once.

    """
    res = []
    params = params.valuesdict()
    for i, f in enumerate(datasets):
        t_id = (params['t_idler'] if 't_idler' in params else
                params[f't_idler_{i}'])
        idler_params = (params[f'phi_idler_{i}'],
                        params[f'I_idler_{i}'],
                        t_id)
        for j, g in enumerate(datasets[f]):
            pfield, curr = datasets[f][g]
            model = eval_squid_current(pfield, i, j, params)
            res.append(model - curr)
    res = np.concatenate(res)
    return res

# Guess reasonable parameters
freq = []
for f in datasets:
    for rfield, curr in datasets[f].values():
        step = rfield[1] - rfield[0]
        period_index = np.argmax(np.abs(np.fft.rfft(curr)[1:])) + 1
        fft_freq = np.fft.fftfreq(len(curr), step)
        freq.append(fft_freq[period_index])
phi_conversion =  2*np.pi*np.average(freq)
params['phase_conversion'].value = (phi_conversion *
                                    CONVERSION_FACTOR_CORRECTION)
params['fraun_scale'].value = phi_conversion / 60
for i, f in enumerate(datasets):
    i_idler = []
    phi_active = {}
    i_active = {}
    field_at_max = []
    for g in datasets[f]:
        rfield, curr = datasets[f][g]
        field_at_max.append(rfield[np.argmax(curr)])
        phi_active[g] = ((phi_conversion*rfield[np.argmax(curr)]) % (2*np.pi))
        maxc, minc = np.amax(curr), np.amin(curr)
        avgc = (maxc + minc)/2
        amp = (maxc - minc)/2
        # Assume that for the low gate the idler has a larger current
        if not i_idler:
            i_idler.append(amp)
            i_active[g] = amp
        # Identify the idler current as the one closest to the one previously
        # identified
        else:
            if abs(avgc - i_idler[0]) < abs(amp - i_idler[0]):
                i_idler.append(avgc)
                i_active[g] = amp
            else:
                i_idler.append(amp)
                i_active[g] = avgc
    params[f'I_idler_{i}'].value = np.average(i_idler)
    for j, g in enumerate(datasets[f]):
        params[f'I_active_{i}_{j}'].value = i_active[g]
        if f'phi_active_{i}_{j}' in params:
            params[f'phi_active_{i}_{j}'].value = HANDEDNESS*phi_active[g]
    # If we enforce a common phase difference (simply a field offset) use the
    # average guess
    if f'phi_active_{i}' in params:
        params[f'phi_active_{i}'].value = (HANDEDNESS *
                                           np.average(list(phi_active.values())
                                           )
    # Now that rfield refers to the proper field compute the offset for the
    # Fraunhofer pattern
    params[f'fraun_offset_{i}'].value = - (np.average(field_at_max) *
                                           params['fraun_scale'].value)

if PLOT_INITIAL_GUESS:
    for i, f in enumerate(datasets):
        plt.figure()
        for j, g in enumerate(datasets[f]):
            rfield, curr = datasets[f][g]
            plt.plot(rfield, curr)
    # Plot the initial guesses to check our automatic guesses
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f], sharex=True,
                                 figsize=(10, 15),
                                 constrained_layout=True)
        fig.suptitle(f'Parallel field {f} mT')
        for j, g in enumerate(datasets[f]):
            rfield, curr = datasets[f][g]
            axes[j].plot(rfield, curr)
            axes[j].plot(rfield,
                         eval_squid_current(rfield, i, j, params.valuesdict()))
            phi_active = (params[f'phi_active_{i}']
                          if FIX_PHI_ZERO else params[f'phi_active_{i}_{j}'])
            axes[j].axvline((phi_active.value +
                             params[f'phi_idler_{i}'].value)/phi_conversion)
            axes[j].set_title(f'Gate voltage {g} V')
    plt.show()

# Perform the fit
result = minimize(target_function, params, args=(datasets,),
                  method='leastsq')
params = result.params

if PLOT_FITS:
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(len(datasets[f]),
                                figsize=(10, 15),
                                sharex=True,
                                constrained_layout=True)
        fig.suptitle(f'Parallel field {f} mT')
        for j, g in enumerate(datasets[f]):
            field, curr = datasets[f][g]
            axes[j].plot(field, curr, '+')
            axes[j].plot(field,
                         eval_squid_current(field, i, j, params.valuesdict()))
            axes[j].set_title(f'Gate voltage {g} V')
        if ANALYSIS_PATH:
            fig.savefig(os.path.join(ANALYSIS_PATH, f'fit_f_{f}.pdf'))
    plt.show()

# Build result arrays from the fitted parameters
results = {}
results['field'] = np.array(list(datasets))
for name in ('I_idler', 't_idler', 't_active'):
    if FIX_IDLER_TRANSPARENCY and name == 't_idler':
        continue
    results[name] = np.array([params[name + f'_{i}'].value
                              for i, _ in enumerate(datasets)])

results['gate'] = np.unique([list(datasets[f]) for f in datasets])
for name in ('I_active', 'phi_active'):
    if FIX_PHI_ZERO and name == 'phi_active':
        results[name] = np.array([params[name + f'_{i}'].value
                                  for i, f in enumerate(datasets)])
        continue
    results[name] = np.array([[params[name + f'_{i}_{j}'].value
                               for j, _ in enumerate(datasets[f])]
                              for i, f in enumerate(datasets)])

# Substract the phase at the lowest gate to define the phase  difference.
if FIX_PHI_ZERO:
    results['dphi'] = np.zeros_like(results['phi_active'])
else:
    results['dphi'] = - (PHASE_SIGN * (results['phi_active'].T -
                                       results['phi_active'][:, 0]).T)
    results['dphi'] %= 2*np.pi

# Save the data if a file was provided.
if ANALYSIS_PATH:
    with h5py.File(os.path.join(ANALYSIS_PATH, 'results.h5'), 'w') as storage:
        storage.attrs['periodicity'] = 2*np.pi/params['phase_conversion']
        storage.attrs['res_threshold'] = RESISTANCE_THRESHOLD
        for k, v in results.items():
            storage[k] = v

# Plot meaningful results summary.

# Idler parameters
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
fig.suptitle('Idler junction parameters')
sort_index = np.argsort(results['field'])
axes[0].plot(results['field'][sort_index], results['I_idler'][sort_index])
axes[0].set_xlabel('Parallel field (mT)')
axes[0].set_ylabel('Idler JJ current (µA)')
if not FIX_IDLER_TRANSPARENCY:
    axes[1].plot(results['field'][sort_index], results['t_idler'][sort_index])
    axes[1].set_xlabel('Parallel field (mT)')
    axes[1].set_ylabel('Idler JJ transparency')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'idler_jj.pdf'))
print(f"Idler JJ transparency {results['t_idler']}")

# Active parameters vs gate
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
fig.suptitle('Active junction parameters vs gate')
for i, f in enumerate(results['field']):
    axes[0].plot(results['gate'], results['I_active'][i],
                 label=f'By={f} mT')
    axes[0].set_ylabel('Active JJ current (µA)')
    if not FIX_PHI_ZERO:
        axes[1].plot(results['gate'], results['phi_active'][i],
                    label=f'By={f} mT')
        axes[1].set_ylabel('Active JJ phase (rad)')
for i in range(2):
    axes[i].legend()
    axes[i].set_xlabel('Gate voltage (V)')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'active_jj_vs_gate.pdf'))

# Active parameters vs field
fig, axes = plt.subplots(2, 2, figsize=(9, 9),
                         constrained_layout=True)
fig.suptitle('Active junction parameters vs field')
gs = axes[1, 0].get_gridspec()
# remove the underlying axes
for ax in axes[1]:
    ax.remove()
axes = [axes[0, 0], axes[0, 1], fig.add_subplot(gs[1, :])]
sort_index = np.argsort(results['field'])
axes[1].plot(results['field'][sort_index], results['t_active'][sort_index],
             label=f'Vg={g} V')
axes[1].set_ylabel('Active JJ transparency')
for i, g in enumerate(results['gate']):
    axes[0].plot(results['field'][sort_index],
                 results['I_active'][:, i][sort_index],
                 label=f'Vg={g} V')
    axes[0].set_ylabel('Active JJ current (µA)')
    phi = (results['phi_active'] if FIX_PHI_ZERO else
           results['phi_active'][:, i])
    axes[2].plot(results['field'][sort_index],
                 phi[sort_index],
                 label=f'Vg={g} V')
    axes[2].set_ylabel('Active JJ phase (rad)')
for i in range(3):
    if i != 1:
        axes[i].legend()
    axes[i].set_xlabel('Parallel field (mT)')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'active_jj_vs_field.pdf'))
print(f"Active JJ transparency {results['t_active']}")

# Phase difference vs gate and field
if not FIX_PHI_ZERO:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                            constrained_layout=True)
    fig.suptitle('Phase difference')
    for i, f in enumerate(results['field']):
        axes[0].plot(results['gate'][1:], results['dphi'][i, 1:],
                    label=f'By={f} mT')
        axes[0].set_xlabel('Gate voltage (V)')
        axes[0].set_ylabel('Phase difference (rad)')
        axes[0].legend()
    for i, g in enumerate(results['gate']):
        if i == 0:
            continue

        # Perform a linear fit
        field = results['field']
        dphi  = results['dphi'][:, i]
        if len(dphi) > 1:
            model = LinearModel()
            p = model.guess(dphi, x=results['field'])
            res = model.fit(dphi, p, x=results['field'])
            ex_field = np.linspace(0, max(field))
            axes[1].plot(ex_field, res.eval(x=ex_field), color=f'C{i}')

        axes[1].plot(field, dphi, '+', color=f'C{i}', label=f'Vg={g} V')

    axes[1].set_xlabel('Parallel field (mT)')
    axes[1].set_ylabel('Phase difference (rad)')
    axes[1].set_ylim((0, None))
    axes[1].legend()

if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'dphi.pdf'))

plt.show()
