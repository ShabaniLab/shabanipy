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

#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field, file path.
DATA_PATHS = {400: 'Data_0405/JS124S_BM002_465.hdf5',
              350: 'Data_0406/JS124S_BM002_466.hdf5',
              300: 'Data_0406/JS124S_BM002_467.hdf5',
              250: 'Data_0406/JS124S_BM002_468.hdf5',
              200: 'Data_0407/JS124S_BM002_470.hdf5',
              150: 'Data_0407/JS124S_BM002_471.hdf5',
              100: 'Data_0409/JS124S_BM002_474.hdf5',}
            #   50:  'Data_0409/JS124S_BM002_476.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (),
                350: (),
                300: (),
                250: (),
                200: (),
                150: (),
                100: (),
                50: ()}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5]

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

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
PLOT_FITS = True

#: Path to which save the graphs and fitted parameters.
ANALYSIS_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                 'SQUID/By/active_t_fixed')

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

from shabanipy.squid.squid_analysis import extract_switching_current
from shabanipy.squid.squid_model import compute_squid_current
from shabanipy.squid.cpr import (fraunhofer_envelope,
                                 finite_transparency_jj_current)
from shabanipy.utils.labber_io import LabberData


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
        gates_number[f] = (data.get_axis_dimension(GATE_COLUMN) -
                           len(EXCLUDED_GATES))

        if PLOT_EXTRACTED_SWITCHING_CURRENT:
            fig, axes = plt.subplots(gates_number[f], sharex=True,
                                     figsize=(10, 15),
                                     constrained_layout=True)
            fig.suptitle(f'Parallel field {f} mT')

        for i, gate in enumerate(np.unique(data.get_data(GATE_COLUMN))):
            if gate in EXCLUDED_GATES:
                continue

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

            if PLOT_EXTRACTED_SWITCHING_CURRENT:
                axes[i].imshow(diff.T,
                               extent=(rfield[0], rfield[-1],
                                       bias[0, 0], bias[0, -1]),
                               origin='lower',
                               aspect='auto',
                               vmin=0,
                               vmax=np.max(diff[0, -1]))
                axes[i].plot(rfield, curr, color='C1')
                axes[i].set_title(f'Gate voltage {gate} V')

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
        params.add(f't_idler_{i}', min=0.0, max=1.0, value=0.5)

    params.add(f't_active_{i}', min=0.0, max=1.0, value=0.5)
    for j, gate in enumerate(datasets[f]):
        params.add(f'I_active_{i}_{j}')
        params.add(f'phi_active_{i}_{j}', value=0.0, min=0, max=2*np.pi)


def eval_squid_current(pfield, i, j, params):
    """Compute the squid flowing as a function of the perpendicular field.

    """
    t_id = (params['t_idler'] if 't_idler' in params else
            params[f't_idler_{i}'])
    idler_params = (params[f'phi_idler_{i}'],
                    params[f'I_idler_{i}'],
                    t_id)
    active_params = (params[f'phi_active_{i}_{j}'],
                     params[f'I_active_{i}_{j}'],
                     params[f't_active_{i}'])
    fe = fraunhofer_envelope(pfield*params[f'fraun_scale'] +
                             params[f'fraun_offset_{i}'])
    sq = compute_squid_current(pfield*params['phase_conversion'] *
                               PHASE_SIGN,
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
    return np.concatenate(res)

# Guess reasonable parameters
freq = []
for f in datasets:
    for rfield, curr in datasets[f].values():
        step = rfield[1] - rfield[0]
        period_index = np.argmax(np.abs(np.fft.rfft(curr)[1:])) + 1
        fft_freq = np.fft.fftfreq(len(curr), step)
        freq.append(fft_freq[period_index])
phi_conversion =  2*np.pi*np.average(freq)
params['phase_conversion'].value = phi_conversion
params['fraun_scale'].value = phi_conversion / 60
for i, f in enumerate(datasets):
    i_idler = []
    phi_idler = {}
    i_active = {}
    params[f'fraun_offset_{i}'].value = - phi_conversion*np.average(rfield)/60
    for g in datasets[f]:
        rfield, curr = datasets[f][g]
        phi_idler[g] = ((phi_conversion*rfield[np.argmax(curr)]) % (2*np.pi))
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
        params[f'phi_active_{i}_{j}'].value = phi_idler[g]

if PLOT_INITIAL_GUESS:
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
            axes[j].axvline((params[f'phi_active_{i}_{j}'].value +
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
    results[name] = np.array([params[name + f'_{i}'].value
                              for i, _ in enumerate(datasets)])

results['gate'] = np.unique([list(datasets[f]) for f in datasets])
for name in ('I_active', 'phi_active'):
    results[name] = np.array([[params[name + f'_{i}_{j}'].value
                               for j, _ in enumerate(datasets[f])]
                              for i, f in enumerate(datasets)])

# Substract the phase at the lowest gate to define the phase  difference.
results['dphi'] = - (results['phi_active'].T - results['phi_active'][:, 0]).T
results['dphi'] %= 2*np.pi

# Save the data if a file was provided.
if ANALYSIS_PATH:
    with h5py.File(os.path.join(ANALYSIS_PATH, 'results.h5'), 'w') as storage:
        storage.attrs['periodicity'] = 2*np.pi/params['phase_conversion']
        for k, v in results.items():
            storage[k] = v

# Plot meaningful results summary.

# Idler parameters
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
fig.suptitle('Idler junction parameters')
axes[0].plot(results['field'], results['I_idler'])
axes[0].set_xlabel('Parallel field (mT)')
axes[0].set_ylabel('Idler JJ current (µA)')
axes[1].plot(results['field'], results['t_idler'])
axes[1].set_xlabel('Parallel field (mT)')
axes[1].set_ylabel('Idler JJ transparency')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'idler_jj.pdf'))

# Active parameters vs gate
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
fig.suptitle('Active junction parameters vs gate')
for i, f in enumerate(results['field']):
    axes[0].plot(results['gate'], results['I_active'][i],
                 label=f'By={f} mT')
    axes[0].set_ylabel('Active JJ current (µA)')
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
axes[1].plot(results['field'], results['t_active'],
             label=f'Vg={g} V')
axes[1].set_ylabel('Active JJ transparency')
for i, g in enumerate(results['gate']):
    axes[0].plot(results['field'], results['I_active'][:, i],
                 label=f'Vg={g} V')
    axes[0].set_ylabel('Active JJ current (µA)')
    axes[2].plot(results['field'], results['phi_active'][:, i],
                 label=f'Vg={g} V')
    axes[2].set_ylabel('Active JJ phase (rad)')
for i in range(3):
    if i != 1:
        axes[i].legend()
    axes[i].set_xlabel('Parallel field (V)')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'active_jj_vs_field.pdf'))

# Phase difference vs gate and field
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
    axes[1].plot(results['field'], results['dphi'][:, i],
                 label=f'Vg={g} V')
    axes[1].set_xlabel('Parallel field (mT)')
    axes[1].set_ylabel('Phase difference (rad)')
    axes[1].legend()
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'dphi.pdf'))

plt.show()
