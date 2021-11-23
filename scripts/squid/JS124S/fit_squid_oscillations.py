# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit a set of data with a constrained SQUID model.

We pack together data obtained at different parallel by sweeping the gate
either of junction 1 (max gate of 1) or of junction 2 (max gate of 4).

Based on zero field measurements we assume the following:
- the transparencies of the junctions depend only on the parallel field
- at a given field, the amplitude of the current and the phase difference
  between the junctions is a function of the gate voltage.
- at a given field the envelope of the squid oscillation is fixed

"""

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'both_phaseshift_by.py'

#: Dictionary of parallel field: pair of file path. The first path should refer
#: to the dataset in which the j1 junction is gated, the second to the dataset
#: in which the j2 junction is gated.
DATA_PATHS = {400: ['04/Data_0405/JS124S_BM002_465.hdf5',
                    '03/Data_0316/JS124S_BM002_390.hdf5'],
            #   350: ['04/Data_0406/JS124S_BM002_466.hdf5',
            #         '03/Data_0317/JS124S_BM002_392.hdf5'],
            #   300: ['04/Data_0406/JS124S_BM002_467.hdf5',
            #         '03/Data_0318/JS124S_BM002_394.hdf5'],
            #   250: ['04/Data_0406/JS124S_BM002_468.hdf5',
            #         '03/Data_0318/JS124S_BM002_395.hdf5'],
            #   200: ['04/Data_0407/JS124S_BM002_470.hdf5',
            #         '03/Data_0318/JS124S_BM002_396.hdf5'],
            #   150: ['04/Data_0407/JS124S_BM002_471.hdf5',
            #         '03/Data_0319/JS124S_BM002_397.hdf5'],
              100: ['04/Data_0409/JS124S_BM002_474.hdf5',
                    '03/Data_0321/JS124S_BM002_405.hdf5']}

#: Perpendicular field range to fit for each parallel field.
FIELD_RANGES = {400: [(), (-8e-3, -5.5e-3)],
                350: [(None, 0.2e-3), (None, -6e-3)],
                300: [(), (-6.59e-3, -4.75e-3)],
                250: [(None, 0.2e-3), ()],
                200: [(), ()],
                150: [(), (-4.7e-3, None)],
                100: [(None, 0.2e-3), (-3.9e-3, -1.1e-3)]}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,
                      350: 0.1,
                      300: 0.2,
                      250: 0.3,
                      200: 0.4,
                      150: 0.6,
                      100: 0.8}

#: Name/index of the gate column.
GATE_COLUMN = 1

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = [-4.75, -3.5, -2.5, -2.0, -1.0, 1.0, 2.0, 3.0]

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 2

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 3

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7 # 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = False

#: Sign of the phase difference created by the perpendicular field. The
#: phase difference is applied on the junction j1.
PHASE_SIGN = (1, -1)

#: Handedness of the system ie does a positive field translate in a negative
#: or positive phase difference.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = (1.05, 1.07)

#: Fix the anomalous phase to 0.
FIX_PHI_ZERO = False

#: Enforce equality of the transparencies
EQUAL_TRANSPARENCIES = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
#: Recognized values are False, True, 'color' (to plot over the colormap)
PLOT_FITS = 'color'

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings
import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, conf_interval
from lmfit.models import LinearModel

from shabanipy.squid.squid_analysis import extract_switching_current
from shabanipy.squid.squid_model import compute_squid_current
from shabanipy.squid.cpr import (fraunhofer_envelope,
                                 finite_transparency_jj_current)
from shabanipy.labber import LabberData
from shabanipy.plotting.utils import format_phase

from patch_labber_io import patch_labberdata

patch_labberdata()

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 13
plt.rcParams['pdf.fonttype'] = 42

if CONFIG_NAME:
    print(f"Using configuration {CONFIG_NAME}, all scripts constants will be"
          " overwritten.")
    path = os.path.join(os.path.dirname(__file__), 'configs', CONFIG_NAME)
    with open(path) as f:
        exec(f.read())

if not DATA_ROOT_FOLDER:
    raise ValueError('No root path for the date was specified.')

os.makedirs(ANALYSIS_PATH, exist_ok=True)

gates_number = {}
datasets = {}
datasets_color = {}

# Load and filter all the datasets
for f, jj_paths in DATA_PATHS.items():

    datasets[f] = [{}, {}]
    datasets_color[f] = [{}, {}]

    for ind, jj_path in enumerate(jj_paths):

        with LabberData(os.path.join(DATA_ROOT_FOLDER, jj_path)) as data:
            frange = FIELD_RANGES[f][ind]
            gates = [g for g in np.unique(data.get_data(GATE_COLUMN))
                     if g not in EXCLUDED_GATES and not np.isnan(g)]
            gates_number[f] = len(gates)

            if PLOT_EXTRACTED_SWITCHING_CURRENT:
                fig, axes = plt.subplots(gates_number[f], sharex=True,
                                        figsize=(10, 15),
                                        constrained_layout=True)
                fig.suptitle(f'Parallel field {f} mT')

            for i, gate in enumerate(gates):
                filt = {GATE_COLUMN: gate}
                field = data.get_data(FIELD_COLUMN, filters=filt)
                bias = data.get_data(BIAS_COLUMN, filters=filt)
                diff = data.get_data(RESISTANCE_COLUMN, filters=filt)
                diff = np.abs(diff)
                rfield, curr = extract_switching_current(field, bias, diff,
                                                        RESISTANCE_THRESHOLD)

                # Flip the first axis if the field was ramped down.
                if rfield[0] > rfield[1]:
                    diff = diff[::-1]

                # Filter out Nan
                mask = np.logical_not(np.isnan(curr))
                rfield, curr = rfield[mask], curr[mask]
                sort_index = np.argsort(rfield)
                rfield, curr = rfield[sort_index], curr[sort_index]

                if any(frange):
                    masks = []
                    if frange[0]:
                        masks.append(np.greater(rfield, frange[0]))
                    if frange[1]:
                        masks.append(np.less(rfield, frange[1]))
                    index = (np.nonzero(np.logical_and(*masks))
                            if len(masks) == 2 else np.nonzero(masks[0]))
                    datasets[f][ind][gate] = (rfield[index], curr[index])
                    datasets_color[f][ind][gate] = (diff[index],
                                                    (rfield[index][0],
                                                     rfield[index][-1],
                                                     bias[0, 0],
                                                     bias[0, -1]))
                else:
                    datasets[f][ind][gate] = (rfield, curr)
                    datasets_color[f][ind][gate] = (diff,
                                                    (rfield[0], rfield[-1],
                                                     bias[0, 0], bias[0, -1]))

                if PLOT_EXTRACTED_SWITCHING_CURRENT:
                    axes[i].imshow(diff.T,
                                   extent=(min(rfield), max(rfield),
                                           bias[0, 0], bias[0, -1]),
                                   origin='lower',
                                   aspect='auto',
                                   vmin=0,
                                   vmax=np.max(diff[0, -1]))
                    axes[i].plot(*datasets[f][ind][gate], color='C1')
                    axes[i].set_title(f'Gate voltage {gate} V')

if PLOT_EXTRACTED_SWITCHING_CURRENT:
    plt.show()

# Setup the fit
params = Parameters()
params.add(f'phase_conversion_j1')
params.add(f'phase_conversion_j2')
params.add(f'fraun_scale', value=0)

for i, f in enumerate(datasets):

    params.add(f'fraun_offset_{i}', value=0.0)
    params.add(f't_j1_{i}', min=0.0, max=0.999, value=TRANSPARENCY_GUESS[f])
    params.add(f't_j2_{i}', min=0.0, max=0.999, value=TRANSPARENCY_GUESS[f])
    if EQUAL_TRANSPARENCIES:
        params[f't_j2_{i}'].set(expr=f't_j1_{i}')

    for j, g_dataset in enumerate(datasets[f]):
        # Parameters of the idler junction (j2 for the first dataset, j1 for
        # the second).
        params.add(f'I_j{(j+1)%2+1}_idler_{i}')
        params.add(f'Boffset_j{j+1}_{i}', vary=False)

        if FIX_PHI_ZERO:
            params.add(f'phi_j{j+1}_active_{i}', value=0.0,
                       min=-np.pi, max=np.pi)
        for k, gate in enumerate(g_dataset):
            params.add(f'I_j{j+1}_active_{i}_{k}')
            if not FIX_PHI_ZERO:
                params.add(f'phi_j{j+1}_active_{i}_{k}',
                           value=0.0, min=-np.pi, max=np.pi)


def eval_squid_current(pfield, i, j, k, params):
    """Compute the squid flowing as a function of the perpendicular field.

    Parameters
    ----------
    pfield : np.ndarray
        Perpendicular field applied on the SQUID

    i : int
        Index of the field dataset.
    j : int
        j+1 is the index of the active junction
    k : int
        Index of the gate dataset considered
    params : dict
        Key value pairs of the fitting parameters.

    """
    phi_id = (params[f'phi_j{j+1}_active_{i}_{k}']
              if f'phi_j{j+1}_active_{i}_{k}' in params else
              params[f'phi_j{j+1}_active_{i}'])
    if j == 0:
        j1_params = (phi_id,
                     params[f'I_j1_active_{i}_{k}'],
                     params[f't_j1_{i}'])
        j2_params = (0,
                     params[f'I_j2_idler_{i}'],
                     params[f't_j2_{i}'])
    else:
        j1_params = (phi_id,
                     params[f'I_j1_idler_{i}'],
                     params[f't_j1_{i}'])
        j2_params = (0,
                     params[f'I_j2_active_{i}_{k}'],
                     params[f't_j2_{i}'])
    f = (pfield - params[f'Boffset_j{j+1}_{i}'])
    fraun_phase = f*params[f'fraun_scale'] + params[f'fraun_offset_{i}']
    fe = fraunhofer_envelope(fraun_phase)
    sq = compute_squid_current(HANDEDNESS*f*params[f'phase_conversion_j{j+1}'],
                               finite_transparency_jj_current,
                               j1_params,
                               finite_transparency_jj_current,
                               j2_params)
    return fe*sq


def target_function(params, datasets):
    """Target function used to fit multiple SQUID traces at once.

    """
    res = []
    params = params.valuesdict()
    for i, f in enumerate(datasets):
        for j, g_dataset in enumerate(datasets[f]):
            for k, g in enumerate(g_dataset):
                pfield, curr = g_dataset[g]
                model = eval_squid_current(pfield, i, j, k, params)
                res.append(model - curr)
    res = np.concatenate(res)
    return res

# --- Guess reasonable parameters

# Start with the frequency by performing a fft
freq = [[], []]
for f in datasets:
    for j, g_dataset in enumerate(datasets[f]):
        for rfield, curr in g_dataset.values():
            step = rfield[1] - rfield[0]
            period_index = np.argmax(np.abs(np.fft.rfft(curr)[1:])) + 1
            fft_freq = np.fft.fftfreq(len(curr), step)
            freq[j].append(fft_freq[period_index])
phi_conversion = [2*np.pi*np.average(np.abs(f)) for f in freq]
params['phase_conversion_j1'].value = (phi_conversion[0] *
                                       CONVERSION_FACTOR_CORRECTION[0])
params['phase_conversion_j2'].value = (phi_conversion[1] *
                                       CONVERSION_FACTOR_CORRECTION[1])
params['fraun_scale'].value = phi_conversion[0] / 60

# Go to the amplitudes and phases
max_fields = []
for i, f in enumerate(datasets):
    max_fields.append([])

    for j, g_dataset in enumerate(datasets[f]):
        i_idler = []
        i_active = {}
        field_at_max = []
        max_fields[i].append(field_at_max)

        # Go through the data once to identify the amplitudes and phases
        for g in g_dataset:
            rfield, curr = g_dataset[g]
            field_at_max.append(rfield[np.argmax(curr)])
            maxc, minc = np.amax(curr), np.amin(curr)
            avgc = (maxc + minc)/2
            amp = (maxc - minc)/2
            # Assume that for the low gate the idler has a larger current
            if not i_idler:
                i_idler.append(max(amp, avgc))
                i_active[g] = min(amp, avgc)
            # Identify the idler current as the one closest to the one
            # previously identified
            else:
                if abs(avgc - i_idler[0]) < abs(amp - i_idler[0]):
                    i_idler.append(avgc)
                    i_active[g] = amp
                else:
                    i_idler.append(amp)
                    i_active[g] = avgc

        # Set the guessed values
        params[f'I_j{(j+1)%2+1}_idler_{i}'].value = np.average(i_idler)
        for k, g in enumerate(g_dataset):
            params[f'I_j{j+1}_active_{i}_{k}'].value = i_active[g]

        # Set the field offset based on the lowest gate
        params[f'Boffset_j{j+1}_{i}'].value = max_fields[i][j][0]

# To estimate the phase compare the position of the maximum in the data and in
# the model.
for i, f in enumerate(datasets):
    for j, g_dataset in enumerate(datasets[f]):
        for k, g in enumerate(g_dataset):
            rfield, curr = datasets[f][j][g]
            model_curr = eval_squid_current(rfield, i, j, k,
                                            params.valuesdict())
            f_index = np.argmin(np.abs(rfield - max_fields[i][j][k]))
            period = int(2*np.pi/phi_conversion[j]/abs(rfield[1] - rfield[0]))
            mask = slice(max(0, f_index - period//2), f_index + period//2)
            max_model = np.argmax(model_curr[mask])
            phi = phi_conversion[j]*(max_fields[i][j][k] -
                                     rfield[mask][max_model])
            params[f'phi_j{j+1}_active_{i}_{k}'].value = phi


if PLOT_INITIAL_GUESS:
    # Plot the initial guesses to check our automatic guesses
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f], 2,
                                 figsize=(10, 15),
                                 constrained_layout=True)
        fig.suptitle(f'Initial guess: Parallel field {f} mT')
        for j, g_dataset in enumerate(datasets[f]):
            for k, g in enumerate(g_dataset):
                rfield, curr = datasets[f][j][g]
                axes[k, j].plot(rfield, curr)
                model_curr = eval_squid_current(rfield, i, j, k,
                                                params.valuesdict())
                axes[k, j].plot(rfield, model_curr)
                axes[k, j].axvline(max_fields[i][j][k])
                axes[k, j].set_title(f'JJ {j+1} Gate voltage {g} V')
    plt.show()

# Perform the fit
mini = Minimizer(target_function, params, fcn_args=(datasets,))
result = mini.minimize(method='leastsq')
params = result.params

if PLOT_FITS:
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f], 2,
                                figsize=(12, 12),
                                constrained_layout=True)
        fig.suptitle(f'Fits: Parallel field {f} mT')
        color_max = 1.2*max([np.max(datasets_color[f][j][g][0][0, -1])*1e8
                            for j, g_data in enumerate(datasets[f])
                            for g in g_data])
        for j, g_dataset in enumerate(datasets[f]):
            for k, g in enumerate(g_dataset):
                field, curr = datasets[f][j][g]
                phase = ((field - params[f'Boffset_j{j+1}_{i}']) *
                         params[f'phase_conversion_j{j+1}'].value/np.pi)
                if PLOT_FITS == 'color':
                    diff, extent = datasets_color[f][j][g]
                    f0 = ((extent[0] - params[f'Boffset_j{j+1}_{i}']) *
                          params[f'phase_conversion_j{j+1}'].value/np.pi)
                    f1 = ((extent[1] - params[f'Boffset_j{j+1}_{i}']) *
                          params[f'phase_conversion_j{j+1}'].value/np.pi)
                    im = axes[-k-1, j].imshow(diff.T*1e8,
                                           extent=(f0, f1,
                                                   extent[2], extent[3]),
                                           origin='lower',
                                           aspect='auto',
                                           vmin=0,
                                           vmax=np.max(diff[0, -1])*1e8)
                else:
                    axes[-k-1, j].plot(field, curr, '+')
                if MULTIPLE_TRANSPARENCIES:
                    for l, t in enumerate(MULTIPLE_TRANSPARENCIES):
                        p = params.valuesdict()
                        # p[f't_j1_{i}'] = t
                        p[f't_j2_{i}'] = t
                        model = eval_squid_current(field, i, j, k, p)
                        axes[-k-1, j].plot(phase, model, color=f'C{l+1}',
                                           label=f't={t}')
                else:
                    model = eval_squid_current(field, i, j, k,
                                               params.valuesdict())
                    axes[-k-1, j].plot(phase, model, color='C1')
                axes[-k-1, j].xaxis.set_major_formatter(plt.FuncFormatter(format_phase))
                axes[-k-1, j].tick_params(direction='in', width=1.5)
                axes[-k-1, j].legend(title=f'Vg{j+1} = {g} V', loc=1)

            axes[-1, j].set_xlabel('SQUID phase')
            axes[len(g_dataset)//2, j].set_ylabel('Bias current (µA)')
            if PLOT_FITS == 'color':
                cb = plt.colorbar(im, ax=axes[len(g_dataset)//2, j])
                cb.set_label('Resistance (Ω)')
        if ANALYSIS_PATH:
            fig.savefig(os.path.join(ANALYSIS_PATH, f'fit_f_{f}.pdf'))
    plt.show()

# Build result arrays from the fitted parameters
results = {}
results['field'] = np.array(list(datasets))
for name in ('I_j1_idler', 'I_j2_idler', 't_j1', 't_j2'):
    results[name] = np.array([params[name + f'_{i}'].value
                              for i, _ in enumerate(datasets)])

results['gate'] = np.unique([list(datasets[f][0]) for f in datasets])
for name in ('I_j1_active', 'I_j2_active',
             'phi_j1_active', 'phi_j2_active'):
    if FIX_PHI_ZERO and name in ('phi_j1_active', 'phi_j2_active'):
        results[name] = np.array([params[name + f'_{i}'].value
                                  for i, f in enumerate(datasets)])
        continue
    index = 0 if '1' in name else 1
    results[name] = np.array([[params[name + f'_{i}_{j}'].value
                               for j, _ in enumerate(datasets[f][index])]
                              for i, f in enumerate(datasets)])

# Substract the phase at the lowest gate to define the phase difference.
if FIX_PHI_ZERO:
    results['dphi_j1'] = np.zeros_like(results['phi_j1_active'])
    results['dphi_j2'] = np.zeros_like(results['phi_j2_active'])
else:
    for i in range(2):
        results[f'dphi_j{i+1}'] = - (PHASE_SIGN[i] *
                                     (results[f'phi_j{i+1}_active'].T -
                                      results[f'phi_j{i+1}_active'][:, 0]).T)
        results[f'dphi_j{i+1}'] %= 2*np.pi
        mask = np.greater(results[f'dphi_j{i+1}'], np.pi).astype(int)
        results[f'dphi_j{i+1}'] -= 2*np.pi*mask

# Save the data if a file was provided.
if ANALYSIS_PATH:
    with h5py.File(os.path.join(ANALYSIS_PATH, 'results.h5'), 'w') as storage:
        storage.attrs['periodicity_j1'] = 2*np.pi/params['phase_conversion_j1']
        storage.attrs['periodicity_j2'] = 2*np.pi/params['phase_conversion_j2']
        storage.attrs['res_threshold'] = RESISTANCE_THRESHOLD
        storage.attrs['equal_transparencies'] = EQUAL_TRANSPARENCIES
        for k, v in results.items():
            storage[k] = v

# Plot meaningful results summary.

# Idler current and junction transparencies
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
sort_index = np.argsort(results['field'])
for i in (1, 2):
    axes[0].plot(results['field'][sort_index],
                 results[f'I_j{i}_idler'][sort_index],
                 label=f'JJ {i}')
axes[0].set_xlabel('Parallel field (mT)')
axes[0].set_ylabel('Idler JJ current (µA)')
axes[0].legend()
for i in (1, 2):
    axes[1].errorbar(results['field'][sort_index],
                     results[f't_j{i}'][sort_index],
                     yerr=0.2,
                     label=f'JJ {i}')
    axes[1].set_ylim((0, 1))
axes[1].set_xlabel('Parallel field (mT)')
axes[1].set_ylabel('JJ transparency')
axes[1].legend()
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'idler_jj.pdf'))

# Active parameters vs gate
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 8),
                         constrained_layout=True)
fig.suptitle('Active junction parameters vs gate')
for i, f in enumerate(results['field']):
    for j in range(2):
        axes[j, 0].plot(results['gate'], results[f'I_j{j+1}_active'][i],
                        label=f'By={f} mT')
        axes[j, 0].set_ylabel(f'Active JJ {j+1} current (µA)')
        if not FIX_PHI_ZERO:
            axes[j, 1].plot(results['gate'], results[f'phi_j{j+1}_active'][i],
                            label=f'By={f} mT')
            axes[j, 1].set_ylabel(f'Active JJ {j+1} phase (rad)')
        for k in range(2):
            axes[j, k].legend()
            axes[j, k].set_xlabel('Gate voltage (V)')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'active_jj_vs_gate.pdf'))

# Active parameters vs field
fig, axes = plt.subplots(2, 2, figsize=(8, 8),
                         constrained_layout=True)
fig.suptitle('Active junction parameters vs field')
sort_index = np.argsort(results['field'])
for i, g in enumerate(results['gate']):
    for j in range(2):
        axes[j, 0].plot(results['field'][sort_index],
                        results[f'I_j{j+1}_active'][:, i][sort_index],
                        label=f'Vg={g} V')
        axes[j, 0].set_ylabel(f'Active JJ {j+1} current (µA)')
        phi = (results[f'phi_j{j+1}_active'] if FIX_PHI_ZERO else
               results[f'phi_j{j+1}_active'][:, i])
        axes[j, 1].plot(results['field'][sort_index],
                    phi[sort_index],
                    label=f'Vg={g} V')
        axes[j, 1].set_ylabel(f'Active JJ {j+1} phase (rad)')
for i in range(2):
    for j in range(2):
        axes[i, j].legend()
        axes[i, j].set_xlabel('Parallel field (mT)')
if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'active_jj_vs_field.pdf'))

# Phase difference vs gate and field
if not FIX_PHI_ZERO:
    fig, axes = plt.subplots(2, 2, figsize=(9, 9),
                            constrained_layout=True)
    fig.suptitle('Phase difference')
    for i, f in enumerate(results['field']):
        for j in range(2):
            axes[j, 0].plot(results['gate'][1:],
                            results[f'dphi_j{j+1}'][i, 1:], '+',
                            label=f'By={f} mT')
            axes[j, 0].set_xlabel('Gate voltage (V)')
            axes[j, 0].set_ylabel('Phase difference (rad)')
            axes[j, 0].legend()
    for i, g in enumerate(results['gate']):
        if i == 0:
            continue
        for j in range(2):
            # Perform a linear fit
            field    = results['field']
            dphi     = results[f'dphi_j{j+1}'][:, i]
            if len(dphi) > 1:
                model = LinearModel()
                p = model.guess(dphi, x=results['field'])
                res = model.fit(dphi, p, x=results['field'])
                ex_field = np.linspace(min(0, min(field)), max(field))
                axes[j, 1].plot(ex_field, res.eval(x=ex_field), color=f'C{i}')

            axes[j, 1].errorbar(field, dphi, yerr=0.1*np.ones_like(dphi),
                                fmt='+', color=f'C{i}', label=f'Vg={g} V')

    for j in range(2):
        axes[j, 1].set_xlabel('Parallel field (mT)')
        axes[j, 1].set_ylabel('Phase difference (rad)')
        axes[j, 1].set_ylim((min(0, np.min(dphi)), None))
        axes[j, 1].legend()

    if ANALYSIS_PATH:
        fig.savefig(os.path.join(ANALYSIS_PATH, 'dphi.pdf'))


if not FIX_PHI_ZERO:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True,
                             constrained_layout=True)
    fig.suptitle('Phase difference')
    for i, f in enumerate(results['field']):
        for j in range(2):
            axes[0].errorbar(results['gate'][1:],
                             results[f'dphi_j{j+1}'][i, 1:],
                             yerr=0.,
                             fmt='+' if j == 0 else '*',
                             color=f'C{i}', label=f'JJ {j+1} By={f} mT')
        axes[0].set_xlabel('Gate voltage (V)')
        axes[0].set_ylabel('Phase difference (rad)')
        axes[0].legend()
    for i, g in enumerate(results['gate']):
        if i == 0:
            continue
        for j in range(2):
            field    = results['field']
            dphi     = results[f'dphi_j{j+1}'][:, i]
            axes[1].errorbar(field, dphi, yerr=0.,
                             fmt='+' if j == 0 else '*',
                             color=f'C{i}', label=f'JJ {j+1} Vg={g} V')
        axes[1].set_xlabel('Parallel field (mT)')
        axes[1].set_xlim((min(0, np.min(field)), None))
        axes[1].set_ylabel('Phase difference (rad)')
        axes[1].set_ylim((min(0, np.min(dphi)), None))
        axes[1].legend()

    if ANALYSIS_PATH:
        fig.savefig(os.path.join(ANALYSIS_PATH, 'fig4.pdf'))

plt.show()
