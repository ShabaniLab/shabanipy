# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit a set of data with a constrained SQUID model.

We consider here data in which both the positive and negative current branches
have been measured.

We pack together different gate voltage data and different parallel field. We
assume the following:
- one junction (idler) is always at the same gate -> its transparency is fixed
- at a given field the critical current and the phase of the idler is fixed
- at a given field the envelope of the squid oscillation is fixed

"""

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'pos_neg_j2_fig3.py'

#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/05'

#: Dictionary of parallel field, file path.
DATA_PATHS = {-300: 'Data_0513/JS124S_BM002_575.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {-300: (-4.8e-3, None)}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {-300: 0.2,}

#: Name/index of the gate column.
GATE_COLUMN = 2

#: Gate values for which to skip the analysis. The values should be present
#: in the datasets.
EXCLUDED_GATES = []

#: Name/index of the perpendicular field column.
FIELD_COLUMN = 3

#: Name/index of the bias current column.
BIAS_COLUMN = 0

#: Name/index of the counter current column used to determine the sweep
#: direction.
COUNTER_COLUMN = 1

#: Branch of the SQUID current measured for a given counter value
COUNTER_MEANING = {0.0: 'positive', 1.0: 'negative'}

#: Are data symmetric in bias or did we acquire only the interesting half
SYMMETRIC_DATA = {-300: True}

#: Name/column of the differential resistance column.
RESISTANCE_COLUMN = 4

#: Threshold value used to determine the switching current.
RESISTANCE_THRESHOLD = 1.4e-7

#: Should we plot the extracted switching current on top of the SQUID
#: oscillations
PLOT_EXTRACTED_SWITCHING_CURRENT = False

#: Should we enforce the equality of the transparencies.
EQUAL_TRANSPARENCIES = True

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = -1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 0.91

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Should the idler jj current be fixed across gates
FIX_IDLER_CURRENT = True

#: Should we plot the initial guess for each trace.
PLOT_INITIAL_GUESS = False

#: Should we plot the fit for each trace.
PLOT_FITS = 'color'

#: Should the fits be plotted vs the field rather than the phase
PLOT_FITS_FIELD = True

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
for f, ppath in DATA_PATHS.items():

    datasets[f] = {}
    datasets_color[f] = {}

    with LabberData(os.path.join(DATA_ROOT_FOLDER, ppath)) as data:

        frange = FIELD_RANGES[f]
        gates = [g for g in np.unique(data.get_data(GATE_COLUMN))
                 if g not in EXCLUDED_GATES and not np.isnan(g)]
        gates_number[f] = len(gates)

        if PLOT_EXTRACTED_SWITCHING_CURRENT:
            fig, axes = plt.subplots(gates_number[f], sharex=True,
                                     figsize=(10, 15),
                                     constrained_layout=True)
            fig.suptitle(f'Parallel field {f} mT')

        for i, gate in enumerate(gates):
            datasets[f][gate] = {}
            datasets_color[f][gate] = {}
            for counter, meaning in COUNTER_MEANING.items():
                filt = {GATE_COLUMN: gate, COUNTER_COLUMN: counter}
                field = data.get_data(FIELD_COLUMN, filters=filt)
                bias = data.get_data(BIAS_COLUMN, filters=filt)
                diff = data.get_data(RESISTANCE_COLUMN, filters=filt)
                diff = np.abs(diff)

                # Filter out Nan values
                mask = np.logical_not(np.any(np.isnan(diff), axis=-1))
                field, bias, diff = field[mask], bias[mask], diff[mask]

                if SYMMETRIC_DATA[f]:
                    # bias values are not sorted as a consequences we always
                    # want the second part of the acquired data
                    zero_bias_ind = np.argmin(np.abs(bias[0]))
                    field = field[:, zero_bias_ind:]
                    bias = bias[:, zero_bias_ind:]
                    diff = diff[:, zero_bias_ind:]

                rfield, curr = extract_switching_current(field, bias, diff,
                                                         RESISTANCE_THRESHOLD)

                # Make sure the field is an ascending order
                sort_index = np.argsort(rfield)
                rfield, curr = rfield[sort_index], curr[sort_index]
                field = field[sort_index]
                bias = bias[sort_index]
                diff = diff[sort_index]

                if any([v is not None for v in frange]):
                    masks = []
                    if frange[0] is not None:
                        masks.append(np.greater(rfield, frange[0]))
                    if frange[1] is not None:
                        masks.append(np.less(rfield, frange[1]))
                    index = (np.nonzero(np.logical_and(*masks))
                            if len(masks) == 2 else np.nonzero(masks[0]))
                    datasets[f][gate][meaning] = (rfield[index], curr[index])
                    datasets_color[f][gate][meaning] = (diff[index],
                                                        (field[index],
                                                         bias[index]))
                else:
                    datasets[f][gate][meaning] = (rfield, curr)
                    datasets_color[f][gate][meaning] = (diff,
                                                        (field, bias))

                if PLOT_EXTRACTED_SWITCHING_CURRENT:
                    axes[i].pcolormesh(field, bias, diff,
                                       vmin=0,
                                       vmax=np.max(diff[0, -1]))
                    axes[i].plot(*datasets[f][gate][meaning], color='C1')
                    if counter == 0.0:
                        axes[i].legend(title=f'Gate voltage {gate} V')

if PLOT_EXTRACTED_SWITCHING_CURRENT:
    plt.show()

# Setup the fit
params = Parameters()
if not FREQUENCY_PER_FIELD:
    params.add(f'phase_conversion')
    params.add(f'fraun_scale', value=0)

for i, f in enumerate(datasets):
    if FREQUENCY_PER_FIELD:
        params.add(f'phase_conversion_{i}')
        params.add(f'fraun_scale_{i}', value=0)
    if FIX_IDLER_CURRENT:
        params.add(f'I_idler_{i}')
    params.add(f'Boffset_{i}', vary=False)
    params.add(f'phi_idler_{i}', value=0, vary=False)
    params.add(f'fraun_offset_{i}', value=0.0)
    params.add(f't_idler_{i}', min=0.0, max=0.999, value=TRANSPARENCY_GUESS[f])
    if EQUAL_TRANSPARENCIES:
        params[f't_idler_{i}'].set(expr=f't_active_{i}')

    params.add(f't_active_{i}', min=0.0, max=0.999,
               value=TRANSPARENCY_GUESS[f])
    for j, gate in enumerate(datasets[f]):
        if not FIX_IDLER_CURRENT:
            params.add(f'I_idler_{i}_{j}')
        params.add(f'I_active_{i}_{j}')
        params.add(f'phi_active_{i}_{j}', value=0.0, min=-np.pi, max=np.pi)


def eval_squid_current(pfield, i, j, params, branch='positive'):
    """Compute the squid flowing as a function of the perpendicular field.

    """
    t_id = (params['t_idler'] if 't_idler' in params else
            params[f't_idler_{i}'])
    i_id = (params[f'I_idler_{i}'] if f'I_idler_{i}' in params else
            params[f'I_idler_{i}_{j}'])
    idler_params = (params[f'phi_idler_{i}'],
                    i_id,
                    t_id)
    phi_id = (params[f'phi_active_{i}_{j}']
              if f'phi_active_{i}_{j}' in params else
              params[f'phi_active_{i}'])
    active_params = (phi_id,
                     params[f'I_active_{i}_{j}'],
                     params[f't_active_{i}'])
    f = (pfield - params[f'Boffset_{i}'])
    fscale = (params[f'fraun_scale'] if 'fraun_scale' in params else
              params[f'fraun_scale_{i}'])
    fraun_phase = f*fscale + params[f'fraun_offset_{i}']
    fe = fraunhofer_envelope(fraun_phase)
    conversion = (params[f'phase_conversion']
                  if 'phase_conversion' in params else
                  params[f'phase_conversion_{i}'])
    sq = compute_squid_current(HANDEDNESS*f*conversion,
                               finite_transparency_jj_current,
                               idler_params,
                               finite_transparency_jj_current,
                               active_params,
                               positive=bool(branch=='positive'))
    return fe*sq


def target_function(params, datasets, index=None):
    """Target function used to fit multiple SQUID traces at once.

    """
    res = []
    params = params.valuesdict()
    for i, f in enumerate(datasets):
        if index is not None and i != index:
            continue
        for j, g in enumerate(datasets[f]):
            for branch, (pfield, curr) in datasets[f][g].items():
                model = eval_squid_current(pfield, i, j, params,
                                           branch=branch)
                res.append(model - curr)
    res = np.concatenate(res)
    return res

# Guess reasonable parameters
freq = {}
for f in datasets:
    freq[f] = np.empty((len(datasets[f]), 2))
    for j, g in enumerate(datasets[f]):
        for k, (branch, (pfield, curr)) in enumerate(datasets[f][g].items()):
            step = rfield[1] - rfield[0]
            period_index = np.argmax(np.abs(np.fft.rfft(curr)[1:])) + 1
            fft_freq = np.fft.fftfreq(len(curr), step)
            freq[f][j, k] = fft_freq[period_index]

if FREQUENCY_PER_FIELD:
    for i, f in enumerate(datasets):
        cf = (CONVERSION_FACTOR_CORRECTION[f]
              if isinstance(CONVERSION_FACTOR_CORRECTION, dict) else
              CONVERSION_FACTOR_CORRECTION)
        phi_conversion = (2*np.pi*np.average(freq[f]) * cf)
        params[f'phase_conversion_{i}'].value = phi_conversion
        params[f'fraun_scale_{i}'].value = phi_conversion / 60
else:
    phi_conversion = 2*np.pi*np.average([np.average(freq[f])
                                         for f in datasets])
    params['phase_conversion'].value = (phi_conversion *
                                        CONVERSION_FACTOR_CORRECTION)
    params['fraun_scale'].value = phi_conversion / 90

max_fields = []
for i, f in enumerate(datasets):
    i_idler = []
    i_active = {}
    field_at_max = []
    max_fields.append(field_at_max)
    for g in datasets[f]:
        avgc = 0
        amp = 0
        for branch, (rfield, curr) in datasets[f][g].items():
            if branch == 'positive':
                field_at_max.append(rfield[np.argmax(curr)])
            maxc, minc = np.amax(curr), np.amin(curr)
            avgc += abs((maxc + minc))/2
            amp += abs((maxc - minc))/2

        # Average the values extracted from both branches
        avgc /= 2
        amp /= 2

        # Assume that for the low gate the idler has a larger current
        if not i_idler:
            i_idler.append(max(amp, avgc))
            i_active[g] = min(amp, avgc)
        # Identify the idler current as the one closest to the one previously
        # identified
        else:
            if abs(avgc - i_idler[0]) < abs(amp - i_idler[0]):
                i_idler.append(avgc)
                i_active[g] = amp
            else:
                i_idler.append(amp)
                i_active[g] = avgc

    # Set the guessed values
    if FIX_IDLER_CURRENT:
        params[f'I_idler_{i}'].value = np.average(i_idler)
    for k, g in enumerate(datasets[f]):
        params[f'I_active_{i}_{k}'].value = i_active[g]
        if not FIX_IDLER_CURRENT:
            params[f'I_idler_{i}_{k}'].value = i_idler[k]

    # Set the field offset based on the lowest gate
    params[f'Boffset_{i}'].value = field_at_max[0]

# To estimate the phase compare the position of the maximum in the data and in
# the model.
for i, f in enumerate(datasets):
    for k, g in enumerate(datasets[f]):
        rfield, curr = datasets[f][g]['positive']
        model_curr = eval_squid_current(rfield, i, k,
                                        params.valuesdict())
        f_index = np.argmin(np.abs(rfield - max_fields[i][k]))
        phi_conversion = (params['phase_conversion'].value
                          if 'phase_conversion' in params else
                          params[f'phase_conversion_{i}'].value)
        period = int(2*np.pi/phi_conversion/abs(rfield[1] - rfield[0]))
        mask = slice(max(0, f_index - period//2), f_index + period//2)
        max_model = np.argmax(model_curr[mask])
        phi = phi_conversion*(max_fields[i][k] -
                              rfield[mask][max_model])
        params[f'phi_active_{i}_{k}'].value = HANDEDNESS * phi

if PLOT_INITIAL_GUESS:
    # Plot the initial guesses to check our automatic guesses
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f], sharex=True,
                                 figsize=(10, 15),
                                 constrained_layout=True)
        fig.suptitle(f'Parallel field {f} mT')
        for j, g in enumerate(datasets[f]):
            for branch, (rfield, curr) in datasets[f][g].items():
                axes[j].plot(rfield, curr)
                axes[j].plot(rfield,
                            eval_squid_current(rfield, i, j,
                                               params.valuesdict(),
                                               branch))
            axes[j].axvline(max_fields[i][j])
            axes[j].set_title(f'Gate voltage {g} V')
    plt.show()

# Perform the fit
if not FREQUENCY_PER_FIELD:
    result = minimize(target_function, params, args=(datasets,),
                      method='leastsq')
    params = result.params
if FREQUENCY_PER_FIELD:
    # Fit one field at a time to allow for faster processing.
    results = {}
    for i, f in enumerate(datasets):
        print(f'Treating data at {f} mT')
        varying_p = [f'phase_conversion_{i}', f'fraun_scale_{i}',
                     f'Boffset_{i}', f'fraun_offset_{i}',
                     f't_idler_{i}', f't_active_{i}']
        varying_p += [f'I_active_{i}_{j}'
                      for j, gate in enumerate(datasets[f])]
        varying_p += [f'phi_active_{i}_{j}'
                        for j, gate in enumerate(datasets[f])]
        if FIX_IDLER_CURRENT:
            varying_p += [f'I_idler_{i}']
        else:
            varying_p += [f'I_idler_{i}_{j}'
                          for j, gate in enumerate(datasets[f])]

        fit_params = Parameters()
        fit_params.add(f'phi_idler_{i}', 0, vary=False)
        for name in varying_p:
            p = params[name]
            fit_params.add(name, p.value,
                           True if name.startswith('phi') else False,
                           p.min, p.max)
        fit_params[varying_p[0]].set(vary=True)
        # Try to first adjust the frequency and phase
        result = minimize(target_function, fit_params, args=(datasets, i),
                          method='nelder')
        fit_params[varying_p[0]].set(value=result.params[varying_p[0]].value)

        # Adjust all parameters
        for name in varying_p:
            fit_params[name].set(vary=True)
        result = minimize(target_function, fit_params, args=(datasets, i))
        for p in varying_p:
            results[p] = result.params[p]

if PLOT_FITS:
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f],
                                 figsize=(8, 12), sharex=True,
                                 constrained_layout=True)
        fig.suptitle(f'Parallel field {f} mT\n')
        color_max = 1.2*max([np.max(datasets_color[f][g]['positive'][0][0, -1])*1e8
                            for g in datasets[f]])
        for k, g in enumerate(datasets[f]):

            # Plot each branch
            for branch, (field, curr) in datasets[f][g].items():
                name = ('phase_conversion' if 'phase_conversion' in params else
                        f'phase_conversion_{i}')
                phase = ((field - params[f'Boffset_{i}']) *
                         params[name].value/np.pi)
                if PLOT_FITS == 'color':
                    diff, (x, bias) = datasets_color[f][g][branch]
                    if PLOT_FITS_FIELD:
                        x = (x / 18.2 * 1000)
                    else:
                        x = ((x - params[f'Boffset_{i}']) *
                            params[name].value/np.pi)
                    im = axes[-k-1].pcolormesh(x, bias, diff*1e8,
                                               vmin=0,
                                               vmax=color_max,
                                               rasterized=True)
                else:
                    axes[-k-1].plot(phase, curr, '+', color='C0')
                model = eval_squid_current(field, i, k,
                                            params.valuesdict(), branch)
                if PLOT_FITS_FIELD:
                    x = (field / 18.2 * 1000)
                else:
                    x = phase
                axes[-k-1].plot(x, model, color='C1')

            if not PLOT_FITS_FIELD:
                axes[-k-1].xaxis.set_major_formatter(plt.FuncFormatter(format_phase))
            axes[-k-1].tick_params(direction='in', width=1.5)
            axes[-k-1].legend(title=f'Vg2 = {g} V', loc=1)

        if PLOT_FITS_FIELD:
            axes[-1].set_xlabel('Out of plane field (mT)')
        else:
            axes[-1].set_xlabel('SQUID phase')
        axes[len(datasets[f])//2].set_ylabel('Bias current (µA)')

        # Add a colorbar if pertinent
        if PLOT_FITS == 'color':
            cb = plt.colorbar(im, ax=axes[len(datasets[f])//2])
            cb.set_label('Resistance (Ω)')
        if ANALYSIS_PATH:
            fig.savefig(os.path.join(ANALYSIS_PATH, f'fit_f_{f}.pdf'))

    plt.show()

# Build result arrays from the fitted parameters
results = {}
results['field'] = np.array(list(datasets))
if FREQUENCY_PER_FIELD:
    name = 'phase_conversion'
    results['period'] = np.array([2*np.pi/params[name + f'_{i}'].value
                                  for i, _ in enumerate(datasets)])
else:
    print(2*np.pi/params['phase_conversion'].value)
for name in ('I_idler', 't_idler', 't_active'):
    if not FIX_IDLER_CURRENT and name == 'I_idler':
        continue
    results[name] = np.array([params[name + f'_{i}'].value
                              for i, _ in enumerate(datasets)])

results['gate'] = np.unique([list(datasets[f]) for f in datasets])
for name in ('I_idler', 'I_active', 'phi_active'):
    if FIX_IDLER_CURRENT and name == 'I_idler':
        continue
    results[name] = np.array([[params[name + f'_{i}_{j}'].value
                               for j, _ in enumerate(datasets[f])]
                              for i, f in enumerate(datasets)])

# Substract the phase at the lowest gate to define the phase  difference.
results['dphi'] = - (PHASE_SIGN * (results['phi_active'].T -
                                    results['phi_active'][:, 0]).T)
results[f'dphi'] %= 2*np.pi
mask = np.greater(results[f'dphi'], np.pi).astype(int)
results[f'dphi'] -= 2*np.pi*mask

# Save the data if a file was provided.
if ANALYSIS_PATH:
    with h5py.File(os.path.join(ANALYSIS_PATH, 'results.h5'), 'w') as storage:
        # storage.attrs['periodicity'] = 2*np.pi/params['phase_conversion']
        storage.attrs['res_threshold'] = RESISTANCE_THRESHOLD
        for k, v in results.items():
            storage[k] = v

# Plot meaningful results summary.

# Idler parameters
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5),
                         constrained_layout=True)
fig.suptitle('Idler junction parameters')
sort_index = np.argsort(results['field'])
if FIX_IDLER_CURRENT:
    axes[0].plot(results['field'][sort_index], results['I_idler'][sort_index])
else:
    for i, g in enumerate(results['gate']):
        axes[0].plot(results['field'][sort_index],
                    results['I_idler'][sort_index][:, i],
                    label=f'Vg={g} mT')
    axes[0].legend()
axes[0].set_xlabel('Parallel field (mT)')
axes[0].set_ylabel('Idler JJ current (µA)')
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
    phi = results['phi_active'][:, i]
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
        mask = np.less(field, 450)
        p = model.guess(dphi[mask], x=field[mask])
        res = model.fit(dphi[mask], p, x=field[mask])
        ex_field = np.linspace(min(0, np.min(field)), max(field))
        axes[1].plot(ex_field, res.eval(x=ex_field), color=f'C{i}')

    axes[1].errorbar(field, dphi, yerr=0.15,
                        fmt='+', color=f'C{i}', label=f'Vg={g} V')

axes[1].set_xlabel('Parallel field (mT)')
axes[1].set_ylabel('Phase difference (rad)')
axes[1].set_ylim((0 if np.min(dphi) > 0 else None, None))
axes[1].legend()

if ANALYSIS_PATH:
    fig.savefig(os.path.join(ANALYSIS_PATH, 'dphi.pdf'))

if FREQUENCY_PER_FIELD:
    plt.figure()
    plt.plot(results['field'], results['period'])

plt.show()
