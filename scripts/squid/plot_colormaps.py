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
CONFIG_NAME = 'j1_phaseshift_all_by_CPR.py'

#: Common folder in which the data file are related.
DATA_ROOT_FOLDER = '/Users/mdartiailh/Labber/Data/2019/04'

#: Dictionary of parallel field, file path.
DATA_PATHS = {400: 'Data_0405/JS124S_BM002_465.hdf5'}

#: Perpendicular field range to fit for each parallel field
FIELD_RANGES = {400: (-2.28e-3, None)}

#: Guess for the transparency of the junctions as a function of the field.
TRANSPARENCY_GUESS = {400: 0.01,}

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

#: Should we enforce the equality of the transparencies.
EQUAL_TRANSPARENCIES = True

#: Sign of the phase difference created by the perpendicular field.
PHASE_SIGN = 1

#: Handedness of the system.
HANDEDNESS = 1

#: Correction factor to apply on the estimated pulsation
CONVERSION_FACTOR_CORRECTION = 1.03

#: Allow different frequency for each field
FREQUENCY_PER_FIELD = False

#: Fix the anomalous phase to 0.
FIX_PHI_ZERO = False

#: Should the idler jj current be fixed across gates
FIX_IDLER_CURRENT = True

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
from shabanipy.utils.plotting import format_phase

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

gates_number = {}
datasets = {}
datasets_color = {}

# Load and filter all the datasets
for f, ppath in DATA_PATHS.items():

    datasets[f] = {}
    datasets_color[f] = {}

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
                datasets[f][gate] = (rfield[index], curr[index])
                datasets_color[f][gate] = (diff[index],
                                           (rfield[index][0],
                                            rfield[index][-1],
                                            bias[0, 0],
                                            bias[0, -1]))
            else:
                datasets[f][gate] = (rfield, curr)
                datasets_color[f][gate] = (diff,
                                           (rfield[0], rfield[-1],
                                            bias[0, 0], bias[0, -1]))

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
    if FIX_PHI_ZERO:
        params.add(f'phi_active_{i}', value=0.0, min=-np.pi, max=np.pi)
    for j, gate in enumerate(datasets[f]):
        if not FIX_IDLER_CURRENT:
            params.add(f'I_idler_{i}_{j}')
        params.add(f'I_active_{i}_{j}')
        if not FIX_PHI_ZERO:
            params.add(f'phi_active_{i}_{j}', value=0.0, min=-np.pi, max=np.pi)


def eval_squid_current(pfield, i, j, params):
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
                               active_params)
    return fe*sq


# Guess reasonable parameters
freq = {}
for f in datasets:
    freq[f] = []
    for rfield, curr in datasets[f].values():
        step = rfield[1] - rfield[0]
        period_index = np.argmax(np.abs(np.fft.rfft(curr)[1:])) + 1
        fft_freq = np.fft.fftfreq(len(curr), step)
        freq[f].append(fft_freq[period_index])

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
        rfield, curr = datasets[f][g]
        field_at_max.append(rfield[np.argmax(curr)])
        maxc, minc = np.amax(curr), np.amin(curr)
        avgc = (maxc + minc)/2
        amp = (maxc - minc)/2
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
        rfield, curr = datasets[f][g]
        model_curr = eval_squid_current(rfield, i, k,
                                        params.valuesdict())
        f_index = np.argmin(np.abs(rfield - max_fields[i][k]))
        period = int(2*np.pi/phi_conversion/abs(rfield[1] - rfield[0]))
        mask = slice(max(0, f_index - period//2), f_index + period//2)
        max_model = np.argmax(model_curr[mask])
        phi = phi_conversion*(max_fields[i][k] -
                              rfield[mask][max_model])
        params[f'phi_active_{i}_{k}'].value = - phi

if PLOT_FITS:
    for i, f in enumerate(datasets):
        fig, axes = plt.subplots(gates_number[f],
                                 figsize=(6, 15), sharex=True,
                                 constrained_layout=True)
        fig.suptitle(f'Parallel field {f} mT\n')
        color_max = max([np.max(datasets_color[f][g][0][0, -1])*1e8
                         for g in datasets[f]])
        for k, g in enumerate(datasets[f]):
            field, curr = datasets[f][g]
            name = ('phase_conversion' if 'phase_conversion' in params else
                    f'phase_conversion_{i}')
            phase = ((field - params[f'Boffset_{i}']) *
                      params[name].value/np.pi)
            if PLOT_FITS == 'color':
                diff, extent = datasets_color[f][g]
                f0 = ((extent[0] - params[f'Boffset_{i}']) *
                      params[name].value/np.pi)
                f1 = ((extent[1] - params[f'Boffset_{i}']) *
                      params[name].value/np.pi)
                im = axes[-k-1].imshow(diff.T*1e8,
                                       extent=(f0, f1, extent[2], extent[3]),
                                       origin='lower',
                                       aspect='auto',
                                       vmin=0,
                                    #    vmax=color_max,
                                       )
            else:
                axes[-k-1].plot(phase, curr, '+')
            model = eval_squid_current(field, i, k,
                                       params.valuesdict())
            if PLOT_INITIAL_GUESS:
                axes[-k-1].plot(phase, model, color='C1')
            axes[-k-1].xaxis.set_major_formatter(plt.FuncFormatter(format_phase))
            axes[-k-1].tick_params(direction='in', width=1.5)
            axes[-k-1].legend(title=f'Vg2 = {g} V', loc=1)
        axes[-1].set_xlabel('SQUID phase')
        axes[len(datasets[f])//2 + 1].set_ylabel('Bias current (µA)')
        if PLOT_FITS == 'color':
            cb = plt.colorbar(im, ax=axes[len(datasets[f])//2 + 1])
            cb.set_label('Resistance (Ω)')
        if ANALYSIS_PATH:
            fig.savefig(os.path.join(ANALYSIS_PATH, f'fit_f_{f}.pdf'))
    plt.show()

plt.show()
