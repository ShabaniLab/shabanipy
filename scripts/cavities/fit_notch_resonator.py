# -*- coding: utf-8 -*-
"""Fit a notch resonator resonance.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path to the HDF5 storing the data.
PATH = '/Users/goss/Desktop/Shabani/data/test/InPG_JY001_005.hdf5'

#: Name or index of the central frequency column.
FREQ_COLUMN = 0

#: Name or index of the column containing the input power. Use None if the
#: power is kept constant.
POWER_COLUMN = 1

#: Index of the resonances to fit and parameters for the fit:
#: - kind of resonance ('min', 'max')
#: - number of points over which to smooth the data
#: - number of points to use to correct the electrical delay
#: - smoothing coefficient for extracting the baseline
#: - should we fit both amplitude and phase
#: - optimization method to use
RESONANCE_PARAMETERS = {
    1: ('min', 51, 500, 1e12, 'full', 'nelder'),
    # 2: ('max', 51, 1000, 0, 'amplitude', 'leatsq'),
    # 3: ('max', 51, 400, 1e11, 'amplitude', 'nelder'),
    # 4: ('max', 51, 1000, 0, 'amplitude', 'leastsq'),
    # 5: ('max', 51, 400, 1e13, 'amplitude', 'leastsq'),
    }

#: Should we plot the extracted baseline
PLOT_BASELINE = True

#: Should we plot the extracted phase slope.
PLOT_PHASE_SLOPE = True

#: Should we plot the initial guess
PLOT_INITIAL_GUESS = True

#: Should we plot each fit.
PLOT_FITS = True

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit import Parameters, minimize, fit_report

from shabanipy.cavities.notch_geometry import fano_notch
from shabanipy.cavities.utils import (extract_baseline,
                                      estimate_central_frequency,
                                      estimate_peak_amplitude,
                                      estimate_width,
                                      estimate_time_delay,
                                      correct_for_time_delay)
from shabanipy.utils.labber_io import LabberData

with LabberData(PATH) as data:
    shape = data.compute_shape((FREQ_COLUMN, POWER_COLUMN))
    shape = [shape[1], shape[0]]
    powers = np.unique(data.get_data(POWER_COLUMN))

with h5py.File(PATH) as f:
    freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
    real  = f['Traces']['VNA - S21'][:, 0].reshape([-1] + shape)
    imag  = f['Traces']['VNA - S21'][:, 1].reshape([-1] + shape)
    amp = np.abs(real + 1j*imag)
    phase = np.arctan2(imag, real)

for res_index, res_params in RESONANCE_PARAMETERS.items():

    kind, sav_f, e_delay, base_smooth, method, fit_method = res_params
    for p_index, power in enumerate(powers):

        f = savgol_filter(freq[:, p_index, res_index],
                          sav_f, 3)
        a = savgol_filter(amp[:, p_index, res_index],
                          sav_f, 3)
        phi = savgol_filter(phase[:, p_index, res_index],
                            sav_f, 3)

        if base_smooth:
            base = extract_baseline(a,
                                    0.8 if kind == 'min' else 0.2,
                                    base_smooth, plot=PLOT_BASELINE)
        if PLOT_BASELINE:
            plt.show()

        # Make it so that once the baseline is taken into account we have a
        # transmission of 1
        if base_smooth:
            a /= (base/base[0])

        slope = estimate_time_delay(f, phi,
                                    e_delay, PLOT_PHASE_SLOPE)
        if PLOT_PHASE_SLOPE:
            plt.show()

        phi = correct_for_time_delay(f, phi, slope)

        fc = estimate_central_frequency(f, a, kind)
        width = estimate_width(f, a, kind)
        indexes = slice(np.argmin(np.abs(f - fc + 10*width)),
                        np.argmin(np.abs(f - fc - 10*width)))
        f = f[indexes]
        a = a[indexes]
        phi = phi[indexes]

        params = Parameters()
        params.add('f_c', value=fc, min=0)
        params.add('width', value=width, min=0)
        params.add('amplitude', value=estimate_peak_amplitude(a, kind), min=0)
        params.add('phase', value=0 if kind == 'min' else np.pi,
                   min=-np.pi, max=np.pi)
        params.add('baseline', value=a[0], min=0)
        params.add('phase_offset', value=0, vary=(method != 'amplitude'),
                   min=-np.pi, max=np.pi)
        params.add('fano_amp', value=0.01, min=0)
        params.add('fano_phase', value=0, min=-np.pi, max=np.pi)
        params.add('use_complex',
                   value=(method != 'amplitude'),
                   vary=False)

        if PLOT_INITIAL_GUESS:
            fig, axes = plt.subplots(1, 2, sharex=True)
            kwargs = params.valuesdict()
            kwargs.pop('use_complex')
            model = fano_notch(f, **kwargs)

            axes[0].plot(f, a, '+')
            axes[0].plot(f, np.abs(model))
            axes[1].plot(f, phi, '+')
            axes[1].plot(f, np.arctan2(model.imag, model.real))
            plt.show()

        def residual(params, freq, amp, phase):
            kwargs = params.valuesdict()
            use_complex = kwargs.pop('use_complex')
            model = fano_notch(freq, **kwargs)
            if use_complex:
                data = amp*np.exp(1j*phase)
                return np.concatenate([model.real - data.real,
                                       model.imag - data.imag])
            else:
                return np.abs(model) - amp

        res = minimize(residual, params,
                       args=(f, a, phi),
                       method=fit_method)
        print(fit_report(res))

        if PLOT_FITS:
            fig, axes = plt.subplots(1, 2, sharex=True)
            kwargs = res.params.valuesdict()
            kwargs.pop('use_complex')
            model = fano_notch(f, **kwargs)

            axes[0].plot(f, a, '+')
            axes[0].plot(f, np.abs(model))
            axes[1].plot(f, phi, '+')
            axes[1].plot(f, np.arctan2(model.imag, model.real))
            plt.show()
