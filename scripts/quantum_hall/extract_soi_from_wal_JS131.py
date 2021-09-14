# -*- coding: utf-8 -*-
"""Extract the soi from low field measurements.

The density and mobility are extracted at the same time.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r'/Users/mdartiailh/Labber/Data/2018/12/Data_1218/JS131HB_207_BM001_019.hdf5'

#: Index or name of the column containing the gate voltage values.
GATE_COLUMN = 1

#: Index or name of the column containing the applied magnetic field.
FIELD_COLUMN = 0

# WARNING Labber uses 2 columns per lock-in value so you should be careful when
# using indexes. The index is always assumed to refer to the first column of
# the lock-in (ie real values)

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along x.
XX_VOLTAGE_COLUMN = 4

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along y. This data will only be used its XX counterpart is not
#: provided.
YY_VOLTAGE_COLUMN = None

#: Index or name of the column contaning the transverse voltage drop
#: measurement.
XY_VOLTAGE_COLUMN = 2

#: Component of the measured voltage to use for analysis.
#: Recognized values are 'real', 'imag', 'magnitude'
LOCK_IN_QUANTITY = 'real'

#: Value of the excitation current used by the lock-in amplifier in A.
PROBE_CURRENT = 1e-6

#: Sample geometry used to compute the mobility.
#: Accepted values are 'Van der Pauw', 'Standard Hall bar'
GEOMETRY = 'Standard Hall bar'

#: Magnetic field bounds to use when extracting the density.
FIELD_BOUNDS = (-2, -40e-3)

#: Should we plot the fit used to extract the density at each gate.
PLOT_DENSITY_FIT = False

#: Parameters to use to filter the xx and yy data. The first number if the
#: number of points to consider IT MUST BE ODD, the second the order of the
#: polynomial used to smooth the data
FILTER_PARAMS = (11, 3)

#: Should we plot the smoothed data to validate the filter parameters.
PLOT_SMOOTHED = False

#: Method used to symmetrize the wal data.
#: Possible values are: 'average', 'positive', 'negative'
SYM_METHOD = 'average'

#: Model to use to fit the WAL.
#: Acceptable values are 'simplified', 'full'
WAL_MODEL = 'simplified'

#: Fitting method used after a first fit using nelder-meald.
FIT_METHOD = 'least_squares'

#: Reference field to use in the WAL calculation.
WAL_REFERENCE_FIELD = 0.0001

#: Maximal field to consider in WAL fitting procedure
WAL_MAX_FIELD = 90e-3

#: Truncation used in the WAL calculation. Meaning depends on the model used.
WAL_TRUNCATION = 10000

#: Weighting method to use to fir the data.
#: Acceptable values are : 'exp', 'gauss', 'peak-gauss', 'lorentz'
WEIGHT_METHOD = 'gauss'

#: Stiffness of the weight function.
WEIGHT_STIFFNESS = 0.5

#: Fixed Dresselhaus contribution, use None to allow a varying dresselhaus
#: term.
CUBIC_DRESSELHAUS = 2.65e-6

#: Should we plot the Htr field that fix a bound on the validity as a function
#: of the gate.
PLOT_HTR = False

#: Should we plot the WAL fits.
PLOT_WAL = True

#: Effective mass of the carriers in unit of the electron mass.
EFFECTIVE_MASS = 0.03

#: File in which to store the results of the analysis as a function of gate
#: voltage.
RESULT_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/WAL/JS131/'
               'average_rashba_only/JS131HB_207_BM001_019_wal_analysis_avg.csv')

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as cs
from scipy.signal import savgol_filter

from shabanipy.quantum_hall.conversion\
    import (convert_lock_in_meas_to_diff_res, GEOMETRIC_FACTORS,
            htr_from_mobility_density,
            diffusion_constant_from_mobility_density)
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility
from shabanipy.quantum_hall.wal.fitting import (extract_soi_from_wal,
                                                estimate_parameters)
from shabanipy.quantum_hall.wal.utils import (flip_field_axis,
                                              recenter_wal_data,
                                              symmetrize_wal_data,
                                              compute_linear_soi,
                                              compute_dephasing_time)
from shabanipy.utils.labber_io import LabberData

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'pdf.fonttype': 42})


with LabberData(PATH) as data:

    names = data.channel_names
    shape = data.compute_shape((GATE_COLUMN, FIELD_COLUMN))

    gate = data.get_data(GATE_COLUMN)

    # Handle interruptions in the last scan.
    while len(gate) < shape[0]*shape[1]:
        shape[1] -= 1

    length = shape[0]*shape[1]

    gate = gate.reshape(shape).T[:-1]
    field = data.get_data(FIELD_COLUMN).reshape(shape).T[:-1]
    res = dict.fromkeys(('xx', 'yy', 'xy'))
    for k in list(res):
        name = globals()[f'{k.upper()}_VOLTAGE_COLUMN']
        if name is None:
            continue
        index = data.name_or_index_to_index(name)
        if LOCK_IN_QUANTITY == 'real':
            val = data.get_data(index)
        elif LOCK_IN_QUANTITY == 'imag':
            val = data.get_data(index+1)
        else:
            val = data.get_data(index)**2 + data.get_data(index+1)**2
        val = val[:length].reshape(shape).T[:-1]
        res[k] = convert_lock_in_meas_to_diff_res(val, PROBE_CURRENT)

    if res['xx'] is None:
        res['xx'] = res['yy']
    if res['yy'] is None:
        res['yy'] = res['xx']

# Start with smoothing the data if we were asked to and recentering the data
# in field.
# In all the following, we ignore the 'yy' axis.
gate = gate[:, 0]
flip = bool(gate[0] < gate[-1])
flip_field_axis(field, res['xx'], res['xy'])
res['original'] = res['xx'].copy()
if FILTER_PARAMS:
    res['xx'] = savgol_filter(res['xx'], *FILTER_PARAMS)

if flip:
    gate = gate[::-1]
    field = field[::-1]
    for k in res:
        res[k] = res[k][::-1]

# Filter twice when determining the field offset.
field, _ = recenter_wal_data(field,
                             savgol_filter(res['xx'], *FILTER_PARAMS),
                             0.1, 10)

if PLOT_SMOOTHED:
    f, axes = plt.subplots(1, 5, sharex=True, figsize=(12, 4),
                           constrained_layout=True)
    l = len(gate)
    for ax, index in zip(axes, [0, l//4, l//2, 3*l//4, -1]):
        ax.plot(field[index]*1e3, res['original'][index], '+')
        ax.plot(field[index]*1e3, res['xx'][index])
        ax.axvline(0)
        ax.set_xlabel('Field (mT)')
        ax.set_ylabel('Resistance (Ω)')
    plt.show()

# Extract the density and mobility and compute useful quantities.
density = extract_density(field, res['xy'], FIELD_BOUNDS, PLOT_DENSITY_FIT)
if PLOT_DENSITY_FIT:
    plt.show()

mobility, std_mob = extract_mobility(field, res['xx'], res['yy'], density,
                                     GEOMETRIC_FACTORS[GEOMETRY])
density, std_density = density

mass = EFFECTIVE_MASS*cs.electron_mass
htr = htr_from_mobility_density(mobility, density, mass)
diff = diffusion_constant_from_mobility_density(mobility, density,
                                                mass)

# If requested plot the Htr that tells us in what range the WAL can be valid.
if PLOT_HTR:
    f, axes = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
    for ax, x, y, x_lab, y_lab in zip(axes, (gate, density/1e4, density/1e4),
                                      (density/1e4, mobility*1e4, htr*1e3),
                                      ('Gate voltage (V)',
                                       'Density (cm$^{-2}$)',
                                       'Density (cm$^{-2}$)'),
                                      ('Density (cm$^{-2}$)',
                                       'Mobility (cm$^{-2}$V${^-1}$s$^{-1}$)',
                                       'Htr (mT)')):
        ax.plot(x, y, '+')
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
    plt.show()

# Symmetrize the WAL before fitting
field, res['xx'] = symmetrize_wal_data(field, res['xx'], SYM_METHOD)

# Attempt to fit the data
deph, lin_soi, cu_soi = extract_soi_from_wal(field[:], res['xx'][:],
                                             WAL_REFERENCE_FIELD,
                                             WAL_MAX_FIELD,
                                             WAL_MODEL, WAL_TRUNCATION,
                                             guesses=(0.01, 0.02, 0.0, 0.0),
                                             plot_fit=PLOT_WAL,
                                             method=FIT_METHOD,
                                             weigth_method=WEIGHT_METHOD,
                                             weight_stiffness=WEIGHT_STIFFNESS,
                                             htr=htr,
                                             cubic_soi=CUBIC_DRESSELHAUS,
                                             density=density,
                                             plot_path=os.path.dirname(RESULT_PATH))

# If requested plot all the fits to validate them
if PLOT_WAL:
    plt.show()

# Plot the bare fitted quantities as a function of the density.
size = 4 if WAL_MODEL == 'full' else 3
f, axes = plt.subplots(1, size, figsize=(10, 5), constrained_layout=True)
ys = np.array([deph[0], lin_soi[0][0], lin_soi[1][0], cu_soi[0]]
              if size == 4 else
              [deph[0], lin_soi[0][0], cu_soi[0]])
y_labels = ['Dephasing field (T)', 'Rashba SOI field (T)',
            'Dresselhaus SOI field (T)', 'Cubic Dresselhaus (T)']
if size == 3:
    del y_labels[2]
for ax, x, y, x_lab, y_lab in zip(axes, [density/1e4]*size, ys,
                                  ['Density (cm$^{-2}$)']*size,
                                  y_labels):
    ax.plot(x, y, '+')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
plt.show()

# Compute dephasing time and linear SOI strength
deph_time = compute_dephasing_time(deph, diff)
lin_soi_str = compute_linear_soi(lin_soi, mobility, density, mass)

# Plot the fitted quantities as a function of the density.
size = 4 if WAL_MODEL == 'full' else 3
f, axes = plt.subplots(1, size, figsize=(10, 5), constrained_layout=True)
ys = np.array([deph_time[0], lin_soi_str[0][0], lin_soi_str[1][0], cu_soi[0]]
              if size == 4 else
              [deph_time[0], lin_soi_str[0][0], cu_soi[0]])
y_labels = ['Dephasing time (ps)', 'Rashba SOI (meV.A)',
            'Dresselhaus SOI (meV.A)', 'Cubic Dresselhaus (T)']
if size == 3:
    del y_labels[2]
for ax, x, y, x_lab, y_lab in zip(axes, [density/1e4]*size, ys,
                                  ['Density (cm$^{-2}$)']*size,
                                  y_labels):
    ax.plot(x, y, '+')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
plt.show()

# Plot the Rashba component and the mobility
f, axes = plt.subplots(2, 1, constrained_layout=True)
ys = [mobility*1e4, lin_soi_str[0][0]]
y_labels = ['Mobility (cm$^{-2}$V${^-1}$s$^{-1}$)', 'Rashba SOI (meV.A)']
for ax, x, y, y_err, x_lab, y_lab in zip(axes, [density/1e4]*2, ys,
                                         [std_mob*1e4, lin_soi_str[0][1]],
                                         ['Density (cm$^{-2}$)']*2,
                                          y_labels):
    ax.plot(x, y, '+')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
plt.show()

if RESULT_PATH:

    df = pd.DataFrame({'Gate voltage (V)': gate,
                       'Density (m^-2)': density,
                       'Stderr density (m^-2)': std_density,
                       'Mobility (m^2V^-1s^-1)': mobility,
                       'Stderr mobility (m^2V^-1s^-1)': std_mob,
                       'Diffusion (m^2/s)': diff,
                       'Htr (T)': htr,
                       'Dephasing time (ps)': deph_time[0],
                       'Stderr dephasing time (ps)': deph_time[1],
                       'Rashba SOI (meV.A)': lin_soi_str[0][0],
                       'Stderr Rashba SOI (meV.A)': lin_soi_str[0][1],
                       'Dresselhaus SOI (meV.A)': lin_soi_str[1][0],
                       'Stderr dresselhaus SOI (meV.A)': lin_soi_str[1][1],
                       'Stderr cubic Dresselhaus (T)': cu_soi[1]})
    with open(RESULT_PATH, 'w') as f:
        f.write(f'# Probe-current: {PROBE_CURRENT}\n'
                f'# Effective mass: {EFFECTIVE_MASS}\n'
                f'# Geometry: {GEOMETRY}\n'
                f'# Lock-in quantity: {LOCK_IN_QUANTITY}\n'
                f'# Filter params: {FILTER_PARAMS}\n'
                f'# Sym method: {SYM_METHOD}\n'
                f'# WAL model: {WAL_MODEL}\n'
                f'# WAL reference field: {WAL_REFERENCE_FIELD}\n'
                f'# WAL max field: {WAL_MAX_FIELD}\n'
                f'# Weight method: {WEIGHT_METHOD}\n'
                f'# Weight stiffness: {WEIGHT_STIFFNESS}\n'
                f'# Cubic Dresselhaus: {CUBIC_DRESSELHAUS}\n')
        df.to_csv(f, index=False)
