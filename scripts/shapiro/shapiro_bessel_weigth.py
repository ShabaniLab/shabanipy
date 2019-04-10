# -*- coding: utf-8 -*-
"""Bin the shapiro step data (current vs power) for different parallel fields

The plot is done as a function of power and frequency and the power is
normalized by the power at which the step 0 disappear.

The plot use the folling axes:
- x axis: Normalized power
- y axis: Frequency
- color axis: bin count in current

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Labber directory to walk to find all the data files to analyse
LABBER_DIRECTORY = '/Users/mdartiailh/Labber/Data/2019'

#: CSV file containing the frequencies, gate, fields etc associated to each
#: measurement.
#: The expected column names are:
#: Frequency, Gate voltage V, Parallel field mT, Critical current,
#: Normal resistance, Ic file determination, Rn file determination,
#: Attenuation, Shapiro, Comment
CSV_SUMMARY_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
                    'Shapiro/2019-01/2019-data-summary.csv')

#: Path of the directory in which to store the results.
RESULT_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
               'Shapiro/2019-01/StepWidthAnalysis')

#: Index of the setps for which to generate a plot.
STEP_INDEXES = [0, 1, 2, 3, 4]

#: Name of the column containing the frequency for scans in which multiple
#: frequencies exist.
FREQUENCY_NAME = ('SC_C - Frequency', 'EXG - Frequency')

#: Name of the column containing the gate voltage for scans in which multiple
#: gate voltages exist.
GATE_NAME = ('Keithley 1 - Source voltage', )

#: Name of the column containing the parallel field for scans in which multiple
#: parallel fields exist.
FIELD_NAME = ('Magnet - By', )

#: Name or index of the column containing the power data
POWER_NAME = ('SC_C - Amplitude', 'EXG - Power')

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CURRENT_NAME = 'Yoko 1 - Voltage'

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 'DMM 1 - Value'

#: Number of points on which to average to correct the offset in the measured
#: voltage. Use zero to not correct.
CORRECT_VOLTAGE_OFFSET = 20

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Fraction of a shapiro step used for binning
STEP_FRACTION = 0.1

#: Threshold as fraction of the low power step used to identify the normalizing
#: power, defined at the first power for which the count of the step 0 is below
#: the threshold.
NORMALIZING_THRESHOLD = 0.05

#: Should the plots allowing to check the normalizing power be displayed.
PLOT_NORM_POWER_CHECK = False

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shabanipy.shapiro import normalize_db_power
from shabanipy.shapiro.binning import (bin_power_shapiro_steps,
                                       extract_step_weight)
from shabanipy.utils.labber_io import LabberData
from shabanipy.utils.file_discovery import list_all_files, filter_files

summary = pd.read_csv(CSV_SUMMARY_PATH)

# Match only the files corresponding to actual Shapiro steps data.
file_ids = [f'{s:03d}' for s in set(summary['Shapiro'])]
pattern = f"JS124S_BM002_({'|'.join(file_ids)}).hdf5$"

# Identify the measurement by their number to allow to easily retrieve the
# matching parameters.
paths = {int(os.path.split(path)[1][-8:-5]): path
         for path in filter_files(list_all_files(LABBER_DIRECTORY), pattern)}

# Create the analysis summary that contains additional information compared to
# to the data summary.
analysis_summary = {'Meas id': [], 'Frequency': [], 'Gate': [], 'Field': [],
                    'Rn': [], 'Ic': [], 'Pnorm': []}

for _, parameters in summary.iterrows():

    mid = parameters['Shapiro']
    path = paths[mid]

    frequency, gate, field = (parameters['Frequency'],
                              parameters['Gate voltage V'],
                              parameters['Parallel field mT'])

    ic, rn, att = (parameters['Critical current'],
                   parameters['Normal resistance'],
                   parameters['Attenuation'])
    print(f'\nTreating data for dataset {mid}\n'
          f'Frequency {frequency} GHz\n'
          f'Gate voltage {gate} V\n'
          f'Magnetic field {field} mt')

    with LabberData(path) as data:

        filters = {}
        channels = data.list_channels()
        power_name = [p for p in POWER_NAME if p in channels][0]
        for names, val in zip((FREQUENCY_NAME, GATE_NAME, FIELD_NAME),
                              (frequency*1e9, gate, field)):
            for name in names:
                if name in channels:
                    filters[name] = val
        step_counts = {s_i: None for s_i in STEP_INDEXES}

        shape = data.compute_shape((power_name, CURRENT_NAME))

        power = data.get_data(power_name, filters) + att
        curr = data.get_data(CURRENT_NAME, filters)
        volt = data.get_data(VOLTAGE_NAME, filters)

        # Handle interruptions in the last scan.
        while len(power) < shape[0]*shape[1]:
            shape[1] -= 1

        length = shape[0]*shape[1]
        power = power[:length].reshape(shape)
        volt = volt[:length].reshape(shape)
        curr = curr[:length].reshape(shape)

        # Filter out rows that contain a Nan (skipped values)
        mask = np.isfinite(volt).all(axis=0)
        power = power.T[mask].T
        volt = volt.T[mask].T
        curr = curr.T[mask].T

        if CORRECT_VOLTAGE_OFFSET:
            avg_len = CORRECT_VOLTAGE_OFFSET
            low_power_ind = np.unravel_index(np.argmin(power), shape)[1]
            zero_curr_ind = np.unravel_index(np.argmin(np.abs(curr)), shape)[0]
            volt -= np.mean(volt[zero_curr_ind:zero_curr_ind + avg_len,
                                 low_power_ind])

        # Convert the current data if requested
        if CURRENT_CONVERSION is not None:
            curr *= CURRENT_CONVERSION

        # Bin the data
        power, voltage, histo = bin_power_shapiro_steps(power, curr, volt,
                                                        frequency*1e9,
                                                        STEP_FRACTION)

        # Find the normalizing power
        step_0 = extract_step_weight(voltage, histo, 0)
        indexes = np.where(np.less(step_0, step_0[0]*NORMALIZING_THRESHOLD))[0]
        if len(indexes):
            print(f'\tAt f={frequency} threshold power: {power[indexes][0]}')
            norm_p = power[np.min(indexes)]
        else:
            plt.plot(power, step_0)
            plt.show()
            msg = ('\tPower was always lower than threshold for '
                   f'f={frequency} GHz')
            warnings.warn(msg)
            norm_p = power[-1]

        if mid == 150:
            norm_p = -0.4

        if PLOT_NORM_POWER_CHECK:
            plt.figure()
            plt.imshow(volt.T,
                       extent=(curr[0, 0], curr[-1, 0], power[0], power[-1]),
                       origin='lower',
                       aspect='auto')
            cbar = plt.colorbar()
            plt.axhline(norm_p)
            plt.show()

        for n, v in zip(('Meas id', 'Frequency', 'Gate', 'Field', 'Rn', 'Ic',
                         'Pnorm'),
                        (mid, frequency, gate, field, rn, ic, norm_p)):
            analysis_summary[n].append(v)

        # Fill the results
        norm_power = normalize_db_power(power, norm_p)
        norm_bessel_arg = (2*1.6e-19*rn*ic/(6.626e-34*frequency*1e9) *
                           np.power(10, norm_power/20))
        for i in step_counts:
            step_counts[i] = extract_step_weight(voltage, histo, i)/ic/1e-6

    to_save = {}
    to_save['Log power'] = norm_power
    to_save['Scaled ac current'] = norm_bessel_arg
    for s in STEP_INDEXES:
        to_save[f'Step{s}'] = step_counts[s]
    table = pd.DataFrame(to_save)
    filename = f'{mid}_f={frequency}_g={gate}_b={field}.dat'
    with open(os.path.join(RESULT_PATH, filename), 'w') as f:
        f.write(f'# Step fraction {STEP_FRACTION}\n'
                f'# Normalizing threshold {NORMALIZING_THRESHOLD}\n'
                f'# Source file: {path}\n')
        table.to_csv(f, index=False)

table = pd.DataFrame(analysis_summary)
with open(os.path.join(RESULT_PATH, 'analysis_summary.csv'), 'w') as f:
        f.write(f'# Normalizing threshold {NORMALIZING_THRESHOLD}\n')
        table.to_csv(f, index=False)
