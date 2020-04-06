# -*- coding: utf-8 -*-
"""Plot the width of the Shapiro step as a function of power.

We explore here how the nodes of the oscillations evolve as a function of field
and gate voltage.

If a simulation file is specified we compare the results to the simulation.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: CSV file containing the frequencies, gate, fields etc associated to each
#: measurement.
#: The expected column names are:
#: Meas id	Frequency	Gate	Field	Rn	Ic	Pnorm
CSV_ANALYSIS_SUMMARY_PATH =\
    ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/Shapiro/2019-01/'
     'StepWidthAnalysis/analysis_summary.csv')

#: CSV file containing the frequencies, Ic and Rn associated with each
#: simulation.
#: The column names are expected to be:
#: Frequency,Critical current,Normal resistance,σ,ω_ac,Ic*R_n,
#: Critical I_ac,Step width file name
#: And the simulation results files are expected to live in the same folder.
CSV_SIMULATIONS_SUMMARY_PATH =\
    ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/Shapiro/2019-01/'
     'RCSJSimulations/simulation_summary.csv')

#: Indexes of the shapiro steps to consider.
STEP_INDEXES = [0, 1, 2, 3, 4]

#: Frequency to consider or None if the frequency should be varying
FREQUENCY = 6

#: Gate voltage to consider or None if the frequency should be varying
GATE = -2

#: By field to consider or None if the frequency should be varying
FIELD = 0

#: List of measurement ids that should be avoided.
BLACKLIST = [169, 150, 171, 174, 106, 152, 144, 146, 162]

#: Pattern along which the analysis files are named. Analysis files are
#: expected to live in the same folder as the analysis summary file.
FILE_PATTERN = '{mid:d}_f={frequency}_g={gate}_b={field}.dat'

#: Pattern along which the simulation files are named. Analysis files are
#: expected to live in the same folder as the simulation summary file.
SIM_FILE_PATTERN = '{id:d}.csv'

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit.models import LinearModel

from shabanipy.jj.shapiro import bessel_step_width

dirname = os.path.dirname(CSV_ANALYSIS_SUMMARY_PATH)
summary = pd.read_csv(CSV_ANALYSIS_SUMMARY_PATH, comment='#')

if CSV_SIMULATIONS_SUMMARY_PATH:
    sim_dirname = os.path.dirname(CSV_SIMULATIONS_SUMMARY_PATH)
    sim_summary = pd.read_csv(CSV_SIMULATIONS_SUMMARY_PATH, comment='#')

mask = 1
ramped = []
for name, val in zip(('Frequency', 'Gate', 'Field'), (FREQUENCY, GATE, FIELD)):
    if val is not None:
        mask = np.logical_and(mask, summary[name] == val)
    else:
        ramped.append(name)

summary = summary[mask]
label_template = ', '.join([(f'{name}' ' = {}') for name in ramped])

for i, (_, row) in enumerate(summary.iterrows()):
    if int(row['Meas id']) in BLACKLIST:
        continue
    print(row)
    filename = FILE_PATTERN.format(mid=int(row['Meas id']),
                                   frequency=row['Frequency'],
                                   gate=row['Gate'],
                                   field=int(row['Field']))
    data = pd.read_csv(os.path.join(dirname, filename), comment='#')

    # Current at which we close the gap. Fixed to 1 in the absence of
    # simulation data
    norm_current = 1

    # Simulation to data to plot
    sdata = None

    if CSV_SIMULATIONS_SUMMARY_PATH:
        srow = sim_summary.loc[(sim_summary['Frequency'] == row['Frequency']) &
                               (sim_summary['Normal resistance'] ==
                                row['Rn']) &
                               (sim_summary['Critical current'] == row['Ic'])]
        if not srow.empty:
            srow = srow.iloc[0]
            norm_current = srow['Critical I_ac']
            filename =\
                SIM_FILE_PATTERN.format(id=int(srow['Step width file name']))
            path = os.path.join(sim_dirname, filename)
            if os.path.isfile(path):
                sdata = pd.read_csv(path, comment='#')

    for index in STEP_INDEXES:
        plt.figure(index)
        exc = norm_current * np.power(10, (data['Log power'])/20)
        plt.plot(exc, savgol_filter(data[f'Step{index}'], 3, 1), '+-',
                 label=label_template.format(*[row[name] for name in ramped]))
        if sdata is not None:
            plt.plot(sdata['Iac'], sdata[f'Step{index}'], color=f'C{i}')

dirname = os.path.dirname(CSV_ANALYSIS_SUMMARY_PATH)

for index in STEP_INDEXES:
    plt.figure(index)
    plt.title(f'Step {index}: ' +
              ', '.join([f'{name} = {row[name]}'
                         for name in ('Frequency', 'Gate', 'Field')
                         if name not in ramped]))
    plt.legend()
    plt.xlabel('Normalized AC excitation')
    plt.ylabel('Shapiro step width (Ic)')
    filename = (f'Step-{index}_' +
                '_'.join([f'{name}={row[name]}'
                          for name in ('Frequency', 'Gate', 'Field')
                          if name not in ramped]) +
                '_vs_' +
                '_'.join(ramped) + '.pdf')
    plt.savefig(os.path.join(dirname, filename))

plt.show()
