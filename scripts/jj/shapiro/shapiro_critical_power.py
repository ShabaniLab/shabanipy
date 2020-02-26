# -*- coding: utf-8 -*-
"""Plot the dependance of the critical power (closing of the gap)

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

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

summary = pd.read_csv(CSV_ANALYSIS_SUMMARY_PATH, comment='#')

# Plot the linear power as a function of Ic
plt.figure()
for i, gate in enumerate(reversed(sorted(set(summary['Gate'])))):
    mask = np.logical_and(summary['Frequency'] == 6, summary['Gate'] == gate)
    plt.scatter(summary[mask]['Ic'],
                np.power(10, summary[mask]['Pnorm']/20),
                c=summary[mask]['Field'],
                vmax=500,
                marker=(i+2, 1, 0),
                label=f'Gate = {gate}')
plt.legend()
cbar = plt.colorbar()
cbar.ax.set_ylabel('Parallel field (mT)')
plt.xlabel('Ic (µA)')
plt.ylabel('Critical power (a.u)')

# Plot the linear power as a function of the product IcRn
plt.figure()
for i, gate in enumerate(reversed(sorted(set(summary['Gate'])))):
    mask = np.logical_and(summary['Frequency'] == 6, summary['Gate'] == gate)
    plt.scatter(summary[mask]['Ic']*summary[mask]['Rn'],
                np.power(10, summary[mask]['Pnorm']/20),
                c=summary[mask]['Field'],
                vmax=500,
                marker=(i+2, 1, 0),
                label=f'Gate = {gate}')

mask = summary['Frequency'] == 6
x = summary[mask]['Ic']*summary[mask]['Rn']
y = np.power(10, summary[mask]['Pnorm']/20)
model = LinearModel()
pars = model.guess(y, x=x)
fit_result = model.fit(y, pars, x=x)
print(fit_result.fit_report())

plt.plot(x, fit_result.best_fit)
plt.legend()
cbar = plt.colorbar()
cbar.ax.set_ylabel('Parallel field (mT)')
plt.xlabel('Rn Ic product (µV)')
plt.ylabel('Critical power (a.u.)')
plt.show()