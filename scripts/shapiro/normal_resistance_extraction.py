# -*- coding: utf-8 -*-
"""Extract the normal resistance of a junction from an V-I curve.

The extraction is done using a linear fit of the data at large bias.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = '/Users/mdartiailh/Labber/Data/2019/02/Data_0207/JS124S_BM002_176.hdf5'

#: Path of the file in which to store the results.
RESULT_PATH = ('/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/'
               'ShapiroInPlaneField/high_t_normal_resistance.dat')

#: Name or index of the column containing the current data
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
CURRENT_NAME = 0

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 4

#: Names or indexes of the columns which were ramped in addition to the bias
#: and name under which they should be saved.
#: Those are assuemd to be swept independtly from one another.
#: ex: gate voltage, in plane field, ...
SWEPT_PARAMETERS = {1: 'Gate voltage', 2: 'By field'}

#: Current range (low, high) in which to perform the fit to determine the
#: normal state resistance.
FIT_RANGE = [10e-6, 15e-6]

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Should all the line plots be presented
PLOT_FITS = False


# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

from shabanipy.shapiro import normalize_db_power
from shabanipy.shapiro.binning import (bin_power_shapiro_steps,
                                       extract_step_weight)
from shabanipy.utils.labber_io import LabberData

results = {}

with LabberData(PATH) as data:

    print('Available channels:', data.list_channels())
    swept_quantities = [sorted(set(data.get_data(swept)))
                        for swept in SWEPT_PARAMETERS]
    points = np.prod([len(s) for s in swept_quantities])
    for s in SWEPT_PARAMETERS.values():
        results[s] = np.empty(points)
    results['Rn'] = np.empty(points)
    results['dRn'] = np.empty(points)
    results['Iexe'] = np.empty(points)
    results['dIexe'] = np.empty(points)

    for i, values in enumerate(product(*swept_quantities)):

        for j, s in enumerate(SWEPT_PARAMETERS.values()):
            results[s][i] = values[j]

        filters = {s: v for s, v in zip(SWEPT_PARAMETERS, values)}

        volt = data.get_data(VOLTAGE_NAME, filters=filters)
        curr = data.get_data(CURRENT_NAME, filters=filters)

        # Convert the current data if requested
        if CURRENT_CONVERSION is not None:
            curr *= CURRENT_CONVERSION

        # Perform the fit to compute the resistance
        mask = np.where(np.logical_and(np.greater(curr, FIT_RANGE[0]),
                                       np.less(curr, FIT_RANGE[1])))
        c, v = curr[mask], volt[mask]
        if not len(c):
            msg = ('No value in the fitting range data cover the '
                   f'[{np.min(curr)}, {np.max(curr)}] range.')
            raise ValueError(msg)
        model = LinearModel()
        pars = model.guess(v, x=c)
        fit_result = model.fit(v, pars, x=c)

        if PLOT_FITS:
            plt.figure()
            plt.plot(c, v, 'bx')
            plt.plot(c, fit_result.best_fit, 'r-')

        # Fill the results
        results['Rn'][i] = fit_result.best_values['slope']
        results['dRn'][i] = fit_result.params['slope'].stderr
        results['Iexe'][i] = fit_result.best_values['intercept']
        results['dIexe'][i] = fit_result.params['intercept'].stderr

table = pd.DataFrame(results)
with open(RESULT_PATH, 'w') as f:
    f.write(f'# Source file: {PATH}\n')
    table.to_csv(f, index=False)

if len(SWEPT_PARAMETERS) == 1:
    plt.figure()
    plt.errorbar(results[list(SWEPT_PARAMETERS.values())[0]], results['Rn'],
                 yerr=results['dRn'], label="Normal resistance")
    plt.legend()
else:
    # plots only supported for 1D scans
    pass

plt.show()
