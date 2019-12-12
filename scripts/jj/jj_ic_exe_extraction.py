# -*- coding: utf-8 -*-
"""Extract the normal resistance, critical current and excess current from V-I

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = 'JS129.py'

#: Common folder in which all data are stored
BASE_FOLDER = r'/Users/mdartiailh/Labber/Data/2019/12'

#: Name of the sample and associated parameters as a dict.
#: The currently expected keys are:
#: - path
#: - Tc (in K)
#: - gap size (in nm)
SAMPLES = {"JJ100-1": {"path": "Data_1205/JS131A_BM001_JJ100-1_006.hdf5",
                       "Tc": 1.44, "gap size": 100},
           "JJ100-2": {"path": "Data_1205/JS131A_BM001_JJ100-2_011.hdf5",
                       "Tc": 1.44, "gap size": 100},
           "JJ300": {"path": "Data_1205/JS131A_BM001_JJ300_015.hdf5",
                     "Tc": 1.44, "gap size": 300},
           "JJ500": {"path": "Data_1205/JS131A_BM001_JJ500_023.hdf5",
                     "Tc": 1.44, "gap size": 500},
            }

#: Path to the file in which to write the output
OUTPUT = "/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/JJ/JS131/results.csv"

#: For all the following parameters one can use a dictionary with sample names
#: as keys can be used.

#: Name or index of the column containing the voltage bias data.
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
BIAS_NAME = 0

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = {"JJ100-1": 1,
                "JJ100-2": 3,
                "JJ300": 3,
                "JJ500": 3,
                }

#: Name or index of the column containing the counter value for scans with
#: multiple traces (use None if absent). Only the first trace is used in the
#: analysis.
COUNTER_NAME = None

#: Should we correct the offset in voltage and if so on how many points to
#: average
CORRECT_VOLTAGE_OFFSET = 5

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Amplifier gain used to measure the voltage across the junction.
AMPLIFIER_GAIN = 100

#: Threshold to use to determine the critical current (in raw data units).
IC_VOLTAGE_THRESHOLD = 5e-4

#: Bias current at which we consider to be in the high bias regime and can fit
#: the resistance.
HIGH_BIAS_THRESHOLD = 29e-6

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import warnings
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from lmfit.models import LinearModel

from shabanipy.utils.labber_io import LabberData

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 13
plt.rcParams['pdf.fonttype'] = 42

if CONFIG_NAME:
    print(f"Using configuration {CONFIG_NAME}, all scripts constants will be"
          " overwritten.")
    path = os.path.join(os.path.dirname(__file__), 'configs', CONFIG_NAME)
    with open(path) as f:
        exec(f.read())

get_value = lambda s, p: p[s] if isinstance(p, dict) else p

with open(OUTPUT, "w") as f:
    f.write("# Parameters\n")
    f.write(f"# BIAS_NAME = {BIAS_NAME}\n")
    f.write(f"# VOLTAGE_NAME = {VOLTAGE_NAME}\n")
    f.write(f"# CORRECT_VOLTAGE_OFFSET = {CORRECT_VOLTAGE_OFFSET}\n")
    f.write(f"# CURRENT_CONVERSION = {CURRENT_CONVERSION}\n")
    f.write(f"# AMPLIFIER_GAIN = {AMPLIFIER_GAIN}\n")
    f.write(f"# IC_VOLTAGE_THRESHOLD = {IC_VOLTAGE_THRESHOLD}\n")
    f.write(f"# HIGH_BIAS_THRESHOLD = {HIGH_BIAS_THRESHOLD}\n")
    f.write("\t".join(["Sample", "JJGap(nm)", "Tc(K)", "Δ(meV)", "Rn_cold(Ω)",
                       "Rn_hot(Ω)", "Ic_cold(µA)", "Ic_hot(µA)", "I_exe_cold(µA)",
                       "I_exe_hot(µA)", "RnIc_cold(meV)", "RnI_exe_cold(meV)"]) + "\n")

results = defaultdict(list)

for sample, parameters in SAMPLES.items():

    # Superconducting gap in meV
    gap = 1.674*constants.Boltzmann/constants.e*1000*parameters["Tc"]

    with LabberData(os.path.join(BASE_FOLDER, parameters["path"])) as data:

        filters = {}
        counter = get_value(sample, COUNTER_NAME)
        if counter is not None:
            val = data.get_data(counter)[0]
            filters[counter] = val

        current_bias = (data.get_data(get_value(sample, BIAS_NAME), filters=filters) *
                        get_value(sample, CURRENT_CONVERSION))

        if current_bias[0] < 0.0:
            cold_value = lambda p, n: abs(p)
            hot_value = lambda p, n: abs(n)
        else:
            cold_value = lambda p, n: abs(n)
            hot_value = lambda p, n: abs(p)

        measured_voltage = data.get_data(get_value(sample, VOLTAGE_NAME),
                                         filters=filters)

        # Sort the data so that the bias always go from negative to positive
        sorting_index = np.argsort(current_bias)
        current_bias = current_bias[sorting_index]
        measured_voltage = measured_voltage[sorting_index]

        # Index at which the bias current is zero
        index = np.argmin(np.abs(current_bias))

        # Correct the offset in the voltage data
        offset_avg = get_value(sample, CORRECT_VOLTAGE_OFFSET)
        if offset_avg:
            measured_voltage -= np.average(measured_voltage[index-offset_avg+1:
                                                            index+offset_avg])

        # Extract the critical current on the positive and negative branch
        # Express them in µA
        ic_n = current_bias[np.max(
                                np.where(
                                    np.less(measured_voltage[:index],
                                            -get_value(sample, IC_VOLTAGE_THRESHOLD)
                                            )
                                    )[0]
                                )
                            ]   * 1e6
        ic_p = current_bias[np.min(
                                np.where(
                                    np.greater(measured_voltage[index:],
                                               get_value(sample, IC_VOLTAGE_THRESHOLD)
                                               )
                                    )[0]
                                ) + index
                            ] * 1e6

        # Convert the voltage to the physical value
        measured_voltage /= get_value(sample, AMPLIFIER_GAIN)

        # Fit the high positive/negative bias to extract the normal resistance
        # excess current and their product
        index_pos = np.argmin(np.abs(current_bias - HIGH_BIAS_THRESHOLD))
        index_neg = np.argmin(np.abs(current_bias + HIGH_BIAS_THRESHOLD))

        model = LinearModel()
        pars = model.guess(measured_voltage[index_pos:],
                           x=current_bias[index_pos:])
        pos_results = model.fit(measured_voltage[index_pos:], pars,
                                x=current_bias[index_pos:])

        pars = model.guess(measured_voltage[index_neg:],
                           x=current_bias[index_neg:])
        neg_results = model.fit(measured_voltage[:index_neg], pars,
                                x=current_bias[:index_neg])

        rn_p = pos_results.best_values["slope"]
        # In µA
        iexe_p = -pos_results.best_values["intercept"]/rn_p * 1e6

        rn_n = neg_results.best_values["slope"]
        # In µA
        iexe_n = -neg_results.best_values["intercept"]/rn_n * 1e6


        # Store the results to be able to plot a summary at the end
        rn_c = cold_value(rn_p, rn_n)
        rn_h = hot_value(rn_p, rn_n)
        ic_c = cold_value(ic_p, ic_n)
        ic_h = hot_value(ic_p, ic_n)
        iexe_c = cold_value(iexe_p, iexe_n)
        iexe_h = hot_value(iexe_p, iexe_n)
        for n, v in zip(["sample", "gap_size", "Tc", "gap", "rn_cold", "rn_hot",
                         "ic_cold", "ic_hot", "iexe_cold", "iexe_hot"],
                        [sample, parameters["gap size"], parameters["Tc"], gap,
                        rn_c, rn_h, ic_c, ic_h, iexe_c, iexe_h]):
            results[n].append(v)

        # Save the summary of the result
        with open(OUTPUT, "a") as f:
            to_save = (sample, parameters["gap size"], parameters["Tc"],
                       gap, rn_c, rn_h, ic_c, ic_h, iexe_c, iexe_h,
                       rn_p*ic_c/1e3, rn_p*iexe_c/1e3)
            f.write("\t".join([f"v" for v in to_save]) + "\n")

        # Prepare a summary plot: full scale
        fig = plt.figure()
        fig.suptitle(f"Sample {sample}")
        ax = fig.gca()
        ax.plot(current_bias*1e6, measured_voltage*1e3)
        ax.plot(current_bias[index:]*1e6,
                model.eval(pos_results.params, x=current_bias[index:])*1e3,
                "--k")
        ax.plot(current_bias[:index+1]*1e6,
                model.eval(neg_results.params, x=current_bias[:index+1])*1e3,
                "--k")
        ax.set_xlabel("Bias current (µA)")
        ax.set_ylabel("Voltage drop (mV)")

        # Prepare a summary plot: zoomed in
        fig = plt.figure()
        fig.suptitle(f"Sample {sample}: zoom")
        ax = fig.gca()
        mask = np.logical_and(np.greater(current_bias*1e6, -3*ic_p),
                              np.less(current_bias*1e6, 3*ic_p))
        ax.plot(current_bias*1e6, measured_voltage*1e3)
        ax.plot(current_bias[index:]*1e6,
                model.eval(pos_results.params, x=current_bias[index:])*1e3,
                "--")
        ax.plot(current_bias[:index+1]*1e6,
                model.eval(neg_results.params, x=current_bias[:index+1])*1e3,
                "--")
        ax.set_xlim((-3*ic_c, 3*ic_c))
        ax.set_ylim((np.min(measured_voltage[mask]*1e3),
                     np.max(measured_voltage[mask]*1e3)))
        ax.set_xlabel("Bias current (µA)")
        ax.set_ylabel("Voltage drop (mV)")

# Prepare comparative plots between samples
for k in results:
    results[k] = np.array(results[k])

fig = plt.figure()
plt.suptitle("Size dependence")
ax = fig.gca()
ax.plot(results["gap_size"], results["ic_cold"]*results["rn_cold"]/1e3/results["gap"],
        "+", label="$R_N\,I_c/\Delta$")
ax.plot(results["gap_size"], results["iexe_cold"]*results["rn_cold"]/1e3/results["gap"],
        "+", label="$R_N\,I_{exe}/\Delta$")
ax.axhline(np.pi)
ax.set_xlabel("Gap size (nm)")
ax.set_ylabel("")
ax.legend()

# Display all figures
plt.show()
