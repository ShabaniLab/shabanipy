# -*- coding: utf-8 -*-
"""Extract the normal resistance, critical current and excess current from V-I

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Name of the config file (located in the configs folder next to this script)
#: to use. This will overwrite all the following constants. This file should be
#: a python file defining all the constants defined above # --- Execution
CONFIG_NAME = "JS308-4JJ-2HB-1.py"

#: Common folder in which all data are stored
BASE_FOLDER = r"/Users/mdartiailh/Labber/Data/2019/12"

#: Name of the sample and associated parameters as a dict.
#: The currently expected keys are:
#: - path
#: - Tc (in K)
#: - gap size (in nm)
SAMPLES = {
    "JJ100-1": {
        "path": "Data_1205/JS131A_BM001_JJ100-1_006.hdf5",
        "Tc": 1.44,
        "gap size": 100,
    },
    "JJ100-2": {
        "path": "Data_1205/JS131A_BM001_JJ100-2_011.hdf5",
        "Tc": 1.44,
        "gap size": 100,
    },
    "JJ300": {
        "path": "Data_1205/JS131A_BM001_JJ300_015.hdf5",
        "Tc": 1.44,
        "gap size": 300,
    },
    "JJ500": {
        "path": "Data_1205/JS131A_BM001_JJ500_023.hdf5",
        "Tc": 1.44,
        "gap size": 500,
    },
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
VOLTAGE_NAME = {
    "JJ100-1": 1,
    "JJ100-2": 3,
    "JJ300": 3,
    "JJ500": 3,
}

#: Name or index of the column containing the counter value for scans with
#: multiple traces (use None if absent). Only the first trace is used in the
#: analysis.
COUNTER_NAME = None

#: Name or index of the column containing the gate value for scans with
#: gate traces (use None if absent).
GATE_NAME = None

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

from shabanipy.jj.iv_analysis import analyse_vi_curve
from shabanipy.utils.labber_io import LabberData

plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42

if CONFIG_NAME:
    print(
        f"Using configuration {CONFIG_NAME}, all scripts constants will be"
        " overwritten."
    )
    path = os.path.join(os.path.dirname(__file__), "configs", CONFIG_NAME)
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
    f.write(
        "\t".join(
            [
                "Sample",
                "JJGap(nm)",
                "Tc(K)",
                "Δ(meV)",
                "Vg(V)",
                "Rn_cold(Ω)",
                "Rn_hot(Ω)",
                "Ic_cold(µA)",
                "Ic_hot(µA)",
                "I_exe_cold(µA)",
                "I_exe_hot(µA)",
                "RnIc_cold(meV)",
                "RnI_exe_cold(meV)",
            ]
        )
        + "\n"
    )

results = defaultdict(list)

for sample, parameters in SAMPLES.items():

    # Superconducting gap in meV
    gap = 1.674 * constants.Boltzmann / constants.e * 1000 * parameters["Tc"]

    with LabberData(os.path.join(BASE_FOLDER, parameters["path"])) as data:

        print(data.channel_names)
        filters = {}
        counter = get_value(sample, COUNTER_NAME)
        if counter is not None:
            val = data.get_data(counter)[0]
            filters[counter] = val

        gate_col = get_value(sample, GATE_NAME)
        if gate_col:
            gates = np.unique(data.get_data(gate_col))[::-1]
        else:
            gates = [None]

        offset_corr = get_value(sample, CORRECT_VOLTAGE_OFFSET)

        for gate in gates:

            if gate is not None:
                filters[gate_col] = gate

            current_bias = data.get_data(
                get_value(sample, BIAS_NAME), filters=filters
            ) * get_value(sample, CURRENT_CONVERSION)

            measured_voltage = data.get_data(
                get_value(sample, VOLTAGE_NAME), filters=filters
            )

            # Convert the voltage to the physical value
            measured_voltage /= get_value(sample, AMPLIFIER_GAIN)

            title = f"Sample {sample}" + (
                f", Vg = {gate} V" if gate else f", gap {parameters['gap size']} nm"
            )
            # Store the results to be able to plot a summary at the end
            offset_corr, rn_c, rn_h, ic_c, ic_h, iexe_c, iexe_h = analyse_vi_curve(
                current_bias,
                measured_voltage,
                offset_corr,
                get_value(sample, IC_VOLTAGE_THRESHOLD)
                / get_value(sample, AMPLIFIER_GAIN),
                get_value(sample, HIGH_BIAS_THRESHOLD),
                plot_title=title,
            )
            # Convert to µA
            ic_c *= 1e6
            ic_h *= 1e6
            iexe_c *= 1e6
            iexe_h *= 1e6
            for n, v in zip(
                [
                    "sample",
                    "gap_size",
                    "Tc",
                    "gap",
                    "gate",
                    "rn_cold",
                    "rn_hot",
                    "ic_cold",
                    "ic_hot",
                    "iexe_cold",
                    "iexe_hot",
                ],
                [
                    sample,
                    parameters["gap size"],
                    parameters["Tc"],
                    gap,
                    gate,
                    rn_c,
                    rn_h,
                    ic_c,
                    ic_h,
                    iexe_c,
                    iexe_h,
                ],
            ):
                results[n].append(v)

            # Save the summary of the result
            with open(OUTPUT, "a") as f:
                to_save = (
                    sample,
                    parameters["gap size"],
                    parameters["Tc"],
                    gap,
                    gate,
                    rn_c,
                    rn_h,
                    ic_c,
                    ic_h,
                    iexe_c,
                    iexe_h,
                    rn_c * ic_c / 1e3,
                    rn_c * iexe_c / 1e3,
                )
                f.write("\t".join([f"{v}" for v in to_save]) + "\n")

# Prepare comparative plots between samples
for k in results:
    results[k] = np.array(results[k])

if None in results["gate"]:
    fig = plt.figure()
    plt.suptitle("Size dependence")
    ax = fig.gca()
    ax.plot(
        results["gap_size"],
        results["ic_cold"] * results["rn_cold"] / 1e3 / results["gap"],
        "+",
        label="$R_N\,I_c/\Delta$",
    )
    ax.plot(
        results["gap_size"],
        results["iexe_cold"] * results["rn_cold"] / 1e3 / results["gap"],
        "+",
        label="$R_N\,I_{exe}/\Delta$",
    )
    ax.axhline(np.pi, ls="--", label="IcRn ballistic limit")
    ax.axhline(1.32 * np.pi / 2, ls="-.", label="IcRn diffusive limit")
    ax.axhline(8 / 3, color="C1", ls="--", label="IexeRn ballistic limit")
    ax.axhline(1.467, color="C1", ls="-.", label="IexeRn diffusive limit")
    ax.set_xlabel("Gap size (nm)")
    ax.set_ylabel("")
    ax.legend()
else:
    fig = plt.figure()
    plt.suptitle("Gate dependence")
    ax = fig.gca()
    ax.plot(
        results["gate"],
        results["ic_cold"] * results["rn_cold"] / 1e3 / results["gap"],
        "+",
        label="$R_N\,I_c/\Delta$",
    )
    ax.plot(
        results["gate"],
        results["iexe_cold"] * results["rn_cold"] / 1e3 / results["gap"],
        "+",
        label="$R_N\,I_{exe}/\Delta$",
    )
    ax.axhline(np.pi)
    ax.set_xlabel("Gate voltage (V)")
    ax.set_ylabel("")
    ax.legend()

# Display all figures
plt.show()
