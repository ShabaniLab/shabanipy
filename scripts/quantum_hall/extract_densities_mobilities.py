# -*- coding: utf-8 -*-
"""Extract the density and mobility from a quantum hall measurement.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r"/Users/mdartiailh/Labber/Data/2018/08/Data_0827/JS129A_129VP_JY001_002.hdf5"

#: Index or name of the column containing the gate voltage values.
GATE_COLUMN = None

#: Index or name of the column containing the applied magnetic field.
FIELD_COLUMN = 0

# WARNING Labber uses 2 columns per lock-in value so you should be careful when
# using indexes. The index is always assumed to refer to the first column of
# the lock-in (ie real values)

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along x.
XX_VOLTAGE_COLUMN = 1

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along y.
YY_VOLTAGE_COLUMN = 3

#: Index or name of the column contaning the transverse voltage drop
#: measurement.
XY_VOLTAGE_COLUMN = 5

#: Component of the measured voltage to use for analysis.
#: Recognized values are 'real', 'imag', 'magnitude'
LOCK_IN_QUANTITY = "real"

#: Value of the excitation current used by the lock-in amplifier in A.
PROBE_CURRENT = 50e-9

#: Sample geometry used to compute the mobility.
#: Accepted values are 'Van der Pauw', 'Standard Hall bar'
GEOMETRY = "Standard Hall bar"

#: Magnetic field bounds to use when extracting the density.
FIELD_BOUNDS = (500e-3, 2)

#: Should we plot the fit used to extract the density at each gate.
PLOT_DENSITY_FIT = False

#: Effective mass of the carriers in unit of the electron mass.
EFFECTIVE_MASS = 0.03

#: File in which to store the results of the analysis as a function of gate
#: voltage.
RESULT_PATH = (
    "/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/Shapiro/2019-11-JS129"
    "JS129A_129VP_JY001_002_density_mobility.csv"
)

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as cs

from shabanipy.quantum_hall.conversion import (
    convert_lock_in_meas_to_diff_res,
    GEOMETRIC_FACTORS,
    fermi_velocity_from_density,
    mean_free_time_from_mobility,
    diffusion_constant_from_mobility_density,
)
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility
from shabanipy.utils.labber_io import LabberData


with LabberData(PATH) as data:

    names = data.list_channels()
    if GATE_COLUMN is not None:
        shape = data.compute_shape((GATE_COLUMN, FIELD_COLUMN))

        gate = data.get_data(GATE_COLUMN).reshape(shape).T
        field = data.get_data(FIELD_COLUMN).reshape(shape).T
    else:
        field = data.get_data(FIELD_COLUMN).T
        gate = np.zeros(1)
    res = dict.fromkeys(("xx", "yy", "xy"))
    for k in res:
        name = globals()[f"{k.upper()}_VOLTAGE_COLUMN"]
        index = data.name_or_index_to_index(name)
        if LOCK_IN_QUANTITY == "real":
            val = data.get_data(index)
        elif LOCK_IN_QUANTITY == "imag":
            val = data.get_data(index + 1)
        else:
            val = data.get_data(index) ** 2 + data.get_data(index + 1) ** 2

        if GATE_COLUMN is not None:
            val = val.reshape(shape)
        val = val.T

        res[k] = convert_lock_in_meas_to_diff_res(val, PROBE_CURRENT)

if GATE_COLUMN is not None:
    gate = gate[:, 0]
density, std_density = extract_density(field, res["xy"], FIELD_BOUNDS, PLOT_DENSITY_FIT)
if PLOT_DENSITY_FIT:
    plt.show()

mobility = extract_mobility(
    field, res["xx"], res["yy"], density, GEOMETRIC_FACTORS[GEOMETRY]
)

mass = EFFECTIVE_MASS * cs.electron_mass
vf = fermi_velocity_from_density(density, mass)
mft = mean_free_time_from_mobility(mobility, mass)
diff = diffusion_constant_from_mobility_density(mobility, density, mass)

if RESULT_PATH:
    df = pd.DataFrame(
        {
            "Gate voltage (V)": gate,
            "Density (m^-2)": density,
            "Stderr density (m^-2)": std_density,
            "Mobility xx (m^2V^-1s^-1)": mobility[0],
            "Mobility yy (m^2V^-1s^-1)": mobility[1],
            "Mean free time xx (s)": mft[0],
            "Mean free time yy (s)": mft[1],
            "Diffusion xx (m^2/s)": diff[0],
            "Diffusion yy (m^2/s)": diff[1],
        }
    )
    with open(RESULT_PATH, "w") as f:
        f.write(
            f"# Probe-current: {PROBE_CURRENT}\n"
            f"# Effective mass: {EFFECTIVE_MASS}\n"
            f"# Geometry: {GEOMETRY}\n"
            f"# Lock-in quantity: {LOCK_IN_QUANTITY}\n"
        )
        df.to_csv(f, index=False)

fig, axes = plt.subplots(1, 2)
axes[0].errorbar(gate, density / 1e4, std_density / 1e4, fmt="*")
axes[0].set_xlabel("Gate voltage (V)")
axes[0].set_ylabel("Density (cm$^2$)")
axes[1].plot(density / 1e4, mobility[0] * 1e4, "+", label="xx")
axes[1].plot(density / 1e4, mobility[1] * 1e4, "x", label="yy")
axes[1].set_xlabel("Density (cm$^2$)")
axes[1].set_ylabel("Mobility (cm$^2$V$^-1$s$^-1$)")
plt.legend()
plt.tight_layout()
plt.show()
