"""Calculate the inductance of a dc SQUID loop from measurements.

Data input are SQUID oscillations as a function of the gate voltage on one of the
junctions.

See arXiv:2207.06933 Supplementary Info Section 2.
"""
import argparse
import json
import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.constants import physical_constants
from scipy.signal import find_peaks

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.utils import load_config, plot2d

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
args = parser.parse_args()

_, config = load_config(args.config_path, args.config_section)

PHI0 = physical_constants["mag. flux quantum"][0]

plt.style.use("fullscreen13")

datafile = LabberData(get_data_dir() / config.get("DATAPATH"))
with datafile as f:
    ibias, vmeas = f.get_data(config.get("CH_DMM"), get_x=True)
    bfield = f.get_data(config.get("CH_FIELD_PERP"))
    gate = f.get_data(config.get("CH_GATE"))
    gate_axis = f.get_axis(config.get("CH_GATE"))
dvdi = np.gradient(vmeas, axis=-1) / np.gradient(ibias, axis=-1)
icrit_n, icrit_p = extract_switching_current(
    ibias,
    vmeas,
    side="both",
    offset=config.getfloat("V_OFFSET"),
    threshold=config.getfloat("V_THRESHOLD"),
)

# make gate sweep axis 0
for name in ["ibias", "vmeas", "bfield", "gate", "dvdi", "icrit_n", "icrit_p"]:
    vars()[name] = np.moveaxis(vars()[name], gate_axis, 0)

peaks_n, peaks_p, gate_idx = [], [], []
skip_gates = json.loads(config.get("SKIP_GATES"))
for g_idx, (x, y, z, g, ic_n, ic_p) in enumerate(
    zip(bfield, ibias, dvdi, gate, icrit_n, icrit_p)
):
    if len(set(g)) != 1:
        warnings.warn(f"Gate voltage is assumed constant, but {min(g)=}, {max(g)=}")
    g = g[0]
    if g in skip_gates:
        print(f"Skipping {g} V")
        continue
    _, ax = plot2d(
        x / 1e-6,
        y.T / 1e-6,
        z.T / 1e3,
        xlabel="out-of-plane field (μT)",
        ylabel="current bias (μA)",
        zlabel="differential resistance (kΩ)",
        title=f"{config.get('CH_GATE')} = {round(g, 3)} V",
        stamp=f"{config.get('COOLDOWN')} {config.get('SCAN')}",
        vmin=config.getfloat("VMIN_OHMS", None) / 1e3,
        vmax=config.getfloat("VMAX_OHMS", None) / 1e3,
    )
    peak_kwargs = {
        "prominence": config.getfloat("PEAK_PROMINENCE"),
        "distance": config.getfloat("PEAK_DISTANCE"),
    }
    for ic, p in zip((ic_n, ic_p), (peaks_n, peaks_p)):
        peaks, _ = find_peaks(np.abs(ic), **peak_kwargs)
        ax.plot(x / 1e-6, ic / 1e-6, "--k")
        ax.scatter(
            x[peaks] / 1e-6,
            ic[peaks] / 1e-6,
            c=plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(peaks)],
        )
        p.append(peaks)
    gate_idx.append(g_idx)
peaks_n, peaks_p = np.array(peaks_n), np.array(peaks_p)

# calculate the loop area
b_peaks_n = bfield[gate_idx, peaks_n.T].T
b_peaks_p = bfield[gate_idx, peaks_p.T].T
b_peaks = np.concatenate([b_peaks_n, b_peaks_p])
areas = PHI0 / np.diff(b_peaks)
area = np.mean(areas)
area_error = np.std(areas)

fig, ax = plt.subplots()
ax.set_xlabel("loop area (μm$^2$)")
ax.set_ylabel("counts")
ax.hist(areas.flatten() / 1e-12)
ax.axvline(area / 1e-12, color="black")
ax.set_title(
    f"loop area = {round(area / 1e-12)} $\pm$ {round(area_error / 1e-12)} μm$^2$"
)

# calculate the inductance
ic_peaks_n = np.abs(icrit_n[gate_idx, peaks_n.T].T)
ic_peaks_p = np.abs(icrit_p[gate_idx, peaks_p.T].T)
ic_peaks = (ic_peaks_n + ic_peaks_p) / 2
flux_diffs = (b_peaks_p - b_peaks_n) * area

fig, ax = plt.subplots()
ax.set_xlabel("$(I_{c+} + I_{c-}) / 2$ (μA)")
ax.set_ylabel("$\Delta\Phi (\Phi_0)$")

inductances = []
for n, (ic, f) in enumerate(zip(ic_peaks.T, flux_diffs.T)):
    (slope, intercept), cov = np.polyfit(ic, f, deg=1, cov=True)
    # both branches shift in opposite directions
    inductance = abs(slope) / 2
    error = np.sqrt(cov[0, 0]) / 2

    ax.scatter(ic / 1e-6, f / PHI0, label=f"peak {n+1}")
    ax.plot(
        ic / 1e-6,
        (intercept + slope * ic) / PHI0,
        label=f"{round(inductance / 1e-9, 3)} $\pm$ {round(error / 1e-9, 3)} nH",
    )
    inductances.append(inductance)
arm_inductance, arm_std = np.mean(inductances), np.std(inductances)
loop_inductance, loop_std = (
    np.array([arm_inductance, arm_std])
    / config.getfloat("ARM_LENGTH")
    * config.getfloat("LOOP_CIRCUMFERENCE")
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[0::2] + handles[1::2], labels[0::2] + labels[1::2])
ax.text(
    0.5,
    0.99,
    f"arm inductance $\\approx$ {np.round(arm_inductance / 1e-9, 3)} "
    f"$\pm$ {np.round(arm_std / 1e-9, 3)} nH",
    transform=ax.transAxes,
    ha="center",
    va="top",
)
result_text = (
    f"loop inductance $\\approx$ {np.round(loop_inductance / 1e-9, 3)} "
    f"$\pm$ {np.round(loop_std / 1e-9, 3)} nH"
)
ax.set_title(result_text)
print(result_text.replace("$\pm$", "+/-").replace("$\\approx$", "~"))


#plt.show()
