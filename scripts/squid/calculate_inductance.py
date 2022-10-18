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

peaks_n, peaks_p = [], []
skip_gates = json.loads(config.get("SKIP_GATES"))
for idx, (x, y, z, g, ic_n, ic_p) in enumerate(
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

plt.show()
