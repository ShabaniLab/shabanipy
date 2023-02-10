"""Plot weak antilocalization traces for multiple gate voltages."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import jy_pink, load_config, plot

print = partial(print, flush=True)

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
jy_pink.register()
cmap = plt.get_cmap("jy_pink").reversed()
OUTDIR = Path("./output/")
OUTDIR.mkdir(exist_ok=True)
print(f"Output directory: {OUTDIR}")
CHIP_ID = f"{config['WAFER']}-{config['PIECE']}"

filters, n = [], 1
while config.get(f"FILTER{n}_CH"):
    filters.append(
        (
            config[f"FILTER{n}_CH"],
            getattr(np, config[f"FILTER{n}_OP"]),
            config.getfloat(f"FILTER{n}_VAL"),
        )
    )
    n += 1

CH_GATE = config.get("CH_GATE", "gate - Source voltage")


def get_hall_data(datapath, ch_lockin_meas):
    with ShaBlabberFile(datapath) as f:
        gate, bfield, dvdi = f.get_data(
            CH_GATE,
            config["CH_FIELD_PERP"],
            ch_lockin_meas,
            order=(CH_GATE, config["CH_FIELD_PERP"]),
            filters=filters,
        )
        dvdi /= config.getfloat("IBIAS_AC")
        dvdi *= config.getfloat("GEOMETRIC_FACTOR")
    return gate, bfield, dvdi.real


gate_xx, bfield_xx, rxx = get_hall_data(
    config["DATAPATH_RXX"], config.get("CH_LOCKIN_XX", "Rxx - Value")
)
gate_yy, bfield_yy, ryy = get_hall_data(
    config["DATAPATH_RYY"], config.get("CH_LOCKIN_YY", "Ryy - Value")
)


def plot(bfield, gate, res, xx):
    fig, ax = plt.subplots()
    ax.set_xlabel("out-of-plane field (mT)")
    ax.set_ylabel(f"$\Delta\\rho_{{{xx}}}$ (Ω)")
    for i, (b, r) in enumerate(zip(bfield, res)):
        ax.plot(b / 1e-3, r - r[0], color=cmap(i / len(gate)))
    lines = ax.get_lines()
    lines[0].set_label(f"{np.round(np.unique(gate[0]), 2).squeeze()} V")
    lines[-1].set_label(f"{np.round(np.unique(gate[-1]), 2).squeeze()} V")
    ax.legend()
    fig.savefig(OUTDIR / f"{CHIP_ID}_wal{xx}.png")


plot(bfield_xx, gate_xx, rxx, "xx")
plot(bfield_yy, gate_yy, ryy, "yy")

plt.show()
