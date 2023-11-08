"""Determine critical temperature by thresholding.

This script discards repeated temperature measurements due to Oxford software's
infrequent temperature updates from the Lakeshore resistance bridge.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.dvdi import find_rising_edge
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import get_output_dir, load_config, stamp, write_metadata

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("config_path", help="path to .ini config file")
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument("--minT", type=float, default=None, help="minimum temperature")
parser.add_argument("--maxT", type=float, default=None, help="maximum temperature")
parser.add_argument(
    "--makereal",
    choices=["real", "abs"],
    default="real",
    help="use the real part or magnitude (abs) of complex lock-in data",
)
args = parser.parse_args()
_, config = load_config(args.config_path, args.config_section)

outdir = get_output_dir() / "tcbc"
outdir.mkdir(exist_ok=True, parents=True)
print(f"Output directory: {outdir}")
outprefix = (
    outdir
    / f"{Path(args.config_path).stem}_{args.config_section}_Tc_{args.minT}-{args.maxT}"
)

makereal = {"real": np.real, "abs": np.abs}
with ShaBlabberFile(config["DATAPATH"]) as f:
    temp, volt = f.get_data(config["CH_TEMP"], config["CH_VOLT"])
    volt = makereal[args.makereal](volt)

if temp.ndim > 1:
    # assume temperature is swept along last axis (i.e. Labber outer-most loop)
    # assume all other dimensions are trivial counters/timers
    temp = temp.flatten(order="F")
    volt = volt.flatten(order="F")

# remove stale data points due to Oxford control software pulling temperature data
# infrequently from Lakeshore resistance bridge
temp_fresh = [temp[0]]
indexs = [0]
for i, t in enumerate(temp):
    if t == temp_fresh[-1]:
        continue
    temp_fresh.append(t)
    indexs.append(i)
temp_fresh = np.array(temp_fresh)
volt_fresh = volt[indexs]

plt.style.use("fullscreen13")
fig, ax = plt.subplots()
ax.plot(temp, ".", label="stale")
ax.plot(indexs, temp_fresh, ".", label="fresh")
ax.set_ylabel(config["CH_TEMP"])
ax.set_title(args.config_section)
ax.legend()

sort_idx = np.argsort(temp_fresh)
temp_fresh = np.take_along_axis(temp_fresh, sort_idx, axis=0)
volt_fresh = np.take_along_axis(volt_fresh, sort_idx, axis=0)

if args.minT is not None:
    mask = temp_fresh >= args.minT
    temp_fresh = temp_fresh[mask]
    volt_fresh = volt_fresh[mask]
if args.maxT is not None:
    mask = temp_fresh <= args.maxT
    temp_fresh = temp_fresh[mask]
    volt_fresh = volt_fresh[mask]

fig, ax = plt.subplots()
ax.plot(temp, volt, ".", label="stale")
ax.plot(temp_fresh, volt_fresh, label="fresh")
ax.set_xlabel(config["CH_TEMP"])
ax.set_ylabel(config["CH_VOLT"] + f" ({args.makereal})")
ax.set_title(args.config_section)
ax.legend()

tc = find_rising_edge(temp_fresh, volt_fresh, interp=True)

fig, ax = plt.subplots()
ax.plot(temp_fresh, volt_fresh / config.getfloat("IBIAS", 1), marker=".")
ax.set_xlabel("MXC temperature (K)")
ax.set_ylabel(
    ("resistance (Ω)" if config.get("IBIAS") else "voltage (V)") + f" ({args.makereal})"
)
ax.set_title(args.config_section)
ax.axvline(tc, color="k", ls=":")
ax.text(
    tc, volt_fresh.min() / config.getfloat("IBIAS", 1), f"$T_c \\approx {round(tc, 2)}$"
)
stamp(ax, config["DATAPATH"])
fig.savefig(str(outprefix) + ".png")

write_metadata(str(outprefix) + "_metadata.txt", args=args)

plt.show()
