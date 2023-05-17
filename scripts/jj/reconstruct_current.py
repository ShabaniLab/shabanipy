"""Reconstruct critical current distribution from Fraunhofer interference pattern."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants

from shabanipy.dvdi import extract_switching_current
from shabanipy.jj.fraunhofer import recenter_fraunhofer, symmetrize_fraunhofer
from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
)
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import load_config
from shabanipy.utils.plotting import jy_pink, plot, plot2d

print = partial(print, flush=True)

# command-line interface and config file
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--width", "-w", type=float, help="junction width (m)",
)
parser.add_argument(
    "--length", "-l", type=float, help="junction length (m)",
)
parser.add_argument(
    "--center",
    "-c",
    default=True,
    action="store_false",
    help="center fraunhofer at maximum",
)
parser.add_argument(
    "--symmetrize",
    "-s",
    default=False,
    action="store_true",
    help="symmetrize fraunhofer about 0",
)
args = parser.parse_args()
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)
WIDTH = args.width if args.width else config.getfloat("JJ_WIDTH")
LENGTH = args.length if args.length else config.getfloat("JJ_LENGTH")
if WIDTH is None:
    raise ValueError(
        "Junction width must be provided on command line (--width) or in config (JJ_WIDTH)."
    )
if LENGTH is None:
    raise ValueError(
        "Junction length must be provided on command line (--length) or in config (JJ_LENGTH)."
    )

# output
OUTDIR = "./output/"
print(f"Output directory: `{OUTDIR}`")
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
SLICE_STR = (
    f"_idx={config.getfloat('3RD_AXIS_INDEX')}" if "3RD_AXIS_INDEX" in config else ""
)
OUTPATH = Path(OUTDIR) / f"{config['FILENAME']}{SLICE_STR}_current-reconstruction"
jy_pink.register()
plt.style.use(["jy_pink", "fullscreen13"])

if config.getint("3RD_AXIS_INDEX") is not None:
    SLICES = (..., config.getint("3RD_AXIS_INDEX"))
else:
    SLICES = None

# load the data
with ShaBlabberFile(config["DATAPATH"]) as f:
    bfield, ibias, dvdi = f.get_data(
        config["CH_FIELD_PERP"],
        config["CH_BIAS"],
        config["CH_MEASURE"],
        order=(config["CH_FIELD_PERP"], config["CH_BIAS"]),
        slices=SLICES,
    )
    if f.get_channel(config["CH_BIAS"]).unitPhys == "V":
        ibias /= config.getfloat("R_DC_OUT")
    ibias_ac = f.get_channel(config["CH_MEASURE"]).instrument.config[
        "Output amplitude"
    ] / config.getfloat("R_AC_OUT")
    dvdi = np.abs(dvdi) / ibias_ac

if dvdi.ndim > 2:
    raise ValueError(
        f"This script is not yet equipped to handle {dvdi.ndim}-dimensional data."
        "Try providing a 3RD_AXIS_INDEX in your config to slice the data."
    )

# plot the raw data
fig, ax = plot2d(
    bfield / 1e-3,
    ibias / 1e-6,
    dvdi,
    xlabel="x coil field (mT)",
    ylabel="dc bias (μA)",
    zlabel="dV/dI (Ω)",
    title="raw data",
    stamp=config["COOLDOWN"] + "_" + config["SCAN"],
)
fig.savefig(str(OUTPATH) + "_raw-data.png")

# extract the switching current
bfield = np.unique(bfield)  # assumes all field sweeps are identical
ic = extract_switching_current(
    ibias,
    dvdi,
    threshold=config.getfloat("RESISTANCE_THRESHOLD", fallback=None),
    interp=True,
)
ax.set_title("switching current")
plot(bfield / 1e-3, ic / 1e-6, ax=ax, color="k", lw=1)
fig.savefig(str(OUTPATH) + "_ic-extraction.png")

if args.center:
    bfield = recenter_fraunhofer(bfield, ic)
    fig, ax = plt.subplots()
    ax.axvline(0, color="k")
    plot(
        bfield / 1e-3,
        ic / 1e-6,
        ax=ax,
        xlabel="magnetic field (mT)",
        ylabel="critical current (μA)",
        title="centered fraunhofer",
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
    )
    fig.savefig(str(OUTPATH) + "_centered.png")

if args.symmetrize:
    bfield, ic = symmetrize_fraunhofer(bfield, ic)
    fig, ax = plt.subplots()
    ax.axvline(0, color="k")
    plot(
        bfield / 1e-3,
        ic / 1e-6,
        ax=ax,
        xlabel="magnetic field (mT)",
        ylabel="critical current (μA)",
        title="symmetrized fraunhofer",
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
    )
    fig.savefig(str(OUTPATH) + "_symmetrized.png")

PHI0 = physical_constants["mag. flux quantum"][0]
FIELD_TO_WAVENUM = 2 * np.pi * LENGTH / PHI0
x, jx = extract_current_distribution(bfield, ic, FIELD_TO_WAVENUM, WIDTH, len(bfield))
fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel(r"$x$ (μm)")
ax.set_ylabel(r"$J(x)$ (μA/μm)")
ax.axhline(0, color="k")
ax.plot(x * 1e6, jx)
ax.fill_between(x * 1e6, jx, alpha=0.5)
plt.show()
fig.savefig(str(OUTPATH) + "_current-density.png")

plt.show()
