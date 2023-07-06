"""Reconstruct critical current distribution from Fraunhofer interference pattern."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants

from shabanipy.dvdi import extract_switching_current
from shabanipy.jj import (
    extract_current_distribution,
    recenter_fraunhofer,
    symmetrize_fraunhofer,
)
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import get_output_dir, jy_pink, load_config, plot, plot2d

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
    "--width",
    "-w",
    type=float,
    help="junction width (m)",
)
parser.add_argument(
    "--length",
    "-l",
    type=float,
    help="junction length (m)",
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
outdir = get_output_dir() / "current-reconstruction"
print(f"Output directory: {outdir}")
outdirvv = outdir / Path(args.config_path).stem
outdirvv.mkdir(parents=True, exist_ok=True)
SLICE_STR = (
    f"_idx={config.getfloat('3RD_AXIS_INDEX')}" if "3RD_AXIS_INDEX" in config else ""
)
datapath = Path(config["DATAPATH"])
outpath = outdir / f"{datapath.stem}{SLICE_STR}"
outpathvv = outdirvv / f"{datapath.stem}{SLICE_STR}"
jy_pink.register()
plt.style.use(["jy_pink", "fullscreen13"])

if config.getint("3RD_AXIS_INDEX") is not None:
    SLICES = (..., config.getint("3RD_AXIS_INDEX"))
else:
    SLICES = None

filters = []
if config.getfloat("FIELD_MIN"):
    filters.append((config["CH_FIELD_PERP"], np.greater, config.getfloat("FIELD_MIN")))
if config.getfloat("FIELD_MAX"):
    filters.append((config["CH_FIELD_PERP"], np.less, config.getfloat("FIELD_MAX")))

# load the data
with ShaBlabberFile(datapath) as f:
    bfield, ibias, dvdi = f.get_data(
        config["CH_FIELD_PERP"],
        config["CH_BIAS"],
        config["CH_MEAS"],
        order=(config["CH_FIELD_PERP"], config["CH_BIAS"]),
        slices=SLICES,
        filters=filters,
    )
    if f.get_channel(config["CH_BIAS"]).unitPhys == "V":
        ibias /= config.getfloat("R_DC_OUT")
    ibias_ac = f.get_channel(config["CH_MEAS"]).instrument.config[
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
    stamp=datapath.stem,
)
fig.savefig(str(outpathvv) + "_raw-data.png")

# extract the switching current
bfield = np.unique(bfield)  # assumes all field sweeps are identical
ic = extract_switching_current(
    ibias,
    dvdi,
    threshold=config.getfloat("THRESHOLD", fallback=None),
    interp=True,
)
ax.set_title("switching current")
plot(bfield / 1e-3, ic / 1e-6, ax=ax, color="k", lw=1)
fig.savefig(str(outpathvv) + "_ic-extraction.png")

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
        stamp=datapath.stem,
    )
    fig.savefig(str(outpathvv) + "_centered.png")

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
        stamp=datapath.stem,
    )
    fig.savefig(str(outpathvv) + "_symmetrized.png")

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
fig.savefig(str(outpath) + "_current-density.png")

plt.show()
