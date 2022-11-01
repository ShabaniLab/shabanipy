"""Extract the density and mobility vs. gate voltage from a quantum Hall measurement."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from lmfit.models import LinearModel
from matplotlib import pyplot as plt
from scipy import constants as cs

from shabanipy.labber import LabberData, get_data_dir
from shabanipy.quantum_hall import extract_density, extract_mobility
from shabanipy.quantum_hall.conversion import (
    GEOMETRIC_FACTORS,
    convert_lock_in_meas_to_diff_res,
    diffusion_constant_from_mobility_density,
    fermi_velocity_from_density,
    mean_free_time_from_mobility,
)
from shabanipy.utils import load_config, plot

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--no_show", "-n", default=False, action="store_true", help="do not show plots"
)
args = parser.parse_args()

_, config = load_config(args.config_path, args.config_section)

plt.style.use("fullscreen13")
OUTPATH = Path("output")
OUTPATH.mkdir(exist_ok=True)


def get_hall_data(datapath):
    with LabberData(get_data_dir() / datapath) as f:
        gate_axis = f.get_axis(config.get("CH_GATE"))
        gate = f.get_data(config.get("CH_GATE"))
        gate = np.moveaxis(gate, gate_axis, 0)

        if config.get("CH_FIELD_PERP") in f.channel_names:
            bfield = f.get_data(config.get("CH_FIELD_PERP"))
            bfield = np.moveaxis(bfield, gate_axis, 0)
        else:
            bfield = np.zeros(1)

        vmeas = f.get_data(config.get("LOCKIN_NAME") + " - Value").real
        vmeas = np.moveaxis(vmeas, gate_axis, 0)
        vsource = f.get_config_value(config.get("LOCKIN_NAME"), "Output amplitude")
        isource = vsource / config.getfloat("R_OUTPUT")
        rmeas = vmeas / isource

    return gate, bfield, rmeas


gate_xx, bfield_xx, rmeas_xx = get_hall_data(config.get("DATAPATH_RXX"))
gate_yy, bfield_yy, rmeas_yy = get_hall_data(config.get("DATAPATH_RYY"))
gate_xy, bfield_xy, rmeas_xy = get_hall_data(config.get("DATAPATH_RXY"))

if "FIELD_BOUNDS" in config:
    field_bounds = json.loads(config.get("FIELD_BOUNDS"))
else:
    field_bounds = (bfield_xy.min(), bfield_xy.max())
density, density_std, fits = extract_density(bfield_xy, rmeas_xy, field_bounds)

fig, ax = plt.subplots()
for g, b, r, d, d_std, fit in zip(
    gate_xy, bfield_xy, rmeas_xy, density, density_std, fits
):
    if len(set(g)) != 1:
        warnings.warn(f"Gate voltage is assumed constant, but {min(g)=}, {max(g)=}")
    g = g[0]
    plot(
        b / 1e-3,
        r,
        "o",
        ax=ax,
        xlabel="out-of-plane field (mT)",
        ylabel=r"$R_{xy}$ (Ω)",
        title=f"gate voltage = {round(g, 5)} V",
        stamp=f"{config.get('COOLDOWN')}_{config.get('SCAN_RXY')}",
        label="data",
    )
    ax.text(
        0.5,
        0.9,
        f"density $\\approx$ {d / 1e4:.1e} cm$^{-2}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    ax.plot(b / 1e-3, fit.best_fit, label="fit")
    ax.legend()
    fig.savefig(OUTPATH / f"rxy_{round(g, 5)}V.png")
    plt.cla()

# this is particular to the JS703-NW2 dataset;
# the data in this set is inhomogeneous; filter out the extra gate values

# keep only 0.1V steps in Rxx and Ryy
gate_xx = gate_xx[::10]
gate_yy = gate_yy[::5]
r0_xx = rmeas_xx[::10]
r0_yy = rmeas_yy[::5]
# all Rxy gate sweeps are identical
gate_xy = gate_xy[:, 0]
# remove insulating regime at large negative voltages
r0_xx = r0_xx[gate_xx >= config.getfloat("GATE_MIN")]
r0_yy = r0_yy[gate_yy >= config.getfloat("GATE_MIN")]
density_filtered = density[gate_xy >= config.getfloat("GATE_MIN")]
gate_xx = gate_xx[gate_xx >= config.getfloat("GATE_MIN")]
gate_yy = gate_yy[gate_yy >= config.getfloat("GATE_MIN")]
gate_xy = gate_xy[gate_xy >= config.getfloat("GATE_MIN")]
# convert sheet resistance
r0_xx *= config.getfloat("GEOMETRIC_FACTOR")
r0_yy *= config.getfloat("GEOMETRIC_FACTOR")
# calculate mobility
mobility_xx = 1 / cs.e / density_filtered / r0_xx
mobility_yy = 1 / cs.e / density_filtered / r0_yy

fig, ax = plt.subplots()
ax.set_title(f"{config.get('FILENAME_PREFIX')}")
ax.set_xlabel("gate voltage (V)")
ax.set_ylabel("density ($10^{12}$ cm$^{-2}$)")
lines1 = ax.plot(gate_xy, density_filtered / 1e4 / 1e12, "ko-", label="$n$")
ax2 = ax.twinx()
ax2.set_ylabel("mobility (cm$^2$ / V.s)")
lines2 = ax2.plot(gate_xx, mobility_xx / 1e-4, "o-", label="$\mu_\mathrm{xx}$")
lines3 = ax2.plot(gate_yy, mobility_yy / 1e-4, "o-", label="$\mu_\mathrm{yy}$")
lines = lines1 + lines2 + lines3
ax2.legend(lines, [l.get_label() for l in lines])
fig.savefig(OUTPATH / "density-mobility-vs-gate.png")

if not args.no_show:
    plt.show()
