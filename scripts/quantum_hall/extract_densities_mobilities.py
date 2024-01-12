"""Extract the density and mobility vs. gate voltage from a quantum Hall measurement."""
import argparse
import json
from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.constants import e, hbar, m_e

from shabanipy.labber import ShaBlabberFile
from shabanipy.quantum_hall import extract_density, extract_mobility
from shabanipy.utils import get_output_dir, load_config, plot, write_metadata

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

OUTDIR = (
    get_output_dir()
    / "density-mobility"
    / Path(args.config_path).stem
    / args.config_section
)
print(f"Output directory: {OUTDIR}")
OUTPATH = OUTDIR / f"{Path(args.config_path).stem}"
OUTDIRVV = OUTDIR / "fits"
OUTDIRVV.mkdir(parents=True, exist_ok=True)

plt.style.use("fullscreen13")

CH_GATE = config.get("CH_GATE", "gate - Source voltage")
STAMP = f"{config['FRIDGE']}/{config[f'DATAPATH_RXY']}"

filters = []
if config.get("EXCLUDE_GATES"):
    for g in json.loads(config.get("EXCLUDE_GATES")):
        filters.append((CH_GATE, lambda ch, g: ~np.isclose(ch, g), g))


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
    return gate, bfield, dvdi.real


gate_xx, bfield_xx, rxx = get_hall_data(
    config["DATAPATH_RXX"], config.get("CH_LOCKIN_XX", "Rxx - Value")
)
gate_yy, bfield_yy, ryy = get_hall_data(
    config["DATAPATH_RYY"], config.get("CH_LOCKIN_YY", "Ryy - Value")
)
gate_xy, bfield_xy, rxy = get_hall_data(
    config["DATAPATH_RXY"], config.get("CH_LOCKIN_XY", "Rxy - Value")
)
if config.getboolean("INVERT_RXY"):
    rxy *= -1

# calculate density
if "FIELD_CUTOFFS" in config:
    field_cutoffs = json.loads(config.get("FIELD_CUTOFFS"))
else:
    field_cutoffs = (bfield_xy.min(), bfield_xy.max())
density, density_std, fits = extract_density(bfield_xy, rxy, field_cutoffs)
gate0mask = np.isclose(gate_xy[:, 0], 0)
print(
    "Density @ 0V: "
    f"{density[gate0mask].squeeze() / 1e16:.2f} "
    f"(+/- {density_std[gate0mask].squeeze() / 1e16:.2f}) e12/cm2"
)

# plot density fits
fig, ax = plt.subplots()
for g, b, r, n, n_std, fit in zip(gate_xy, bfield_xy, rxy, density, density_std, fits):
    if len(set(g)) != 1:
        warn(
            f"Gate voltage is assumed constant for each field sweep, but {min(g)=}, {max(g)=}"
        )
    g = g[0]
    plot(
        b,
        r,
        "o",
        ax=ax,
        xlabel="out-of-plane field (T)",
        ylabel=r"$R_{xy}$ (Ω)",
        title=f"gate voltage = {round(g, 5)} V",
        stamp=STAMP,
        label="data",
    )
    ax.text(
        0.5,
        0.9,
        f"density $\\approx$ {n / 1e16:.2f} ($\\pm$ {n_std / 1e16:.2f}) e12/cm$^2$",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    ax.plot(fit.xdata, fit.best_fit, label="fit")
    ax.legend()
    fig.savefig(OUTDIRVV / f"rxy_{g:+.3f}V.png")
    plt.cla()

# calculate mobility
warn("Assuming bfield sweeps for rxx and ryy are the same")
mobility_xx, mobility_yy = extract_mobility(
    bfield_xx, rxx, ryy, density, config.getfloat("GEOMETRIC_FACTOR")
)
print(
    "Peak mobility (xx, yy): "
    f"({mobility_xx.max() / 1e-1:.1f}, {mobility_yy.max() / 1e-1:.1f}) e3 cm2/Vs"
)
idx_xx = np.argmax(mobility_xx)
idx_yy = np.argmax(mobility_yy)
print(
    "Density @ peak mobility (xx, yy): "
    f"{density[idx_xx] / 1e16:.2f}, {density[idx_yy] / 1e16:.2f} e12/cm2"
)

# plot density/mobility vs. gate
ax.clear()
fig, _ = plot(
    gate_xy[:, 0],
    density / 1e16,
    "ko-",
    label="$n$",
    xlabel="gate voltage (V)",
    ylabel="density ($10^{12}$ cm$^{-2}$)",
    ax=ax,
    stamp=STAMP,
)
ax.errorbar(
    gate_xy[:, 0],
    density / 1e16,
    yerr=density_std / 1e16,
    fmt="none",
    ecolor="k",
)
ax2 = ax.twinx()
plot(
    gate_xx[:, 0],
    np.array([mobility_xx, mobility_yy]).T / 1e-1,
    "o-",
    label=("$\mu_\mathrm{xx}$", "$\mu_\mathrm{yy}$"),
    ylabel="mobility ($10^3$ cm$^2$ / V.s)",
    ax=ax2,
)
lines = ax.get_lines() + ax2.get_lines()
ax.get_legend().remove()
ax2.legend(lines, [l.get_label() for l in lines])
fig.savefig(str(OUTPATH) + "_density-mobility-vs-gate.png")

# plot mobility vs. density
fig, ax = plot(
    density / 1e16,
    np.array([mobility_xx, mobility_yy]).T / 1e-1,
    "o-",
    xlabel="density ($10^{12}$ cm$^{-2}$)",
    ylabel="mobility ($10^3$ cm$^2$ / V.s)",
    stamp=STAMP,
)
ax.legend(["$\mu_\mathrm{xx}$", "$\mu_\mathrm{yy}$"])
fig.savefig(str(OUTPATH) + "_mobility-vs-density.png")


# compute and save transport parameters vs. gate
mass = 0.03
warn(f"Assuming effective mass: {mass} m_e")
warn(f"Assuming gate voltage sweeps are the same for Rxx, Ryy, and Rxy")
mass *= m_e
kf = np.sqrt(2 * np.pi * density)
vf = hbar * kf / mass
txx = mobility_xx * mass / e
tyy = mobility_yy * mass / e
df = DataFrame(
    {
        "gate (V)": gate_xx[:, 0],
        "density (e12 cm^-2)": density / 1e4 / 1e12,
        "Fermi wavenumber (nm^-1)": kf / 1e9,
        "Fermi wavelength (nm)": 2 * np.pi / kf / 1e-9,
        "Fermi velocity (e6 m/s)": vf / 1e6,
        "Fermi energy (meV)": hbar**2 * kf**2 / (2 * mass) / 1e-3 / e,
        "mobility xx (e3 cm^2/V.s)": mobility_xx / 1e-4 / 1e3,
        "mobility yy (e3 cm^2/V.s)": mobility_yy / 1e-4 / 1e3,
        "mean free time xx (fs)": txx / 1e-15,
        "mean free time yy (fs)": tyy / 1e-15,
        "mean free path xx (nm)": vf * txx / 1e-9,
        "mean free path yy (nm)": vf * tyy / 1e-9,
        "diffusion coefficient xx (m^2/s)": vf**2 * txx / 2,
        "diffusion coefficient yy (m^2/s)": vf**2 * tyy / 2,
    }
)
with open(str(OUTPATH) + "_transport-params.csv", "w") as f:
    df.to_csv(f, index=False)

write_metadata(str(OUTPATH) + f"_metadata.txt", args=args)

plt.show()
