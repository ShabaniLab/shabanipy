"""Extract the density and mobility vs. gate voltage from a quantum Hall measurement."""
import argparse
import json
import warnings
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.constants import e, hbar, m_e

from shabanipy.labber import ShaBlabberFile
from shabanipy.quantum_hall import extract_density, extract_mobility
from shabanipy.utils import load_config, plot

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
OUTDIR = Path("./output/")
OUTDIR.mkdir(exist_ok=True)
print(f"Output directory: {OUTDIR}")
CHIP_ID = f"{config['WAFER']}-{config['PIECE']}"


def get_hall_data(datapath, ch_lockin_meas, ch_lockin_source):
    with ShaBlabberFile(datapath) as f:
        gate, bfield, dvdi = f.get_data(
            config["CH_GATE"],
            config["CH_FIELD_PERP"],
            ch_lockin_meas,
            order=(config["CH_GATE"], config["CH_FIELD_PERP"]),
        )
        ibias_ac = f.get_channel(ch_lockin_source).instrument.config[
            "Output amplitude"
        ] / config.getfloat("R_OUTPUT")
        print(f"AC bias current = {ibias_ac / 1e-6} μA")
        dvdi /= ibias_ac
    return gate, bfield, dvdi.real


gate_xx, bfield_xx, rxx = get_hall_data(
    config["DATAPATH_RXX"], config["CH_LOCKIN_XX_MEAS"], config["CH_LOCKIN_XX_SOURCE"],
)
gate_yy, bfield_yy, ryy = get_hall_data(
    config["DATAPATH_RYY"], config["CH_LOCKIN_YY_MEAS"], config["CH_LOCKIN_YY_SOURCE"],
)
gate_xy, bfield_xy, rxy = get_hall_data(
    config["DATAPATH_RXY"], config["CH_LOCKIN_XY_MEAS"], config["CH_LOCKIN_XY_SOURCE"],
)

# calculate density
if "FIELD_CUTOFFS" in config:
    field_cutoffs = json.loads(config.get("FIELD_CUTOFFS"))
else:
    field_cutoffs = (bfield_xy.min(), bfield_xy.max())
density, density_std, fits = extract_density(bfield_xy, rxy, field_cutoffs)
print(f"Density @ 0V: {density[gate_xy[:, 0] == 0] / 1e16} e12/cm2")

# plot density fits
FITS_DIR = OUTDIR / f"{CHIP_ID}_fits"
FITS_DIR.mkdir(exist_ok=True)
fig, ax = plt.subplots()
for g, b, r, n, n_std, fit in zip(gate_xy, bfield_xy, rxy, density, density_std, fits):
    if len(set(g)) != 1:
        warnings.warn(
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
        stamp=f"{config['COOLDOWN']}_{config['SCAN_RXY']}",
        label="data",
    )
    ax.text(
        0.5,
        0.9,
        f"density $\\approx$ {n / 1e4:.1e} cm$^{-2}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    ax.plot(b, fit.best_fit, label="fit")
    ax.legend()
    fig.savefig(FITS_DIR / f"rxy_{round(g, 5)}V.png")
    plt.cla()

# calculate mobility
print("Assuming bfield sweeps for rxx and ryy are the same")
mobility_xx, mobility_yy = extract_mobility(
    bfield_xx, rxx, ryy, density, config.getfloat("GEOMETRIC_FACTOR")
)
print(
    f"Peak mobility (xx, yy): {mobility_xx.max() / 1e-1, mobility_yy.max() / 1e-1} e3 cm2/Vs"
)

# plot density/mobility vs. gate
ax.clear()
ax.set_title(f"{config.get('FILENAME_PREFIX')}")
ax.set_xlabel("gate voltage (V)")
ax.set_ylabel("density ($10^{12}$ cm$^{-2}$)")
lines_xy = ax.plot(gate_xy[:, 0], density / 1e4 / 1e12, "ko-", label="$n$")
ax2 = ax.twinx()
ax2.set_ylabel("mobility ($10^3$ cm$^2$ / V.s)")
lines_xx = ax2.plot(
    gate_xx[:, 0], mobility_xx / 1e-4 / 1e3, "o-", label="$\mu_\mathrm{xx}$"
)
lines_yy = ax2.plot(
    gate_yy[:, 0], mobility_yy / 1e-4 / 1e3, "o-", label="$\mu_\mathrm{yy}$"
)
lines = lines_xy + lines_xx + lines_yy
ax2.legend(lines, [l.get_label() for l in lines])
fig.savefig(OUTDIR / f"{CHIP_ID}_density-mobility.png")

# compute and save transport parameters vs. gate
mass = 0.03
print(f"Assuming effective mass: {mass} m_e")
print(f"Assuming gate voltage sweeps are the same for Rxx, Ryy, and Rxy")
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
        "Fermi energy (meV)": hbar ** 2 * kf ** 2 / (2 * mass) / 1e-3 / e,
        "mobility xx (e3 cm^2/V.s)": mobility_xx / 1e-4 / 1e3,
        "mobility yy (e3 cm^2/V.s)": mobility_yy / 1e-4 / 1e3,
        "mean free time xx (fs)": txx / 1e-15,
        "mean free time yy (fs)": tyy / 1e-15,
        "mean free path xx (nm)": vf * txx / 1e-9,
        "mean free path yy (nm)": vf * tyy / 1e-9,
        "diffusion coefficient xx (m^2/s)": vf ** 2 * txx / 2,
        "diffusion coefficient yy (m^2/s)": vf ** 2 * tyy / 2,
    }
)
with open(OUTDIR / f"{CHIP_ID}_transport-params.csv", "w") as f:
    df.to_csv(f, index=False)

plt.show()
