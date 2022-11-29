# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit the current-phase relation of a Josephson junction.

The applied field is assumed to drive the phase of the SQUID's active junction only; the
idler junction is assumed to be phase-fixed, contributing a constant offset.
"""

import argparse
import warnings
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path

import numpy as np
from lmfit import Minimizer, Parameters, fit_report
from matplotlib import pyplot as plt
from scipy.constants import eV

from shabanipy.dvdi import extract_switching_current, find_rising_edge
from shabanipy.jj import transparent_cpr as cpr
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.utils import load_config, plot, plot2d
from shabanipy.utils.plotting import jy_pink

# set up the command-line interface
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "config_path", help="path to a config file, relative to this script."
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--plot-guess",
    "-g",
    action="store_true",
    default=False,
    help="plot the initial guess along with the best fit",
)
args = parser.parse_args()

_, config = load_config(args.config_path, args.config_section)

inpath = Path(config.get("LABBERDATA_DIR", get_data_dir())) / config["DATAPATH"]
outdir = Path("output/")
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / f"{config['COOLDOWN']}-{config['SCAN']}"

jy_pink.register()
plt.style.use(["jy_pink", "fullscreen13"])

# magnet coil current-per-field conversion factor
if "Source current" in config["CH_FIELD_PERP"]:
    print("Scaling magnet source current to field")
    AMPS_PER_T = getattr(
        import_module("shabanipy.constants"),
        f"{config['FRIDGE'].upper()}_AMPS_PER_TESLA_{config['PERP_AXIS'].upper()}",
    )
else:
    AMPS_PER_T = 1

# sanity check conversion factor is correct (relies on my local file hierarchy)
if config["FRIDGE"] not in str(inpath):
    warnings.warn(
        f"I can't double check that {config['DATAPATH']} is from {config['FRIDGE']}"
    )

# load the data
with LabberData(inpath) as f:
    bfield = f.get_data(config["CH_FIELD_PERP"]) / AMPS_PER_T
    ibias, lockin = f.get_data(config["CH_LOCKIN"], get_x=True)
    dvdi = np.abs(lockin)
    temp_meas = f.get_data(config["CH_TEMP_MEAS"])


# check for significant temperature deviations
MAX_TEMPERATURE_STD = 1e-3
temp_std = np.std(temp_meas)
if temp_std > MAX_TEMPERATURE_STD:
    warnings.warn(
        f"Temperature standard deviation {temp_std} K > {MAX_TEMPERATURE_STD} K"
    )

# plot the raw data
fig, ax = plot2d(
    *np.broadcast_arrays(bfield[..., np.newaxis] / 1e-3, ibias / 1e-6, dvdi),
    xlabel="x coil field (mT)",
    ylabel="dc bias (μA)",
    zlabel="dV/dI (Ω)",
    title="raw data",
    stamp=f"{config['COOLDOWN']}-{config['SCAN']}",
)
fig.savefig(str(outpath) + "_raw-data.png")

# extract the switching currents
ic_n, ic_p = extract_switching_current(
    ibias,
    dvdi,
    side="both",
    threshold=config.getfloat("RESISTANCE_THRESHOLD", fallback=None),
    interp=True,
)
ax.set_title("$I_c$ extraction")
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color="w", lw=0, marker=".")
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color="w", lw=0, marker=".")
fig.savefig(str(outpath) + "_ic-extraction.png")

# in vector10, positive Bx points into the daughterboard
if config["FRIDGE"] == "vector10":
    bfield = np.flip(bfield) * -1
    ic_p = np.flip(ic_p)
# in vector9, positive Bx points out of the daughterboard
elif config["FRIDGE"] == "vector9":
    pass
else:
    warnings.warn(f"I don't recognize fridge `{config['FRIDGE']}`")

# parameterize the fit
params = Parameters()
params.add(f"transparency", value=0.5, max=1)
params.add(f"switching_current", value=(np.max(ic_p) - np.min(ic_p)) / 2)
params.add(f"temperature", value=round(np.mean(temp_meas), 3), vary=False)
params.add(f"gap", value=200e-6 * eV, vary=False)
params.add(f"ic_offset", value=np.mean(ic_p))
params.add(
    f"bfield_offset",
    value=find_rising_edge(bfield, ic_p, threshold=params["ic_offset"]),
)
params.add(f"radians_per_tesla")

# check bfield offset guess
fig, ax = plot(
    bfield / 1e-3,
    ic_p / 1e-6,
    xlabel="field (mT)",
    ylabel="switching current (μA)",
    title="bfield offset guess",
)
ax.axhline(params["ic_offset"] / 1e-6, color="black")
ax.axvline(params["bfield_offset"] / 1e-3, color="black")
fig.savefig(str(outpath) + "_bfield-offset.png")

# guess the field-to-phase factor by FFT
dbs = np.unique(np.diff(bfield))
try:
    (db,) = dbs
except ValueError:
    assert np.allclose(
        dbs, dbs[0], atol=0
    ), "Samples are not uniformly spaced in B-field"
    db = dbs[0]
abs_fft = np.abs(np.fft.rfft(ic_p)[1:])  # ignore DC component
freq = np.fft.fftfreq(len(ic_p), d=db)[1 : len(bfield) // 2 + 1]
freq_guess = freq[np.argmax(abs_fft)]
params["radians_per_tesla"].set(value=2 * np.pi * freq_guess)
# plot the FFT and resulting guess
fig, ax = plot(
    freq / 1e3,
    abs_fft,
    xlabel="frequency [mT$^{-1}$]",
    ylabel="|FFT| [arb. u.]",
    title="frequency guess",
    stamp=f"{config['COOLDOWN']}-{config['SCAN']}",
    marker="o",
)
ax.axvline(freq_guess / 1e3, color="k")
ax.text(
    0.3,
    0.5,
    f"frequency = {np.round(freq_guess / 1e3)} mT$^{{-1}}$\n"
    f"period = {round(1 / freq_guess / 1e-6)} μT",
    transform=ax.transAxes,
)
fig.savefig(str(outpath) + "_fft.png")

# define the model to fit
def model(params, bfield):
    """The model function to fit against the data."""
    p = params.valuesdict()
    return p["ic_offset"] + cpr(
        (bfield - p["bfield_offset"]) * p["radians_per_tesla"],
        p["switching_current"],
        p["transparency"],
        p["temperature"],
        p["gap"],
    )


# define the objective function to minimize
def residuals(params, bfield, ic):
    """Difference between data and model to minimize with respect to `params`."""
    return ic - model(params, bfield)


# fit the data
print("Optimizing fit...", end="")
mini = Minimizer(residuals, params, fcn_args=(bfield, ic_p))
result = mini.minimize()
print("done.")
print(fit_report(result))
with open(str(outpath) + "_fit-report.txt", "w") as f:
    f.write(fit_report(result))
with open(str(outpath) + "_fit-params.txt", "w") as f:
    with redirect_stdout(f):
        result.params.pretty_print(precision=8)

# plot the initial guess and best fit over the data
popt = result.params.valuesdict()
phase = (bfield - popt["bfield_offset"]) * popt["radians_per_tesla"]
fig, ax = plot(
    phase / (2 * np.pi),
    ic_p / 1e-6,
    marker=".",
    xlabel="phase [2π]",
    ylabel="switching current [μA]",
    label="data",
    stamp=f"{config['COOLDOWN']}-{config['SCAN']}",
)
if args.plot_guess:
    plot(
        phase / (2 * np.pi), model(params, bfield) / 1e-6, ax=ax, label="guess",
    )
plot(
    phase / (2 * np.pi), model(result.params, bfield) / 1e-6, ax=ax, label="fit",
)
fig.savefig(str(outpath) + "_fit.png")

plt.show()
