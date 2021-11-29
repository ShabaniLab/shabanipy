# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit the current-phase relation of a Josephson junction.

The applied field is assumed to drive the phase of the SQUID's active junction only; the
idler junction is assumed to be phase-fixed.
"""

import argparse
from pathlib import Path

import numpy as np
from lmfit import Minimizer, Parameters, fit_report
from matplotlib import pyplot as plt
from scipy.constants import eV

from shabanipy.dvdi import extract_switching_current, find_rising_edge
from shabanipy.labber import LabberData
from shabanipy.plotting import plot, plot2d, jy_pink
from shabanipy.squid.cpr import finite_transparency_jj_current as cpr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "config_path", help="path to a config file, relative to this script."
)
args = parser.parse_args()

# load the config file
with open(Path(__file__).parent / args.config_path) as f:
    print(f"Using config file `{f.name}`")
    exec(f.read())

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
plt.style.use(["jy_pink", "fullscreen13"])

# load the data
with LabberData(DATAPATH) as f:
    bfield = f.get_data(CHAN_FIELD_PERP) / AMPS_PER_T
    ibias, lockin = f.get_data(CHAN_LOCKIN, get_x=True)
    dvdi = np.abs(lockin)
#    temp_set = f.get_data(CHAN_TEMP)
#    temp_meas = f.get_data(CHAN_TEMP_MEAS)

# plot the raw data
fig, ax = plot2d(
    *np.broadcast_arrays(bfield[..., np.newaxis] / 1e-3, ibias / 1e-6, dvdi),
    xlabel="x coil field (mT)",
    ylabel="dc bias (μA)",
    zlabel="dV/dI (Ω)",
    title="raw data",
    stamp=COOLDOWN_SCAN,
)
fig.savefig(str(OUTPATH) + "_raw-data.png")

# extract the switching currents
ic_n, ic_p = extract_switching_current(
    ibias, dvdi, side="both", threshold=RESISTANCE_THRESHOLD
)
ax.set_title("$I_c$ extraction")
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color="w", lw=0, marker=".")
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color="w", lw=0, marker=".")
fig.savefig(str(OUTPATH) + "_ic-extraction.png")

if np.sign(HANDEDNESS) < 0:
    bfield *= -1
    bfield = np.flip(bfield)
    ic_p = np.flip(ic_p)

# parameterize the fit
params = Parameters()
params.add(f"transparency", value=0.5, max=1)
params.add(f"switching_current", value=(np.max(ic_p) - np.min(ic_p)) / 2)
# TODO set temperature from setpoint or MC average
params.add(f"temperature", value=600e-3, vary=False)
params.add(f"gap", value=200e-6 * eV, vary=False)
params.add(f"ic_offset", value=np.mean(ic_p))
params.add(
    f"bfield_offset",
    value=find_rising_edge(bfield, ic_p, threshold=params["ic_offset"]),
)
params.add(f"radians_per_tesla")

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
    stamp=COOLDOWN_SCAN,
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
fig.savefig(str(OUTPATH) + "_fft.png")

# define the model to fit
def model(params, bfield):
    """The model function to fit against the data `ic` vs `bfield`."""
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
mini = Minimizer(residuals, params, fcn_args=(bfield, ic_p))
result = mini.minimize()
print(fit_report(result))
popt = result.params.valuesdict()

# plot the initial guess and best fit over the data
phase = (bfield - popt["bfield_offset"]) * popt["radians_per_tesla"]
fig, ax = plot(
    phase / (2 * np.pi),
    ic_p / 1e-6,
    marker=".",
    xlabel="phase [2π]",
    ylabel="switching current [μA]",
    label="data",
)
if PLOT_GUESS:
    plot(
        phase / (2 * np.pi), model(params, bfield) / 1e-6, ax=ax, label="guess",
    )
plot(
    phase / (2 * np.pi), model(result.params, bfield) / 1e-6, ax=ax, label="fit",
)
fig.savefig(str(OUTPATH) + "_fit.png")

plt.show()
