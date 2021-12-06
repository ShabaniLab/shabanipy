# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit SQUID oscillations to a two-junction CPR model."""
import argparse
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from lmfit import Minimizer, Parameters, fit_report
from matplotlib import pyplot as plt
from scipy.constants import eV

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData
from shabanipy.plotting import plot, plot2d, jy_pink
from shabanipy.squid.cpr import finite_transparency_jj_current as cpr
from shabanipy.squid.squid_model import compute_squid_current

# set up the command-line interface
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
    temp_meas = f.get_data(CHAN_TEMP_MEAS)

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

# TODO: make handedness consistent with ./fit_cpr.py
if False:  # np.sign(HANDEDNESS) < 0:
    bfield *= -1
    bfield = np.flip(bfield)
    ic_p = np.flip(ic_p)


def estimate_radians_per_tesla(bfield, ic):
    """Estimate the field-to-phase conversion factor by Fourier transform.

    The strongest frequency component (excluding the dc component) is returned.
    """
    dbs = np.unique(np.diff(bfield))
    try:
        (db,) = dbs
    except ValueError:
        db = np.mean(dbs)
        if not np.allclose(dbs, dbs[0], atol=0):
            warnings.warn(
                "Samples are not uniformly spaced in B-field;"
                "rad/T conversion factor estimate might be poor"
            )
    abs_fft = np.abs(np.fft.rfft(ic)[1:])  # ignore DC component
    freq = np.fft.fftfreq(len(ic), d=db)[1 : len(bfield) // 2 + 1]
    freq_guess = freq[np.argmax(abs_fft)]

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
    return 2 * np.pi * freq_guess


def estimate_phase_offset(bfield, ic, boffset, radians_per_tesla):
    """Estimate the phase offset of the SQUID oscillations.

    The position of the maximum is returned.
    """
    phase = (bfield - boffset) * radians_per_tesla
    phase_guess = phase[np.argmax(ic)]

    # plot the guess
    fig, ax = plot(
        phase / np.pi,
        ic / 1e-6,
        xlabel="SQUID phase estimate [π]",
        ylabel="supercurrent [μA]",
        title="Δφ guess",
        stamp=COOLDOWN_SCAN,
    )
    ax.axvline(phase_guess / np.pi, color="k")
    ax.text(
        0.5,
        0.5,
        f"phase offset = {np.round(phase_guess / np.pi, 2)}π",
        va="center",
        ha="center",
        transform=ax.transAxes,
    )
    fig.savefig(str(OUTPATH) + "_phase-offset.png")
    return phase_guess


# parameterize the fit; assume Ic1 >> Ic2
params = Parameters()
params.add(f"transparency1", value=0.5, max=1)
params.add(f"transparency2", value=0.5, max=1)
if EQUAL_TRANSPARENCIES:
    params["transparency2"].set(expr="transparency1")
params.add(f"switching_current1", value=np.mean(ic_p))
params.add(f"switching_current2", value=(np.max(ic_p) - np.min(ic_p)) / 2)
params.add(f"bfield_offset", value=np.mean(bfield), vary=False)
params.add(f"radians_per_tesla", value=estimate_radians_per_tesla(bfield, ic_p))
params.add(f"phase1", value=0, vary=False)
params.add(
    f"phase2",
    value=estimate_phase_offset(
        bfield, ic_p, params["bfield_offset"], params["radians_per_tesla"]
    )
    % (2 * np.pi),
    min=-np.pi,
    max=np.pi,
)
params.add(f"temperature", value=round(np.mean(temp_meas), 3), vary=False)
params.add(f"gap", value=200e-6 * eV, vary=False)


# define the model to fit
def model(params, bfield):
    """The model function to fit against the data."""
    p = params.valuesdict()
    bfield = bfield - p["bfield_offset"]

    i_squid = compute_squid_current(
        bfield * p["radians_per_tesla"],
        cpr,
        (p["phase1"], p["switching_current1"], p["transparency1"]),
        cpr,
        (p["phase2"], p["switching_current2"], p["transparency2"]),
    )
    return i_squid


# define the objective function to minimize
def residuals(params, bfield, ic):
    """Difference between data and model to minimize with respect to `params`."""
    return ic - model(params, bfield)


# fit the data
mini = Minimizer(residuals, params, fcn_args=(bfield, ic_p))
result = mini.minimize()
print(fit_report(result))
with open(str(OUTPATH) + "_fit-report.txt", "w") as f:
    f.write(fit_report(result))
with open(str(OUTPATH) + "_fit-params.txt", "w") as f:
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
    stamp=COOLDOWN_SCAN,
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
