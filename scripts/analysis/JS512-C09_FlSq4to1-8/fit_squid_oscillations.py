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
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from lmfit import Minimizer, Parameters, ci_report, conf_interval, fit_report
from lmfit.minimizer import MinimizerResult
from matplotlib import pyplot as plt
from scipy.constants import eV
from scipy.signal import find_peaks

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData
from shabanipy.plotting import plot, plot2d, jy_pink
from shabanipy.squid.cpr import finite_transparency_jj_current as cpr
from shabanipy.squid.squid_model import compute_squid_current

print = partial(print, flush=True)
# set up the command-line interface
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "config_path", help="path to a config file, relative to this script."
)
parser.add_argument(
    "--dry-run",
    "-n",
    default=False,
    action="store_true",
    help="do preliminary analysis but don't run fit",
)
parser.add_argument(
    "--plot-guess",
    "-g",
    default=False,
    action="store_true",
    help="plot the initial guess along with the best fit",
)
parser.add_argument(
    "--conf-interval",
    "-c",
    nargs="*",
    type=int,
    metavar=("σ1", "σ2"),
    help="calculate confidence intervals (optional list of ints specifying sigma values to pass to lmfit.conf_interval)",
)
parser.add_argument(
    "--verbose",
    "-v",
    default=False,
    action="store_true",
    help="print more information to stdout",
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
    ibias, dvdi, side="both", threshold=RESISTANCE_THRESHOLD, interp=True,
)
ax.set_title("$I_c$ extraction")
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color="w", lw=0, marker=".")
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color="w", lw=0, marker=".")
fig.savefig(str(OUTPATH) + "_ic-extraction.png")

# in vector10, positive Bx points into the daughterboard
if FRIDGE == "vector10":
    bfield = np.flip(bfield) * -1
    ic_p = np.flip(ic_p)
    ic_n = np.flip(ic_n)
# in vector9, positive Bx points out of the daughterboard
elif FRIDGE == "vector9":
    pass
else:
    warnings.warn(f"I don't recognize fridge `{FRIDGE}`")


def estimate_radians_per_tesla(bfield: np.ndarray, ic: np.ndarray):
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


def estimate_bfield_offset(bfield: np.ndarray, ic_p: np.ndarray, ic_n=None):
    """Estimate the coil field at which the true flux is integral.

    The position of a maximum near the center of the `bfield` range is returned.
    If `ic_n` is given, a better estimate is midway between a peak in `ic_p` and the
    closest valley in `ic_n`.
    """
    peak_locs, _ = find_peaks(ic_p, prominence=(np.max(ic_p) - np.min(ic_p)) / 2)
    guess_loc = peak_locs[len(peak_locs) // 2]
    bfield_guess = bfield[guess_loc]
    if ic_n is not None:
        peak_locs_n, _ = find_peaks(-ic_n, prominence=(np.max(ic_n) - np.min(ic_n)) / 2)
        guess_loc_n = peak_locs_n[len(peak_locs_n) // 2]
        bfield_guess_n = bfield[guess_loc_n]
        bfield_guess = (bfield_guess + bfield_guess_n) / 2

    # plot the guess
    if ic_n is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("x coil field [mT]")
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        ax2.set_xlabel("x coil field [mT]")
        plot(
            bfield / 1e-3, ic_n / 1e-6, ylabel="supercurrent [μA]", ax=ax2, marker=".",
        )
        ax2.plot(
            bfield[peak_locs_n] / 1e-3,
            ic_n[peak_locs_n] / 1e-6,
            lw=0,
            marker="*",
            label="$I_{c-}$ peaks",
        )
        ax2.plot(
            bfield[guess_loc_n] / 1e-3,
            ic_n[guess_loc_n] / 1e-6,
            lw=0,
            marker="*",
            label="$I_{c-}$ guess",
        )
        ax2.axvline(bfield_guess / 1e-3, color="k")
        ax2.legend()
    plot(
        bfield / 1e-3,
        ic_p / 1e-6,
        ylabel="supercurrent [μA]",
        title="bfield offset guess",
        stamp=COOLDOWN_SCAN,
        ax=ax,
        marker=".",
    )
    ax.axvline(bfield_guess / 1e-3, color="k")
    ax.plot(
        bfield[peak_locs] / 1e-3,
        ic_p[peak_locs] / 1e-6,
        lw=0,
        marker="*",
        label="$I_{c+}$ peaks",
    )
    ax.plot(
        bfield[guess_loc] / 1e-3,
        ic_p[guess_loc] / 1e-6,
        lw=0,
        marker="*",
        label="$I_{c+}$ guess",
    )
    ax.text(
        0.5,
        0.5,
        f"bfield offset $\\approx$ {np.round(bfield_guess / 1e-3, 3)} mT",
        va="center",
        ha="center",
        transform=ax.transAxes,
    )
    ax.legend()
    fig.savefig(str(OUTPATH) + "_bfield-offset.png")
    return bfield_guess


# parameterize the fit
params = Parameters()
params.add(f"transparency1", value=0.5, max=1)
params.add(f"transparency2", value=0.5, max=1)
if EQUAL_TRANSPARENCIES:
    params["transparency2"].set(expr="transparency1")
params.add(f"switching_current1", value=(np.max(ic_p) - np.min(ic_p)) / 2)
params.add(f"switching_current2", value=np.mean(ic_p))
params.add(f"bfield_offset", value=estimate_bfield_offset(bfield, ic_p, ic_n))
params.add(f"radians_per_tesla", value=estimate_radians_per_tesla(bfield, ic_p))
# anomalous phases; if both fixed, then there is no phase freedom in the model (aside
# from bfield_offset), as the two gauge-invariant phases are fixed by two constraints:
#     1. flux quantization:         γ1 - γ2 = 2πΦ/Φ_0 (mod 2π),
#     2. supercurrent maximization: I_tot = max_γ1 { I_1(γ1) + I_2(γ1 - 2πΦ/Φ_0) }
params.add(f"anom_phase1", value=0, vary=False)
params.add(f"anom_phase2", value=0, vary=False)
params.add(f"temperature", value=round(np.mean(temp_meas), 3), vary=False)
params.add(f"gap", value=200e-6 * eV, vary=False)
params.add(f"inductance", value=1e-9)

# define the model to fit
def squid_model(params: Parameters, bfield: np.ndarray, positive: bool = True):
    """The model function to fit against the data."""
    p = params.valuesdict()
    bfield = bfield - p["bfield_offset"]

    i_squid = compute_squid_current(
        bfield * p["radians_per_tesla"],
        cpr,
        (p["anom_phase1"], p["switching_current1"], p["transparency1"]),
        cpr,
        (p["anom_phase2"], p["switching_current2"], p["transparency2"]),
        positive=positive,
        inductance=p["inductance"],
    )
    return i_squid


# define the objective function to minimize
def residuals(
    params: Parameters,
    bfield: np.ndarray,
    ic_p: np.ndarray,
    ic_n: Optional[np.ndarray] = None,
):
    """Difference between data and model to minimize with respect to `params`.

    If the negative critical current branch is provided via `ic_n`, both branches are
    fit simultaneously by concatenating the positive- and negative-branch arrays.
    """
    data = ic_p
    model = squid_model(params, bfield)
    if ic_n is not None:
        data = np.concatenate((data, ic_n))
        model = np.concatenate((model, squid_model(params, bfield, positive=False)))
    return data - model


# fit the data
if args.dry_run:
    result = None
else:
    print("Optimizing fit...", end="")
    mini = Minimizer(
        residuals, params, fcn_args=(bfield, ic_p, ic_n if BOTH_BRANCHES else None)
    )
    result = mini.minimize()
    print("...done.")
    print(fit_report(result))
    with open(str(OUTPATH) + "_fit-report.txt", "w") as f:
        f.write(fit_report(result))
    with open(str(OUTPATH) + "_fit-params.txt", "w") as f:
        with redirect_stdout(f):
            result.params.pretty_print(precision=8)

    if args.conf_interval is not None:
        print("Calculating confidence intervals (this takes a while)...", end="")
        sigmas = args.conf_interval if args.conf_interval else None
        ci = conf_interval(mini, result, sigmas=sigmas, verbose=args.verbose)
        print("...done.")
        print(ci_report(ci, ndigits=10))

# plot the initial guess and best fit over the data
def plot_fit(
    bfield: np.ndarray,
    ic: np.ndarray,
    result: Optional[MinimizerResult] = None,
    guess: Optional[Parameters] = None,
    positive: bool = True,
    ax: Optional[plt.Axes] = None,
):
    popt = (
        result.params.valuesdict()
        if result is not None
        else guess.valuesdict()
        if guess is not None
        else params
    )
    phase = (bfield - popt["bfield_offset"]) * popt["radians_per_tesla"]
    _, ax = plot(
        phase / (2 * np.pi),
        ic / 1e-6,
        marker=".",
        xlabel="phase [2π]",
        ylabel="switching current [μA]",
        label="data",
        stamp=COOLDOWN_SCAN,
        ax=ax,
    )
    if result is not None:
        plot(
            phase / (2 * np.pi),
            squid_model(result.params, bfield, positive=positive) / 1e-6,
            ax=ax,
            label="fit",
        )
    if guess is not None:
        zero_inductance = guess.copy()
        zero_inductance["inductance"].set(value=0)
        plot(
            phase / (2 * np.pi),
            squid_model(zero_inductance, bfield, positive=positive) / 1e-6,
            ax=ax,
            label="guess (L=0)",
            ls=":",
        )
        plot(
            phase / (2 * np.pi),
            squid_model(guess, bfield, positive=positive) / 1e-6,
            ax=ax,
            label="guess",
        )


if BOTH_BRANCHES:
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    plot_fit(
        bfield,
        ic_n,
        result=result,
        guess=params if args.plot_guess else None,
        positive=False,
        ax=ax2,
    )
else:
    fig, ax = plt.subplots()
plot_fit(bfield, ic_p, result=result, guess=params if args.plot_guess else None, ax=ax)
fig.savefig(str(OUTPATH) + "_fit.png")
plt.show()
