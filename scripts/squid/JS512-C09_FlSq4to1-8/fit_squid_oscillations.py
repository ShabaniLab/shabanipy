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
from configparser import ConfigParser, ExtendedInterpolation
from contextlib import redirect_stdout
from functools import partial
from importlib import import_module
from pathlib import Path

import corner
import numpy as np
from lmfit import Model
from lmfit.model import save_modelresult
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.constants import eV
from scipy.signal import find_peaks

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.plotting import jy_pink, plot, plot2d
from shabanipy.utils import to_dataframe

from squid_model_func import squid_model_func

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
    "--emcee",
    "-m",
    default=False,
    action="store_true",
    help="run a Markov Chain Monte Carlo sampler and plot with `corner`",
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
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read_file(f)
    config = config["dummy"]


OUTDIR = f"{__file__.split('.py')[0].replace('_', '-')}-results/"
print(f"All output will be saved to `{OUTDIR}`")
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

OUTPATH = Path(OUTDIR) / (config["COOLDOWN"] + "_" + config["SCAN"])
INPATH = Path(config.get("LABBERDATA_DIR", get_data_dir())) / config["DATAPATH"]
AMPS_PER_T = getattr(
    import_module("shabanipy.constants"),
    f"{config['FRIDGE'].upper()}_AMPS_PER_TESLA_{config['PERP_AXIS'].upper()}",
)
if config["FRIDGE"] not in str(INPATH):
    warnings.warn(
        f"I can't double check that {config['DATAPATH']} is from {config['FRIDGE']}"
    )

jy_pink.register()
plt.style.use(["jy_pink", "fullscreen13"])

# load the data
with LabberData(INPATH) as f:
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
    stamp=config["COOLDOWN"] + "_" + config["SCAN"],
)
fig.savefig(str(OUTPATH) + "_raw-data.png")

# extract the switching currents
ic_n, ic_p = extract_switching_current(
    ibias,
    dvdi,
    side="both",
    threshold=config.getfloat("RESISTANCE_THRESHOLD"),
    interp=True,
)
ax.set_title("$I_c$ extraction")
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color="w", lw=0, marker=".")
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color="w", lw=0, marker=".")
fig.savefig(str(OUTPATH) + "_ic-extraction.png")

# in vector10, positive Bx points into the daughterboard (depends on mount orientation)
if config["FRIDGE"] == "vector10":
    bfield = np.flip(bfield) * -1
    ic_p = np.flip(ic_p)
    ic_n = np.flip(ic_n)
# in vector9, positive Bx points out of the daughterboard
elif config["FRIDGE"] == "vector9":
    pass
else:
    warnings.warn(f"I don't recognize fridge `{config['FRIDGE']}`")


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
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
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
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
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


model = Model(squid_model_func, both_branches=config.getboolean("BOTH_BRANCHES"))

# initialize the parameters
params = model.make_params()
params["transparency1"].set(value=0.5, max=1)
params["transparency2"].set(value=0.5, max=1)
if config.getboolean("EQUAL_TRANSPARENCIES"):
    params["transparency2"].set(expr="transparency1")
params["switching_current1"].set(value=(np.max(ic_p) - np.min(ic_p)) / 2)
params["switching_current2"].set(value=np.mean(ic_p))
params["bfield_offset"].set(value=estimate_bfield_offset(bfield, ic_p, ic_n))
params["radians_per_tesla"].set(value=estimate_radians_per_tesla(bfield, ic_p))
# anomalous phases; if both fixed, then there is no phase freedom in the model (aside
# from bfield_offset), as the two gauge-invariant phases are fixed by two constraints:
#     1. flux quantization:         γ1 - γ2 = 2πΦ/Φ_0 (mod 2π),
#     2. supercurrent maximization: I_tot = max_γ1 { I_1(γ1) + I_2(γ1 - 2πΦ/Φ_0) }
params["anom_phase1"].set(value=0, vary=False)
params["anom_phase2"].set(value=0, vary=False)
params["temperature"].set(value=round(np.mean(temp_meas), 3), vary=False)
params["gap"].set(value=200e-6 * eV, vary=False)
params["inductance"].set(value=1e-9)

# scale the residuals to get a somewhat meaningful χ2 value
ibias_step = np.diff(ibias, axis=-1)
uncertainty = np.mean(ibias_step)
if not np.allclose(ibias_step, uncertainty):
    warnings.warn(
        "Bias current has variable step sizes; "
        "the magnitude of the χ2 statistic may not be meaningful."
    )

# fit the data
if not args.dry_run:
    print("Optimizing fit...", end="")
    result = model.fit(
        data=np.array([ic_p, ic_n]).flatten()
        if config.getboolean("BOTH_BRANCHES")
        else ic_p,
        weights=1 / uncertainty,
        bfield=bfield,
        params=params,
        verbose=args.verbose,
    )
    print("...done.")
    print(result.fit_report())
    with open(str(OUTPATH) + "_fit-report.txt", "w") as f:
        f.write(result.fit_report())
    with open(str(OUTPATH) + "_fit-params.txt", "w") as f, redirect_stdout(f):
        print(to_dataframe(result.params))

    if args.conf_interval is not None:
        print("Calculating confidence intervals (this takes a while)...", end="")
        ci_kwargs = dict(verbose=args.verbose)
        if args.conf_interval:
            ci_kwargs.update(dict(sigmas=args.conf_interval))
        result.conf_interval(**ci_kwargs)
        print("...done.")
        print(result.ci_report(ndigits=10))

    save_modelresult(result, str(OUTPATH) + "_model-result.json")

    if args.emcee:
        print("Calculating posteriors with emcee...")
        mcmc_result = model.fit(
            data=np.array([ic_p, ic_n]).flatten()
            if config.getboolean("BOTH_BRANCHES")
            else ic_p,
            weights=1 / uncertainty,
            bfield=bfield,
            params=result.params,
            # nan_policy="omit",
            method="emcee",
            fit_kws=dict(steps=1000, nwalkers=100, burn=200, thin=10, is_weighted=True),
        )
        print("...done.")
        save_modelresult(mcmc_result, str(OUTPATH) + "_mcmc-result.json")
        print("\nemcee medians and (averaged) +-1σ quantiles")
        print("------------------------------")
        print(mcmc_result.fit_report())
        print("\nemcee max likelihood estimates")
        print("------------------------------")
        mle_loc = np.argmax(mcmc_result.lnprob)
        mle_loc = np.unravel_index(mle_loc, mcmc_result.lnprob.shape)
        mle = mcmc_result.chain[mle_loc]
        for i, p in enumerate([p for p in mcmc_result.params.values() if p.vary]):
            print(f"{p.name}: {mle[i]}")

        fig, ax = plt.subplots()
        ax.plot(mcmc_result.acceptance_fraction)
        ax.set_xlabel("walker")
        ax.set_ylabel("acceptance fraction")
        fig.savefig(str(OUTPATH) + "_emcee-acceptance-fraction.png")
        with plt.style.context("classic"):
            fig = corner.corner(
                mcmc_result.flatchain,
                labels=mcmc_result.var_names,
                truths=[p.value for p in mcmc_result.params.values() if p.vary],
                labelpad=0.1,
            )
            fig.savefig(str(OUTPATH) + "_emcee-corner.png")


# plot the best fit and initial guess over the data
popt = result.params.valuesdict() if not args.dry_run else params
phase = (bfield - popt["bfield_offset"]) * popt["radians_per_tesla"]
if config.getboolean("BOTH_BRANCHES"):
    fig, (ax_p, ax_n) = plt.subplots(2, 1, sharex=True)
    plot(
        phase / (2 * np.pi),
        ic_n / 1e-6,
        ax=ax_n,
        xlabel="phase [2π]",
        ylabel="switching current [μA]",
        label="data",
        marker=".",
    )
    if not args.dry_run:
        best_p, best_n = np.split(result.best_fit, 2)
        plot(phase / (2 * np.pi), best_n / 1e-6, ax=ax_n, label="fit")
    init_p, init_n = np.split(model.eval(bfield=bfield, params=params), 2)
    if args.plot_guess:
        plot(phase / (2 * np.pi), init_n / 1e-6, ax=ax_n, label="guess")
else:
    fig, ax_p = plt.subplots()
    if not args.dry_run:
        best_p = result.best_fit
    init_p = model.eval(bfield=bfield, params=params)
plot(
    phase / (2 * np.pi),
    ic_p / 1e-6,
    ax=ax_p,
    xlabel="phase [2π]",
    ylabel="switching current [μA]",
    label="data",
    marker=".",
    stamp=config["COOLDOWN"] + "_" + config["SCAN"],
)
if not args.dry_run:
    plot(phase / (2 * np.pi), best_p / 1e-6, ax=ax_p, label="fit")
if args.plot_guess:
    plot(phase / (2 * np.pi), init_p / 1e-6, ax=ax_p, label="guess")
fig.savefig(str(OUTPATH) + "_fit.png")
DataFrame(
    {
        "bfield": bfield,
        "phase": phase,
        "ic_p": ic_p,
        "fit_p": best_p,
        "init_p": init_p,
        **(
            {"ic_n": ic_n, "fit_n": best_n, "init_n": init_n}
            if config.getboolean("BOTH_BRANCHES")
            else {}
        ),
    }
).to_csv(str(OUTPATH) + "_fit.csv", index=False)
plt.show()
