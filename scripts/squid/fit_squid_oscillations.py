# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit SQUID oscillations to a two-junction transparent CPR model."""
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
from scipy.constants import eV, physical_constants
from scipy.signal import butter, sosfilt

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.plotting import jy_pink, plot, plot2d
from shabanipy.squid import estimate_boffset, estimate_frequency
from shabanipy.utils import to_dataframe

from squid_model_func import squid_model_func

print = partial(print, flush=True)

PHI0 = physical_constants["mag. flux quantum"][0]

# set up the command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--branch",
    "-b",
    choices=["p", "n", "b"],
    default="n",
    help="fit positive (p), negative (n), or both branches simultaneously (b)",
)
parser.add_argument(
    "--equal-transparencies",
    "-e",
    default=False,
    action="store_true",
    help="constrain junction transparencies to be equal",
)
parser.add_argument(
    "--filter-fraunhofer",
    "-f",
    default=False,
    action="store_true",
    help=(
        "filter out low-frequency fraunhofer envelope; "
        "only supported for --branch=p currently"
    ),
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
    default=None,
    action="store_true",
    help=(
        "plot the initial guess along with the best fit; "
        "if None, defaults to --dry-run"
    ),
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
if args.plot_guess is None:
    args.plot_guess = args.dry_run

# load the config file
with open(Path(__file__).parent / args.config_path) as f:
    print(f"Using config file `{f.name}`")
    ini = ConfigParser(interpolation=ExtendedInterpolation())
    ini.read_file(f)
    config = ini[args.config_section]

# get the path to the datafile
INPATH = Path(config.get("LABBERDATA_DIR", get_data_dir())) / config["DATAPATH"]

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
if config["FRIDGE"] not in str(INPATH):
    warnings.warn(
        f"I can't double check that {config['DATAPATH']} is from {config['FRIDGE']}"
    )

# set up plot styles
jy_pink.register()
plt.style.use(["jy_pink", "fullscreen13"])

# set up output directory and filename prefix
OUTDIR = (
    f"{__file__.split('.py')[0].replace('_', '-')}-results/"
    f"{config['WAFER']}-{config['PIECE']}_{config['LAYOUT']}/"
    f"{config['DEVICE']}"
)
print(f"All output will be saved to `{OUTDIR}`")
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
FILTER_STR = f"_{config.getfloat('FILTER_VALUE')}" if "FILTER_VALUE" in config else ""
OUTPATH = Path(OUTDIR) / (
    f"{config['COOLDOWN']}-{config['SCAN']}{FILTER_STR}_{args.branch}"
)

# load the data
with LabberData(INPATH) as f:
    if config.get("FILTER_CH") and config.getfloat("FILTER_VALUE"):
        filters = {config["FILTER_CH"]: config.getfloat("FILTER_VALUE")}
    else:
        filters = None

    bfield = f.get_data(config["CH_FIELD_PERP"], filters=filters) / AMPS_PER_T
    ibias, lockin = f.get_data(config["CH_LOCKIN"], get_x=True, filters=filters)
    dvdi = np.abs(lockin)
    temp_meas = f.get_data(config["CH_TEMP_MEAS"], filters=filters)
    f.warn_not_constant(config["CH_TEMP_MEAS"])

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
    threshold=config.getfloat("RESISTANCE_THRESHOLD", fallback=None),
    interp=True,
)
ax.set_title("$I_c$ extraction and field limit")
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color="k", lw=1)
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color="k", lw=1)
for field_lim in ("FIELD_MIN", "FIELD_MAX"):
    if config.getfloat(field_lim):
        ax.axvline(config.getfloat(field_lim) / 1e-3, color="black", linestyle="--")
fig.savefig(str(OUTPATH) + "_ic-extraction.png")

# limit field range to fit
for field_lim, op in zip(("FIELD_MIN", "FIELD_MAX"), (np.greater, np.less)):
    if config.getfloat(field_lim):
        mask = op(bfield, config.getfloat(field_lim))
        bfield = bfield[mask, ...]
        ic_p = ic_p[mask, ...]
        ic_n = ic_n[mask, ...]

# vector10 Bx points opposite vector9 Bx
# (vector10 mount orientation has since changed on 2022/01/25)
if config["FRIDGE"] == "vector10":
    bfield = np.flip(bfield) * -1
    ic_p = np.flip(ic_p)
    ic_n = np.flip(ic_n)
elif config["FRIDGE"] == "vector9":
    pass
else:
    warnings.warn(f"I don't recognize fridge `{config['FRIDGE']}`")


model = Model(
    squid_model_func,
    positive=(True,)
    if args.branch == "p"
    else (False,)
    if args.branch == "n"
    else (True, False),  # args.branch == "b"
    param_names=[
        "transparency1",
        "transparency2",
        "switching_current1",
        "switching_current2",
        "bfield_offset",
        "radians_per_tesla",
        "anom_phase1",
        "anom_phase2",
        "temperature",
        "gap",
        "inductance",
    ],
)

# initialize the parameters
params = model.make_params()
params["transparency1"].set(value=0, min=0, max=1)
params["transparency2"].set(value=0, min=0, max=1)
if args.equal_transparencies:
    params["transparency2"].set(expr="transparency1")
ic_amp = [np.abs(np.max(ic) - np.min(ic)) / 2 for ic in [ic_p, ic_n]]
ic_mean = [np.abs(np.mean(ic)) for ic in [ic_p, ic_n]]
if args.branch == "p":
    ic_amp_guess = ic_amp[0]
    ic_mean_guess = ic_mean[0]
elif args.branch == "n":
    ic_amp_guess = ic_amp[-1]
    ic_mean_guess = ic_mean[-1]
else:  # args.branch == "b"
    ic_amp_guess = np.mean(ic_amp)
    ic_mean_guess = np.mean(ic_mean)
params["switching_current1"].set(value=ic_amp_guess, min=0)
params["switching_current2"].set(value=ic_mean_guess, min=0)
boffset = config.getfloat("BOFFSET_GUESS")
if boffset is None:
    boffset, (peak_idxs, valley_idxs) = estimate_boffset(
        bfield,
        ic_p if args.branch != "n" else None,
        ic_n if args.branch != "p" else None,
    )
params["bfield_offset"].set(value=boffset)

# filter out low-frequency fraunhofer envelope
cyc_per_T, (freqs, fft) = estimate_frequency(
    bfield, ic_p if args.branch in {"p", "b"} else ic_n
)
if args.filter_fraunhofer:
    # manual highpass step-filter
    fft_filt = np.where(
        (freqs > 0) & (freqs < config.getfloat("LOOP_AREA") / PHI0), 0, fft
    )
    ic_filtered = np.fft.irfft(fft_filt, n=len(bfield))
    cyc_per_T, (freqs_filt, fft_filt) = estimate_frequency(bfield, ic_filtered)
params["radians_per_tesla"].set(value=2 * np.pi * cyc_per_T)
# anomalous phases; if both fixed, then there is no phase freedom in the model (aside
# from bfield_offset), as the two gauge-invariant phases are fixed by two constraints:
#     1. flux quantization:         γ1 - γ2 = 2πΦ/Φ_0 (mod 2π),
#     2. supercurrent maximization: I_tot = max_γ1 { I_1(γ1) + I_2(γ1 - 2πΦ/Φ_0) }
params["anom_phase1"].set(value=0, vary=False)
params["anom_phase2"].set(value=0, vary=False)
params["temperature"].set(value=round(np.mean(temp_meas), 3), vary=False)
params["gap"].set(value=200e-6 * eV, vary=False)
params["inductance"].set(value=1e-9)

# plot the radians_per_tesla estimate and fraunhofer filter
abs_fft = np.abs(fft)
fig, ax = plot(
    freqs / 1e3,
    abs_fft,
    xlabel="frequency (mT$^{-1}$)",
    ylabel="|FFT| (arb.)",
    title="frequency estimate",
    stamp=config["COOLDOWN"] + "_" + config["SCAN"],
    label="data",
)
ax.set_ylim(0, np.max(abs_fft[1:]) * 1.05)  # ignore dc component
ax.axvline(cyc_per_T / 1e3, color="k")
ax.text(
    0.3,
    0.5,
    f"frequency $\sim$ {np.round(cyc_per_T / 1e3)} mT$^{{-1}}$\n"
    f"period $\sim$ {round(1 / cyc_per_T / 1e-6)} μT",
    transform=ax.transAxes,
)
if args.filter_fraunhofer:
    plot(
        freqs_filt / 1e3, np.abs(fft_filt), ax=ax, label="filtered",
    )
    ax.legend()
fig.savefig(str(OUTPATH) + "_fft.png")

# from now on deal with the filtered data
if args.filter_fraunhofer:
    if args.branch == "p":
        ic_p = ic_filtered
    elif args.branch == "n":
        ic_n = ic_filtered
    else:  # args.branch == "b"
        raise NotImplementedError(
            "Filtering both positive and negative branches is currently unsupported"
        )

# plot the bfield_offset estimate
if args.branch == "p":
    fig, ax_p = plt.subplots()
    ax_p.set_xlabel("x coil field (mT)")
    ax_p.set_title("bfield offset estimate")
elif args.branch == "n":
    fig, ax_n = plt.subplots()
    ax_n.set_xlabel("x coil field (mT)")
    ax_n.set_title("bfield offset estimate")
else:  # args.branch == "b":
    fig, (ax_p, ax_n) = plt.subplots(2, 1, sharex=True)
    ax_n.set_xlabel("x coil field (mT)")
    ax_p.set_title("bfield offset estimate")
if args.branch in {"p", "b"}:
    plot(
        bfield / 1e-3,
        ic_p / 1e-6,
        ylabel="switching current (μA)",
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
        ax=ax_p,
        marker=".",
    )
    ax_p.axvline(boffset / 1e-3, color="k")
    if "BOFFSET_GUESS" not in config:
        ax_p.plot(
            bfield[peak_idxs] / 1e-3, ic_p[peak_idxs] / 1e-6, lw=0, marker="o",
        )
if args.branch in {"n", "b"}:
    plot(
        bfield / 1e-3,
        ic_n / 1e-6,
        ylabel="switching current (μA)",
        stamp=config["COOLDOWN"] + "_" + config["SCAN"],
        ax=ax_n,
        marker=".",
    )
    ax_n.axvline(boffset / 1e-3, color="k")
    if "BOFFSET_GUESS" not in config:
        ax_n.plot(
            bfield[valley_idxs] / 1e-3, ic_n[valley_idxs] / 1e-6, lw=0, marker="o",
        )
ax = ax_p if args.branch in {"p", "b"} else ax_n
ax.text(
    0.5,
    0.5,
    f"bfield offset $\\approx$ {np.round(boffset / 1e-3, 3)} mT",
    va="center",
    ha="center",
    transform=ax.transAxes,
)
fig.savefig(str(OUTPATH) + "_bfield-offset.png")

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
    if args.branch == "p":
        data = ic_p
    elif args.branch == "n":
        data = ic_n
    else:  # args.branch == "b"
        data = np.array([ic_p, ic_n]).flatten()
    result = model.fit(
        data=data,
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
            data=data,
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
if args.branch == "p":
    fig, ax_p = plt.subplots()
    ax_p.set_xlabel("phase (2π)")
    init_p = model.eval(bfield=bfield, params=params)
    if not args.dry_run:
        best_p = result.best_fit
elif args.branch == "n":
    fig, ax_n = plt.subplots()
    ax_n.set_xlabel("phase (2π)")
    init_n = model.eval(bfield=bfield, params=params)
    if not args.dry_run:
        best_n = result.best_fit
else:  # args.branch == "b"
    fig, (ax_p, ax_n) = plt.subplots(2, 1, sharex=True)
    ax_n.set_xlabel("phase (2π)")
    init_p, init_n = np.split(model.eval(bfield=bfield, params=params), 2)
    if not args.dry_run:
        best_p, best_n = np.split(result.best_fit, 2)
if args.branch in {"p", "b"}:
    plot(
        phase / (2 * np.pi),
        ic_p / 1e-6,
        ax=ax_p,
        ylabel="switching current (μA)",
        label="data",
        marker=".",
        linewidth=0,
    )
    if not args.dry_run:
        plot(phase / (2 * np.pi), best_p / 1e-6, ax=ax_p, label="fit")
    ax_p.legend()
if args.branch in {"n", "b"}:
    plot(
        phase / (2 * np.pi),
        ic_n / 1e-6,
        ax=ax_n,
        ylabel="switching current (μA)",
        label="data",
        marker=".",
        linewidth=0,
    )
    if not args.dry_run:
        plot(phase / (2 * np.pi), best_n / 1e-6, ax=ax_n, label="fit")
    ax_n.legend()
fig.savefig(str(OUTPATH) + "_fit.png")
if args.plot_guess:
    if args.branch in {"p", "b"}:
        plot(phase / (2 * np.pi), init_p / 1e-6, ax=ax_p, label="guess")
        ax_p.legend()
    if args.branch in {"n", "b"}:
        plot(phase / (2 * np.pi), init_n / 1e-6, ax=ax_n, label="guess")
        ax_n.legend()
    fig.savefig(str(OUTPATH) + "_guess.png")

# save the fit plot data to csv for later re-plotting
if not args.dry_run:
    DataFrame(
        {
            "bfield": bfield,
            "phase": phase,
            **(
                {"ic_p": ic_p, "fit_p": best_p, "init_p": init_p}
                if args.branch in {"p", "b"}
                else {}
            ),
            **(
                {"ic_n": ic_n, "fit_n": best_n, "init_n": init_n}
                if args.branch in {"n", "b"}
                else {}
            ),
        }
    ).to_csv(str(OUTPATH) + "_fit.csv", index=False)

plt.show()
