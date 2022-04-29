# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit SQUID oscillations to a two-junction transparent CPR model."""
import argparse
import sys
from configparser import ConfigParser, ExtendedInterpolation
from contextlib import redirect_stdout
from functools import partial
from importlib import import_module
from pathlib import Path
from warnings import warn

import corner
import numpy as np
from lmfit import Model
from lmfit.model import save_modelresult
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.constants import physical_constants

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.plotting import jy_pink, plot, plot2d
from shabanipy.squid import estimate_boffset, estimate_frequency, squid_model
from shabanipy.utils import to_dataframe

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
    choices=["+", "-", "+-"],
    default="+",
    help="fit positive (+), negative (-), or both branches simultaneously (+-)",
)
parser.add_argument(
    "--equal-transparencies",
    "-e",
    default=False,
    action="store_true",
    help="constrain junction transparencies to be equal",
)
parser.add_argument(
    "--fraunhofer",
    "-f",
    choices=["filter", "fit"],
    help=(
        "how to handle the low-frequency fraunhofer modulation. "
        "high-pass `filter` with FREQUENCY_CUTOFF given in config; "
        "`fit` to a quadratic used as Ic(Φ_ext) of the larger-area junction"
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
    warn(f"I can't double check that {config['DATAPATH']} is from {config['FRIDGE']}")

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

#######################
# data pre-processing #
#######################

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

# vector9 Bx points out of the daughterboard; vector10 Bx points opposite
# (vector10 mount orientation has since changed on 2022/01/25)
if config["FRIDGE"] == "vector10":
    bfield = np.flip(bfield) * -1
    ic_p = np.flip(ic_p)
    ic_n = np.flip(ic_n)
elif config["FRIDGE"] == "vector9":
    pass
else:
    warn(f"I don't recognize fridge `{config['FRIDGE']}`")

############################
# parameter initialization #
############################
# TODO clean this up; priority order for init value overrides:
#   0. CLI for testing values
#   1. param in config (to fix value) for saving/fixing values
#   2. param_guess in config (to vary in fit) for saving/varying values
#   3. estimate value as last resort
# priority 3 can be refactored into guess() method of lmfit.Model subclass

# create the model
param_specs = {
    "bfield_offset": None,
    "radians_per_tesla": {"min": 0},
    "anomalous_phase1": {"value": 0, "vary": False},
    "anomalous_phase2": {"value": 0, "vary": False},
    "critical_current1": {"min": 0},
    "critical_current2": {"min": 0},
    "transparency1": {"value": 0, "min": 0, "max": 1},
    "transparency2": {"value": 0, "min": 0, "max": 1},
    "inductance": {"value": 0, "min": 0, "vary": False},
    "temperature": {"vary": False},
    "gap": {"value": 200e-6, "vary": False},
    "nbrute": {"value": 101, "vary": False},
    "ninterp": {"value": 101, "vary": False},
}
model = Model(squid_model, branch=args.branch, param_names=list(param_specs.keys()))

# initial values can be fixed or guessed in config
for name, spec in param_specs.items():
    if spec is not None:
        model.set_param_hint(name, **spec)
    if name in config:
        model.set_param_hint(name, value=config.getfloat(name), vary=False)
    elif name + "_guess" in config:
        model.set_param_hint(name, value=config.getfloat(name + "_guess"))

# estimate magnetic field offset
if "bfield_offset" not in model.param_hints:
    boffset, *_ = estimate_boffset(
        bfield,
        ic_p if args.branch != "-" else None,
        ic_n if args.branch != "+" else None,
    )
    model.set_param_hint("bfield_offset", value=boffset)

# plot magnetic field offset
fig, ax = plt.subplots()
fig.suptitle("magnetic field offset")
ax.plot(
    bfield - model.param_hints["bfield_offset"]["value"],
    ic_p if args.branch in {"+", "+-"} else ic_n,
)
ax.axvline(0, color="k")
fig.savefig(str(OUTPATH) + "_bfield-offset.png")

# enforce equal transparencies
if args.equal_transparencies:
    model.set_param_hint("transparency2", expr="transparency1")

# estimate junction critical currents
amplitude = [np.abs(np.max(ic) - np.min(ic)) / 2 for ic in [ic_p, ic_n]]
mean = [np.abs(np.mean(ic)) for ic in [ic_p, ic_n]]
# SMALLER_IC_JJ should be 1 or 2, to specify which JJ has the smaller critical current.
# Numbering and sign conventions are documented in `shabanipy.squid.squid` module.
if "SMALLER_IC_JJ" in config:
    SMALLER_IC_JJ = config.getint("SMALLER_IC_JJ")
    BIGGER_IC_JJ = SMALLER_IC_JJ % 2 + 1
    if args.branch == "+":
        amplitude_guess = amplitude[0]
        mean_guess = mean[0]
    elif args.branch == "-":
        amplitude_guess = amplitude[-1]
        mean_guess = mean[-1]
    else:  # args.branch == "+-"
        amplitude_guess = np.mean(amplitude)
        mean_guess = np.mean(mean)
    if f"CRITICAL_CURRENT{SMALLER_IC_JJ}_GUESS" not in config:
        model.set_param_hint(f"critical_current{SMALLER_IC_JJ}", value=amplitude_guess)
    if f"CRITICAL_CURRENT{BIGGER_IC_JJ}_GUESS" not in config:
        model.set_param_hint(f"critical_current{BIGGER_IC_JJ}", value=mean_guess)
else:
    guess = np.mean([amplitude, mean])
    for critical_current in ("critical_current1", "critical_current2"):
        if critical_current + "_guess" not in config:
            model.set_param_hint(f"critical_current1", value=guess)

if args.fraunhofer == "fit":
    if args.branch == "+-":
        raise ValueError("background fitting is not yet supported for option +-")
    poly = np.polynomial.Polynomial.fit(bfield, ic_p if args.branch == "+" else ic_n, 2)

    # this is bad...make separate lmfit.model subclasses
    background_ic = f"critical_current{config.getint('LARGER_AREA_JJ')}"
    # if nonzero inductance, this assumes IcJJ(Φ_ext) = IcJJ(Φ)
    model.opts[background_ic] = poly(bfield)
    model.param_names.remove(background_ic)
    model.param_hints.pop(background_ic)

    fig, ax = plt.subplots()
    fig.suptitle("background Ic fit")
    ax.plot(bfield, ic_p if args.branch == "+" else ic_n, ".")
    ax.plot(bfield, poly(bfield))
    fig.savefig(str(OUTPATH) + "_ic-background.png")

# estimate frequency of oscillations
# TODO switch lmfit model to take area instead of radians_per_tesla
if "LOOP_AREA_GUESS" in config:
    model.set_param_hint(
        "radians_per_tesla", value=2 * np.pi * config.getfloat("LOOP_AREA_GUESS") / PHI0
    )
cyc_per_T, (freqs, fft) = estimate_frequency(
    bfield, ic_p if args.branch in {"+", "+-"} else ic_n
)
# filter out low-frequency fraunhofer envelope
if args.fraunhofer == "filter":
    # manual highpass step-filter
    fft_filt = np.where(
        (freqs > 0) & (freqs < config.getfloat("FREQUENCY_CUTOFF")), 0, fft
    )
    ic_filtered = np.fft.irfft(fft_filt, n=len(bfield))
    cyc_per_T, (freqs_filt, fft_filt) = estimate_frequency(bfield, ic_filtered)
if "LOOP_AREA_GUESS" not in config:
    model.set_param_hint("radians_per_tesla", value=2 * np.pi * cyc_per_T)

# plot the radians_per_tesla estimate and fraunhofer filter
abs_fft = np.abs(fft)
fig, ax = plot(
    freqs / 1e3,
    abs_fft,
    xlabel="frequency (mT$^{-1}$)",
    ylabel="|FFT| (arb.)",
    title="frequency estimate",
)
ax.set_ylim(0, np.max(abs_fft[1:]) * 1.05)  # ignore dc component
ax.axvline(cyc_per_T / 1e3, color="k")
ax.text(
    0.3,
    0.5,
    f"frequency $\sim$ {np.round(cyc_per_T / 1e3)} mT$^{{-1}}$\n"
    f"period $\sim$ {round(1 / cyc_per_T / 1e-6)} μT\n"
    f"area $\sim$ {round(PHI0 * cyc_per_T / 1e-12)} μT$^2$",
    transform=ax.transAxes,
)
if args.fraunhofer == "filter":
    plot(
        freqs_filt / 1e3, np.abs(fft_filt), ax=ax, label="filtered",
    )
    ax.legend()
fig.savefig(str(OUTPATH) + "_fft.png")

# initialize remaining parameters
model.set_param_hint("temperature", value=round(np.mean(temp_meas), 3), vary=False)

params = model.make_params()

# from now on deal only with the filtered data
if args.fraunhofer == "filter":
    if args.branch == "+":
        ic_p = ic_filtered
    elif args.branch == "-":
        ic_n = ic_filtered
    else:  # args.branch == "+-"
        raise NotImplementedError(
            "Filtering both positive and negative branches is currently unsupported"
        )

# plot initial guess
fig, ax = plt.subplots(len(args.branch), 1, sharex=True)
fig.suptitle("initial guess")
if args.branch == "+":
    ax.plot(bfield, ic_p, ".")
    init_p = model.eval(bfield=bfield, params=params)
    ax.plot(bfield, init_p)
elif args.branch == "-":
    ax.plot(bfield, ic_n, ".")
    init_n = model.eval(bfield=bfield, params=params)
    ax.plot(bfield, init_n)
else:  # args.branch == "+-"
    ax[0].plot(bfield, ic_p, ".")
    ax[1].plot(bfield, ic_n, ".")
    init_p, init_n = model.eval(bfield=bfield, params=params)
    ax[0].plot(bfield, init_p)
    ax[1].plot(bfield, init_n)
fig.savefig(str(OUTPATH) + "_init.png")

#######################
# fit, plot, and save #
#######################

if args.dry_run:
    plt.show()
    sys.exit()

# scale the residuals to get a somewhat meaningful χ2 value
ibias_step = np.diff(ibias, axis=-1)
uncertainty = np.mean(ibias_step)
if not np.allclose(ibias_step, uncertainty):
    warn(
        "Bias current has variable step sizes; "
        "the magnitude of the χ2 statistic may not be meaningful."
    )

print("Optimizing fit...", end="")
if args.branch == "+":
    data = ic_p
elif args.branch == "-":
    data = ic_n
else:  # args.branch == "+-"
    data = [ic_p, ic_n]
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

# confidence intervals
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

# plot the best fit over the data
popt = result.params.valuesdict()
phase = (bfield - popt["bfield_offset"]) * popt["radians_per_tesla"]
if args.branch == "+":
    fig, ax_p = plt.subplots()
    ax_p.set_xlabel("$\Phi_\mathrm{ext}$ ($\Phi_0$)")
    best_p = result.best_fit
elif args.branch == "-":
    fig, ax_n = plt.subplots()
    ax_n.set_xlabel("$\Phi_\mathrm{ext}$ ($\Phi_0$)")
    best_n = result.best_fit
else:  # args.branch == "+-"
    fig, (ax_p, ax_n) = plt.subplots(2, 1, sharex=True)
    ax_n.set_xlabel("$\Phi_\mathrm{ext}$ ($\Phi_0$)")
    best_p, best_n = result.best_fit
if args.branch in {"+", "+-"}:
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
if args.branch in {"-", "+-"}:
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

# save the fit plot data to csv for later re-plotting
if not args.dry_run:
    DataFrame(
        {
            "bfield": bfield,
            "phase": phase,
            **(
                {"ic_p": ic_p, "fit_p": best_p, "init_p": init_p}
                if args.branch in {"+", "+-"}
                else {}
            ),
            **(
                {"ic_n": ic_n, "fit_n": best_n, "init_n": init_n}
                if args.branch in {"-", "+-"}
                else {}
            ),
        }
    ).to_csv(str(OUTPATH) + "_fit.csv", index=False)

plt.show()
