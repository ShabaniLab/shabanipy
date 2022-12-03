# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Fit SQUID oscillations to a two-junction transparent CPR model."""
import argparse
import subprocess
import sys
from contextlib import redirect_stdout
from functools import partial, reduce
from importlib import import_module
from pathlib import Path
from warnings import warn

import corner
import numpy as np
from lmfit.model import save_modelresult
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.constants import physical_constants

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.squid import SquidModel, estimate_frequency
from shabanipy.utils import load_config, to_dataframe
from shabanipy.utils.plotting import jy_pink, plot, plot2d

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
    choices=["filter", "fit", "fitrolling"],
    help=(
        "how to handle the low-frequency fraunhofer modulation. "
        "high-pass `filter` with FREQUENCY_CUTOFF given in config; "
        "`fit` to a quadratic used as Ic(Φ_ext) of the larger-area junction; "
        "`fitrolling` fits the quadratic to a rolling average with window-length "
        "specified by command-line `--window-length` or config WINDOW_LENGTH"
    ),
)
parser.add_argument(
    "--window-length",
    "-w",
    type=int,
    default=None,
    help="window length to use if --fraunhofer=fitrolling",
)
parser.add_argument(
    "--dry-run",
    "-n",
    default=False,
    action="store_true",
    help="do preliminary analysis but don't run fit",
)
parser.add_argument(
    "--nbrute",
    default=101,
    type=int,
    help="number of points to use in the brute-force optimization of the SQUID current",
)
parser.add_argument(
    "--ninterp",
    default=101,
    type=int,
    help="number of points in Φ ~ [0, 2π] used to interpolate the SQUID behavior",
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
    "--quiet", "-q", default=False, action="store_true", help="do not show plots",
)
parser.add_argument(
    "--verbose",
    "-v",
    default=False,
    action="store_true",
    help="print more information to stdout",
)
args = parser.parse_args()

_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

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
    f"{config['WAFER']}-{config['PIECE']}_{config['LAYOUT']}/fits/{config['DEVICE']}"
)
if not args.dry_run:
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
if not args.dry_run:
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
if not args.dry_run:
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

# handle low-frequency fraunhofer background
if args.fraunhofer and args.branch == "+-":
    raise ValueError("--fraunhofer behavior with --branch +- is not yet defined")
if args.fraunhofer == "filter":
    _, (freqs, fft) = estimate_frequency(bfield, ic_p if args.branch == "+" else ic_n)
    fft_filt = np.where(
        (freqs > 0) & (freqs < config.getfloat("FREQUENCY_CUTOFF")), 0, fft
    )
    ic_filtered = np.fft.irfft(fft_filt, n=len(bfield))
    if args.branch == "+":
        ic_p = ic_filtered
    else:  # args.branch == "-":
        ic_n = ic_filtered
elif args.fraunhofer and args.fraunhofer.startswith("fit"):
    # current assumptions:
    #  - variable background level due to larger-area JJ only
    #  - inductance = 0; TODO for nonzero inductance, need to get IcJJ(Φ) from
    #    IcJJ(Φ_ext) self-consistently
    if args.fraunhofer == "fitrolling":
        wlen = args.window_length
        if wlen is None:
            wlen = config.getint("WINDOW_LENGTH")
        if wlen is None:
            raise ValueError(
                "window length must be specified on command-line with "
                "`--window-length` or in config with WINDOW_LENGTH"
            )
        ic_to_fit = (
            np.convolve(
                ic_p if args.branch == "+" else ic_n, np.ones(wlen), mode="same"
            )
            / wlen
        )
        # remove boundary effects
        ic_to_fit = ic_to_fit[wlen // 2 + 1 : -(wlen // 2 + 1)]
        bfield_to_fit = bfield[wlen // 2 + 1 : -(wlen // 2 + 1)]
    else:
        bfield_to_fit, ic_to_fit = bfield, ic_p if args.branch == "+" else ic_n

    poly = np.polynomial.Polynomial.fit(bfield_to_fit, ic_to_fit, 2)

    fig, ax = plt.subplots()
    fig.suptitle("background Ic fit")
    ax.plot(bfield, ic_p if args.branch == "+" else ic_n, ".", label="data")
    if args.fraunhofer == "fitrolling":
        ax.plot(bfield_to_fit, ic_to_fit, label="rolling average")
    ax.plot(bfield, poly(bfield), label="polyfit")
    ax.legend()
    if not args.dry_run:
        fig.savefig(str(OUTPATH) + "_ic-background.png")

############################
# parameter initialization #
############################

if args.fraunhofer and args.fraunhofer.startswith("fit"):
    model = SquidModel(
        branch=args.branch,
        **{f"critical_current{config.getint('LARGER_AREA_JJ')}": poly(bfield)},
        nbrute=args.nbrute,
        ninterp=args.ninterp,
    )
else:
    model = SquidModel(branch=args.branch, nbrute=args.nbrute, ninterp=args.nbrute)

# initial values can be fixed or guessed in config
for name in model.param_names:
    if name in config:
        model.set_param_hint(name, value=config.getfloat(name), vary=False)
    elif "guess_" + name in config:
        model.set_param_hint(name, value=config.getfloat("guess_" + name))

# enforce equal transparencies
if args.equal_transparencies:
    model.set_param_hint("transparency2", expr="transparency1")

# guess remaining initial parameters
ic = ic_p if args.branch == "+" else ic_n if args.branch == "-" else [ic_p, ic_n]
params = model.guess(
    ic, bfield, temp=temp_meas, smaller_ic_jj=config.getint("SMALLER_IC_JJ")
)

if (
    (params["inductance"].vary or params["inductance"].value != 0)
    and args.fraunhofer
    and args.fraunhofer.startswith("fit")
):
    raise NotImplementedError(
        "if inductance != 0, need to get fraunhofer IcJJ(Φ) "
        "from IcJJ(Φ_ext) self-consistently"
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
if not args.dry_run:
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
    f.write(
        "shabanipy@"
        + subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode("utf-8")
        + f"\n{reduce(lambda a, b: a + ' ' + b, sys.argv)}"
    )
    f.write("\n" + result.fit_report())
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

save_modelresult(result, str(OUTPATH) + "_modelresult.json")

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
phase_ext = 2 * np.pi * popt["loop_area"] * (bfield - popt["bfield_offset"]) / PHI0
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
        phase_ext / (2 * np.pi),
        ic_p / 1e-6,
        ax=ax_p,
        ylabel="switching current (μA)",
        label="data",
        marker=".",
        linewidth=0,
    )
    plot(phase_ext / (2 * np.pi), best_p / 1e-6, ax=ax_p, label="fit")
    ax_p.legend()
if args.branch in {"-", "+-"}:
    plot(
        phase_ext / (2 * np.pi),
        ic_n / 1e-6,
        ax=ax_n,
        ylabel="switching current (μA)",
        label="data",
        marker=".",
        linewidth=0,
    )
    plot(phase_ext / (2 * np.pi), best_n / 1e-6, ax=ax_n, label="fit")
    ax_n.legend()
fig.savefig(str(OUTPATH) + "_fit.png")

# save the fit plot data to csv for later re-plotting
DataFrame(
    {
        "bfield": bfield,
        "phase_ext": phase_ext,
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

if not args.quiet:
    plt.show()
