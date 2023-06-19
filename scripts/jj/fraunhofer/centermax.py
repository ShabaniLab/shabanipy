"""Analysis of fraunhofer center and maximum.

This extracts the center and maximum of the fraunhofer as a function of some independent
variable specified by CH_VARIABLE in the config.

This can be used e.g. for field alignment or diode analysis.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.signal import savgol_filter

from shabanipy.dvdi import extract_switching_current
from shabanipy.jj import find_fraunhofer_center
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import load_config, plot, plot2d

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
    default="+",
    choices=["+", "-", "+-"],
    help="which branch of critical current to analyze",
)
parser.add_argument(
    "--align",
    "-a",
    default=False,
    action="store_true",
    help="fit fraunhofer center vs. CH_VARIABLE for field alignment angle",
)
parser.add_argument(
    "--debug",
    "-d",
    default=False,
    action="store_true",
    help="show additional plots for debugging purposes",
)
args = parser.parse_args()
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

plt.style.use(["fullscreen13"])
outdir = Path(f"output/{Path(args.config_path).stem}")
outdir.mkdir(exist_ok=True, parents=True)


def plot_data(b_perp, ibias, dvdi, ax=None, cb=True):
    return plot2d(
        b_perp / 1e-3,
        ibias / 1e-6,
        dvdi,
        xlabel="out-of-plane field (mT)",
        ylabel="current bias (μA)",
        zlabel=dvdi_label,
        title=f"{config['CH_VARIABLE']} = {var}",
        ax=ax,
        stamp=f"{config['FRIDGE']}/{config[f'DATAPATH{i}']}",
        vmin=config.getfloat("VMIN"),
        vmax=config.getfloat("VMAX"),
        extend_min=False,
        colorbar=cb,
    )


variable = []
fraun_center = []
fraun_max = []
datafiles = []
i = 1
while config.get(f"DATAPATH{i}"):
    datafiles.append(config.get(f"DATAPATH{i}"))
    ch_bias = config.get(f"CH_BIAS{i}", config["CH_BIAS"])
    ch_meas = config.get(f"CH_MEAS{i}", config["CH_MEAS"])
    with ShaBlabberFile(config[f"DATAPATH{i}"]) as f:
        filter_val = config.getfloat(f"FILTER_VAL{i}")
        if filter_val is not None:
            filters = [
                (
                    config.get(f"FILTER_CH{i}", config["CH_VARIABLE"]),
                    np.isclose,
                    filter_val,
                )
            ]
            var = filter_val
        else:
            filters = None
            var = f.get_fixed_value(config["CH_VARIABLE"])
        variable.append(var)
        print(f"Processing {var}")
        b_perp, ibias, meas = f.get_data(
            config["CH_FIELD_PERP"],
            ch_bias,
            ch_meas,
            order=(config["CH_FIELD_PERP"], ch_bias),
            filters=filters,
        )
        if ch_meas.endswith(("VI curve", "SingleValue")):  # DC volts from DMM
            volts = meas / config.getfloat("DC_AMP_GAIN")
            dvdi = np.gradient(volts, axis=-1) / np.gradient(ibias, axis=-1)
            dvdi_label = "ΔV/ΔΙ (Ω)"
        else:  # differential Ω from lock-in
            dvdi = np.abs(meas)
            dvdi_label = "|dV/dI| (Ω)"
            if ch_meas.endswith("Value"):  # lock-in directly (not VICurveTracer)
                ibias_ac = f.get_channel(ch_meas).instrument.config[
                    "Output amplitude"
                ] / config.getfloat("R_AC_OUT")
                dvdi /= ibias_ac
                ibias /= config.getfloat("R_DC_OUT")

    # in case bias sweep was done in disjoint sections about Ic+ and Ic-
    ibias_1d = np.unique(ibias, axis=0).squeeze()
    ibias_deltas = np.unique(np.diff(ibias_1d))
    if np.allclose(ibias_deltas, ibias_deltas[0]):
        fig, ax = plot_data(b_perp, ibias, dvdi)
    else:
        fig, ax = plt.subplots()
        for mask, cb in zip((ibias_1d < 0, ibias_1d > 0), (True, False)):
            fig, ax = plot_data(
                b_perp[:, mask], ibias[:, mask], dvdi[:, mask], ax=ax, cb=cb
            )

    b_perp = np.unique(b_perp)

    threshold = config.getfloat(f"THRESHOLD{i}", config.getfloat("THRESHOLD"))
    if config.getboolean(f"THRESHOLD_DC{i}", config.getboolean(f"THRESHOLD_DC")):
        ic_signal = volts
    else:
        ic_signal = dvdi
    side = {"+": "positive", "-": "negative", "+-": "both"}
    ic = extract_switching_current(
        ibias, ic_signal, side=side[args.branch], threshold=threshold
    )
    if args.branch != "+-":
        ic = [ic]
    savgol_wlen = config.getint(f"SAVGOL_WLEN{i}", config.getint(f"SAVGOL_WLEN", None))
    if savgol_wlen is not None:
        ic = [savgol_filter(i, savgol_wlen, 2) for i in ic]
    [ax.plot(b_perp / 1e-3, i / 1e-6, "k-", lw=1) for i in ic]

    if args.debug:
        plt.show()

    field_lim = config.get(f"FIELD_LIM{i}")
    if field_lim is not None:
        field_lim = tuple(json.loads(field_lim))
    else:
        field_lim = (-np.inf, np.inf)

    outpath = str(outdir / f"{Path(config[f'DATAPATH{i}']).stem}")
    center = [
        find_fraunhofer_center(b_perp, i, field_lim=field_lim, debug=args.debug)
        for i in np.abs(ic)
    ]
    fraun_center.append(center)
    [ax.axvline(c / 1e-3, color="k", lw=1) for c in center]

    max_ = [
        np.max(np.where((field_lim[0] < b_perp) & (b_perp < field_lim[1]), i, -np.inf))
        for i in np.abs(ic)
    ]
    if "-" in args.branch:
        max_[0] *= -1
    fraun_max.append(max_)
    [ax.axhline(m / 1e-6, color="k", lw=1) for m in max_]
    fig.savefig(outpath + f"_{var}_centermax.png")

    i += 1
sort_idx = np.argsort(variable)
variable = np.array(variable)[sort_idx]
fraun_center = np.array(fraun_center)[sort_idx]
fraun_max = np.array(fraun_max)[sort_idx]
datafiles = np.array(datafiles)[sort_idx]

database_path = outdir / f"{args.config_section}.csv"
if database_path.exists():
    write = input(f"{database_path} already exists.  Overwrite? [y/n]: ")
    write = True if write.lower() == "y" else False
else:
    write = True
if write:
    df = DataFrame(
        {
            "datafile": datafiles,
            config["CH_VARIABLE"]: variable,
        },
    )
    if args.branch == "+-":
        df["ic-"] = fraun_max[:, 0]
        df["ic+"] = fraun_max[:, 1]
    else:
        df[f"ic{args.branch}"] = fraun_max
    df.to_csv(database_path, index=False)

last_scan = re.split("-|_|\.", config[f"DATAPATH{i-1}"])[-2]
stamp = f"{config['FRIDGE']}/{config['DATAPATH1'].removesuffix('.hdf5')}$-${last_scan}"
outpath = f"output/{Path(config['DATAPATH1']).stem}-{last_scan}"
ic_label = (
    ("$I_{c-}$", "$I_{c+}$") if args.branch == "+-" else f"$I_{{c{args.branch}}}$"
)
fig, ax = plot(
    variable,
    fraun_center / 1e-3,
    "o-",
    xlabel=config["CH_VARIABLE"],
    ylabel="fraunhofer center (mT)",
    stamp=stamp,
    label=ic_label,
)
if args.align:
    for fc in fraun_center.T:
        m, b = np.polyfit(variable, fc, 1)
        ax.plot(
            variable,
            (m * variable + b) / 1e-3,
            label=f"arcsin$(y/x)$ = {round(np.degrees(np.arcsin(m)), 3)} deg",
        )
ax.legend()
fig.savefig(outpath + "_center.png")

fig, ax = plot(
    variable,
    fraun_max / 1e-6,
    "o-",
    color="tab:blue",
    xlabel=config["CH_VARIABLE"],
    ylabel="critical current (μA)",
    stamp=stamp,
)
for fm in fraun_max.T:
    ax.fill_between(variable, fm / 1e-6, color="tab:blue", alpha=0.5)
ax.set_xlim((variable.min(), variable.max()))
fig.savefig(outpath + "_max.png")

if args.branch == "+-":
    fig, ax = plot(
        variable,
        np.abs(fraun_max) / 1e-6,
        "o-",
        xlabel=config["CH_VARIABLE"],
        ylabel="critical current (μA)",
        label=ic_label,
        stamp=stamp,
    )
    ax.fill_between(variable, *np.abs(fraun_max.T / 1e-6), color="tab:gray", alpha=0.5)
    fig.savefig(outpath + "_absmax.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    plot(
        variable,
        np.diff(np.abs(fraun_max)).squeeze() / 1e-9,
        "o-",
        xlabel=config["CH_VARIABLE"],
        ylabel="$\Delta I_c$ (nA)",
        ax=ax,
        stamp=stamp,
    )
    fig.savefig(outpath + "_deltamax.png")

if i < 10:
    plt.show()
print("All plots written to output/")
