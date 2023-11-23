"""Analysis of fraunhofer center and maximum.

This extracts the center and maximum of the fraunhofer as a function of some independent
variable specified by CH_VARIABLE in the config.

This can be used e.g. for field alignment or diode analysis.
"""
import argparse
import json
from pathlib import Path
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.signal import savgol_filter

from shabanipy.dvdi import extract_switching_current
from shabanipy.jj import find_fraunhofer_center
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import get_output_dir, jy_pink, load_config, plot2d, write_metadata

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
    "--offset_npoints",
    type=int,
    help=(
        "number of points around 0 bias to average when correcting d.c. offset;"
        "this assumes the current bias is swept symmetrically about 0"
    ),
)
parser.add_argument(
    "--debug",
    "-d",
    default=False,
    action="store_true",
    help="show plots for debugging purposes",
)
args = parser.parse_args()
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

outdir = (
    get_output_dir() / "centermax" / Path(args.config_path).stem / args.config_section
)
print(f"Output directory: {outdir}")
outdirvv = outdir / "centermax-vv"
outdirvv.mkdir(parents=True, exist_ok=True)
jy_pink.register()
plt.style.use(["fullscreen13", "jy_pink"])


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
fraun_maxfit = []
fraun_max = []
datafiles = []
i = 1
while config.get(f"DATAPATH{i}"):
    fig, ax = plt.subplots()
    datafiles.append(config.get(f"DATAPATH{i}"))
    ch_bias = config.get(f"CH_BIAS{i}", config.get("CH_BIAS"))
    ch_field = config.get(f"CH_FIELD_PERP{i}", config.get("CH_FIELD_PERP"))
    ch_meas = config.get(f"CH_MEAS{i}", config.get("CH_MEAS"))
    with ShaBlabberFile(config[f"DATAPATH{i}"]) as f:
        filter_val = config.getfloat(f"FILTER_VAL{i}")
        if filter_val is not None:
            filter_ch = config.get(f"FILTER_CH{i}", config.get("FILTER_CH"))
            filters = [(filter_ch, np.isclose, filter_val)]
        else:
            filter_ch = None
            filters = []

        if filter_val is not None and filter_ch == config["CH_VARIABLE"]:
            var = filter_val
        else:
            var = f.get_fixed_value(config["CH_VARIABLE"])
        variable.append(var)
        print(f"Processing {var}")

        for ch, op, val in (
            (ch_field, np.greater, "FIELD_MIN"),
            (ch_field, np.less, "FIELD_MAX"),
            (ch_bias, np.greater, "BIAS_MIN"),
            (ch_bias, np.less, "BIAS_MAX"),
        ):
            minmax = config.getfloat(f"{val}{i}", config.getfloat(f"{val}"))
            if minmax:
                filters.append((ch, op, minmax))

        b_perp, ibias, meas = f.get_data(
            ch_field,
            ch_bias,
            ch_meas,
            order=(ch_field, ch_bias),
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

    # in case bias sweep was done in disjoint sections about Ic+ and Ic-,
    # plot +ve and -ve bias separately
    for mask, cb in zip((ibias <= 0, ibias >= 0), (True, False)):
        shape = (b_perp.shape[0], -1)
        plot_data(
            *(_[mask].reshape(shape) for _ in (b_perp, ibias, dvdi)), ax=ax, cb=cb
        )

    b_perp = np.unique(b_perp)

    try:
        threshold = config.getfloat(f"THRESHOLD{i}", config.getfloat("THRESHOLD"))
    except ValueError:
        threshold = None
    if config.getboolean(f"THRESHOLD_DC{i}", config.getboolean(f"THRESHOLD_DC")):
        ic_signal = volts
    else:
        ic_signal = dvdi
    side = {"+": "positive", "-": "negative", "+-": "both"}
    offset_npoints = config.getint(
        "OFFSET_NPOINTS{i}", config.getint("OFFSET_NPOINTS", args.offset_npoints)
    )
    offset = config.getfloat("OFFSET{i}", config.getfloat("OFFSET", 0))
    ic = extract_switching_current(
        ibias,
        ic_signal,
        side=side[args.branch],
        threshold=threshold,
        offset=offset,
        offset_npoints=offset_npoints,
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

    center = []
    maxfit = []
    for ii in np.abs(ic):
        try:
            c, m = find_fraunhofer_center(
                b_perp,
                ii,
                field_lim=field_lim,
                debug=args.debug,
                return_max=True,
            )
            center.append(c)
            maxfit.append(m)
        except TypeError as e:
            warn(f"Failed to fit fraunhofer.")
            center.append(np.nan)
            maxfit.append(np.nan)
    if "-" in args.branch:
        maxfit[0] *= -1
    fraun_center.append(center)
    fraun_maxfit.append(maxfit)
    for c in center:
        if b_perp.min() < c and c < b_perp.max():
            ax.axvline(c / 1e-3, color="k", lw=1)

    max_ = [
        np.max(np.where((field_lim[0] < b_perp) & (b_perp < field_lim[1]), i, -np.inf))
        for i in np.abs(ic)
    ]
    if "-" in args.branch:
        max_[0] *= -1
    fraun_max.append(max_)
    [ax.axhline(m / 1e-6, color="k", lw=1) for m in max_]
    fig.savefig(str(outdirvv / f"{Path(config[f'DATAPATH{i}']).stem}_{var}.png"))

    if args.debug:
        plt.show()
    plt.close()
    i += 1
sort_idx = np.argsort(variable)
variable = np.array(variable)[sort_idx]
fraun_center = np.array(fraun_center)[sort_idx]
fraun_maxfit = np.array(fraun_maxfit)[sort_idx]
fraun_max = np.array(fraun_max)[sort_idx]
datafiles = np.array(datafiles)[sort_idx]

database_path = outdir / f"{Path(args.config_path).stem}_{args.config_section}.csv"
if database_path.exists():
    write = None
    while write not in ("y", "n"):
        write = input(f"{database_path} already exists.  Overwrite? [y/n]: ").lower()
    write = True if write.lower() == "y" else False
else:
    write = True
if write:
    df = DataFrame({config["CH_VARIABLE"]: variable})
    if args.branch == "+-":
        df["ic-"] = fraun_max[:, 0]
        df["ic+"] = fraun_max[:, 1]
        df["center-"] = fraun_center[:, 0]
        df["center+"] = fraun_center[:, 1]
        df["ic- from fit"] = fraun_maxfit[:, 0]
        df["ic+ from fit"] = fraun_maxfit[:, 1]
    else:
        df[f"ic{args.branch}"] = fraun_max
        df[f"center{args.branch}"] = fraun_center
        df[f"ic{args.branch} from fit"] = fraun_maxfit
    df["datafile"] = datafiles
    df.to_csv(database_path, index=False)
    print(f"Wrote {database_path}")

    write_metadata(
        outdir / f"{Path(args.config_path).stem}_{args.config_section}_metadata.txt",
        args=args,
    )
