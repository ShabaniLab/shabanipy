import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
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
    help="which branch of critical current to use",
)
parser.add_argument(
    "--align",
    "-a",
    default=False,
    action="store_true",
    help="plot fraunhofer center vs. in-plane for field alignment",
)
parser.add_argument(
    "--max",
    "-m",
    default=False,
    action="store_true",
    help="plot fraunhofer max (of main lobe) vs. in-plane",
)
parser.add_argument(
    "--diode",
    "-d",
    default=False,
    action="store_true",
    help="plot difference between positive and negative branch critical currents",
)
parser.add_argument(
    "--debug",
    "-g",
    default=False,
    action="store_true",
    help="show additional plots for debugging purposes",
)
args = parser.parse_args()
if args.diode:
    args.branch = "+-"
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

plt.style.use(["fullscreen13"])
Path("output").mkdir(exist_ok=True)


def plot_data(b_perp, ibias, dvdi, ax=None, cb=True):
    return plot2d(
        b_perp / 1e-3,
        ibias / 1e-6,
        dvdi,
        xlabel="out-of-plane field (mT)",
        ylabel="current bias (μA)",
        zlabel=dvdi_label,
        title=f"in-plane field = {round(inplane / 1e-3)} mT",
        ax=ax,
        stamp=f"{config['FRIDGE']}/{config[f'DATAPATH{i}']}",
        vmin=config.getfloat("VMIN"),
        vmax=config.getfloat("VMAX"),
        extend_min=False,
        colorbar=cb,
    )


b_inplane = []
fraun_center = []
fraun_max = []
i = 1
while config.get(f"DATAPATH{i}"):
    ch_bias = config.get(f"CH_BIAS{i}", config["CH_BIAS"])
    ch_meas = config.get(f"CH_MEAS{i}", config["CH_MEAS"])
    with ShaBlabberFile(config[f"DATAPATH{i}"]) as f:
        filter_val = config.getfloat(f"FILTER_VAL{i}")
        if filter_val is not None:
            filters = [
                (
                    config.get(f"FILTER_CH{i}", config["CH_FIELD_INPLANE"]),
                    np.equal,
                    filter_val,
                )
            ]
            inplane = filter_val
        else:
            filters = None
            inplane = f.get_fixed_value(config["CH_FIELD_INPLANE"])
        b_inplane.append(inplane)
        print(f"Processing {inplane=}")
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

    outpath = f"output/{Path(config[f'DATAPATH{i}']).stem}"
    center = [
        find_fraunhofer_center(b_perp, i, field_lim=field_lim, debug=args.debug)
        for i in np.abs(ic)
    ]
    fraun_center.append(center)
    [ax.axvline(c / 1e-3, color="k", lw=1) for c in center]
    fig.savefig(outpath + f"_{inplane=}_fraun-center.png")

    max_ = [
        np.max(np.where((field_lim[0] < b_perp) & (b_perp < field_lim[1]), i, -np.inf))
        for i in np.abs(ic)
    ]
    if "-" in args.branch:
        max_[0] *= -1
    fraun_max.append(max_)
    [ax.axhline(m / 1e-6, color="k", lw=1) for m in max_]
    fig.savefig(outpath + f"_{inplane=}_fraun-max.png")

    i += 1
sort_idx = np.argsort(b_inplane)
b_inplane = np.array(b_inplane)[sort_idx]
fraun_center = np.array(fraun_center)[sort_idx]
fraun_max = np.array(fraun_max)[sort_idx]
last_scan = re.split("-|_|\.", config[f"DATAPATH{i-1}"])[-2]
stamp = f"{config['FRIDGE']}/{config['DATAPATH1'].removesuffix('.hdf5')}$-${last_scan}"
outpath = f"output/{Path(config['DATAPATH1']).stem}-{last_scan}"

if args.align:
    fig, ax = plot(
        b_inplane / 1e-3,
        fraun_center / 1e-3,
        "o-",
        xlabel="in-plane field (mT)",
        ylabel="fraunhofer center (mT)",
        stamp=stamp,
    )
    for fc in fraun_center.T:
        m, b = np.polyfit(b_inplane, fc, 1)
        ax.plot(
            b_inplane / 1e-3,
            (m * b_inplane + b) / 1e-3,
            label=f"arcsin$(B_\perp/B_\parallel)$ = {round(np.degrees(np.arcsin(m)), 3)} deg",
        )
    ax.legend()
    fig.savefig(outpath + "_field-alignment.png")

if args.max or args.diode:
    fig, ax = plot(
        b_inplane / 1e-3,
        (np.abs(fraun_max) if args.diode else fraun_max) / 1e-6,
        "o-",
        xlabel="in-plane field (mT)",
        ylabel="critical current (μA)",
        stamp=stamp,
        color="tab:blue" if args.max else None,
        label=("$I_{c-}$", "$I_{c+}$") if args.diode else None,
    )
    if args.max:
        for fm in fraun_max.T:
            ax.fill_between(b_inplane / 1e-3, fm / 1e-6, color="tab:blue", alpha=0.5)
        ax.set_xlim((b_inplane.min() / 1e-3, None))
    fig.savefig(outpath + "_ic-vs-inplane.png")

if args.diode:
    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    plot(
        b_inplane / 1e-3,
        np.diff(np.abs(fraun_max)).squeeze() / 1e-9,
        "o-",
        xlabel="in-plane field (mT)",
        ylabel="$\Delta I_c$ (nA)",
        ax=ax,
        stamp=stamp,
    )
    fig.savefig(outpath + "_diode.png")

if i < 10:
    plt.show()
print("All plots written to output/")
