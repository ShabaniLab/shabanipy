"""Plot critical current as a function of some variable.

The independent variable is assumed to be in the first column.
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv

from shabanipy.utils import plot

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "csv_path", help="path to .csv file output by centermax.py, relative to this script"
)
parser.add_argument("--icp_col", help="Name of column containing Ic+ data")
parser.add_argument("--icm_col", help="Name of column containing Ic- data")
parser.add_argument(
    "--icp_err_col", help="Name of column containing Ic+ uncertainty (for errorbars)"
)
parser.add_argument(
    "--icm_err_col", help="Name of column containing Ic- uncertainty (for errorbars)"
)
parser.add_argument("--xmin", help="minimum value of x axis", type=float)
parser.add_argument("--xmax", help="maximum value of x axis", type=float)
parser.add_argument(
    "--quiet",
    default=False,
    action="store_true",
    help="do not show plots and suppress console output",
)
args = parser.parse_args()

outdir = Path(args.csv_path).parent / Path(__file__).stem
outdir.mkdir(exist_ok=True)
outprefix = outdir / Path(args.csv_path).stem
plt.style.use("fullscreen13")

csv = read_csv(args.csv_path)
branch = ""
branch += "+" if args.icp_col is not None else ""
branch += "-" if args.icm_col is not None else ""
xname = csv.columns[0]
if args.xmin:
    csv = csv.loc[csv[xname] >= args.xmin]
if args.xmax:
    csv = csv.loc[csv[xname] <= args.xmax]
if args.xmin or args.xmax:
    outprefix = Path(str(outprefix) + f"_{args.xmin}-{args.xmax}")
x = csv[xname].to_numpy()
ic_cols = list(filter(lambda _: _ is not None, [args.icp_col, args.icm_col]))
if len(ic_cols) == 0:
    raise ValueError("One of --icp_col or --icm_col must be specified")
ic = csv[ic_cols].to_numpy()
err_cols = list(filter(lambda _: _ is not None, [args.icp_err_col, args.icm_err_col]))
if len(err_cols) > 0:
    rmse = csv[err_cols].to_numpy()
else:
    rmse = []

stamp = Path(*Path(args.csv_path).parts[-4:])
fig, ax = plot(
    x,
    ic / 1e-6,
    ".-",
    color="tab:blue",
    xlabel=xname,
    ylabel="critical current (μA)",
    stamp=stamp,
)
if len(rmse) > 0:
    for ic_, rmse_ in zip(ic.T, rmse.T):
        ax.errorbar(
            x, ic_ / 1e-6, yerr=rmse_ / 1e-6, lw=0, elinewidth=2, color="tab:blue"
        )
for i in ic.T:
    ax.fill_between(x, i / 1e-6, color="tab:blue", alpha=0.5)
ax.set_xlim((x.min(), x.max()))
fig.savefig(str(outprefix) + "_ic.png")

if branch == "+-":
    label = ("$I_{c+}$", "$I_{c-}$")
    fig, ax = plot(
        x,
        np.abs(ic) / 1e-6,
        ".-",
        xlabel=xname,
        ylabel="critical current (μA)",
        label=label,
        stamp=stamp,
    )
    if len(rmse) > 0:
        for ic_, rmse_, color in zip(ic.T, rmse.T, ("tab:blue", "tab:orange")):
            ax.errorbar(
                x,
                np.abs(ic_) / 1e-6,
                yerr=rmse_ / 1e-6,
                color=color,
                lw=0,
                elinewidth=2,
            )
    ax.fill_between(x, *np.abs(ic.T / 1e-6), color="tab:gray", alpha=0.5)
    fig.savefig(str(outprefix) + "_ic-mag.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    delta = -np.diff(np.abs(ic)).squeeze()
    plot(
        x,
        delta / 1e-9,
        ".-",
        xlabel=xname,
        ylabel="$\Delta I_c$ (nA)",
        ax=ax,
        stamp=stamp,
    )
    if len(rmse) > 0:
        delta_err = np.sqrt(np.sum(rmse**2, axis=-1))
        ax.errorbar(
            x, delta / 1e-9, yerr=delta_err / 1e-9, color="tab:blue", lw=0, elinewidth=2
        )
    fig.savefig(str(outprefix) + "_ic-diff.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    sigma = np.sum(np.abs(ic), axis=-1)
    eta = delta / sigma
    plot(
        x,
        eta * 100,
        ".-",
        xlabel=xname,
        ylabel="$\Delta I_c / \Sigma I_c$ (%)",
        ax=ax,
        stamp=stamp,
    )
    if len(rmse) > 0:
        sigma_err = delta_err
        eta_err = np.sqrt((delta_err / sigma) ** 2 + (eta * sigma_err / sigma) ** 2)
        ax.errorbar(
            x, eta * 100, yerr=eta_err * 100, color="tab:blue", lw=0, elinewidth=2
        )
    fig.savefig(str(outprefix) + "_ic-diff-norm.png")

print(f"Plotted Ic: {outdir}")
if not args.quiet:
    plt.show()
