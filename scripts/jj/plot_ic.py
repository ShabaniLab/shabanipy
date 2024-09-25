"""Plot critical current as a function of some variable.

Expected column names: VARIABLE, ic+, ic-

Note VARIABLE is the independent variable; the actual name of that column doesn't
matter, but it must be the first column.

Both + and - data need not be present; e.g. if ic- is missing, that's ok.
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
parser.add_argument(
    "--no-errorbars", help="don't plot errorbars", default=False, action="store_true"
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
print(f"Output directory: {outdir}")
outprefix = outdir / Path(args.csv_path).stem
plt.style.use("fullscreen13")

csv = read_csv(args.csv_path)
branch = ""
branch += "+" if "ic+" in csv.columns else ""
branch += "-" if "ic-" in csv.columns else ""
xname = csv.columns[0]
if args.xmin:
    csv = csv.loc[csv[xname] >= args.xmin]
if args.xmax:
    csv = csv.loc[csv[xname] <= args.xmax]
if args.xmin or args.xmax:
    outprefix = Path(str(outprefix) + f"_{args.xmin}-{args.xmax}")
x = csv[xname].to_numpy()
ic = csv[[f"ic{sign} from fit" for sign in branch]].to_numpy()
if np.all(f"rmse{sign}" in csv for sign in branch) and not args.no_errorbars:
    errorbars = True
    rmse = csv[[f"rmse{sign}" for sign in branch]].to_numpy()
else:
    errorbars = False

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
if errorbars:
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
    if errorbars:
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
    if errorbars:
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
    if errorbars:
        sigma_err = delta_err
        eta_err = np.abs(eta) * np.sqrt(
            (delta_err / delta) ** 2 + (sigma_err / sigma) ** 2
        )
        ax.errorbar(
            x, eta * 100, yerr=eta_err * 100, color="tab:blue", lw=0, elinewidth=2
        )
    fig.savefig(str(outprefix) + "_ic-diff-norm.png")

if not args.quiet:
    plt.show()
