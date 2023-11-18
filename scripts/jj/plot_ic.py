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
parser.add_argument("--xmin", help="minimum value of x axis", type=float)
parser.add_argument("--xmax", help="maximum value of x axis", type=float)
args = parser.parse_args()

outdir = Path(args.csv_path).parent
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

stamp = Path(args.csv_path).name
fig, ax = plot(
    x,
    ic / 1e-6,
    ".-",
    color="tab:blue",
    xlabel=xname,
    ylabel="critical current (μA)",
    stamp=stamp,
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
    ax.fill_between(x, *np.abs(ic.T / 1e-6), color="tab:gray", alpha=0.5)
    fig.savefig(str(outprefix) + "_ic-mag.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    plot(
        x,
        -np.diff(np.abs(ic)).squeeze() / 1e-9,
        ".-",
        xlabel=xname,
        ylabel="$\Delta I_c$ (nA)",
        ax=ax,
        stamp=stamp,
    )
    fig.savefig(str(outprefix) + "_ic-diff.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    plot(
        x,
        -np.diff(np.abs(ic)).squeeze() / np.sum(np.abs(ic), axis=-1),
        ".-",
        xlabel=xname,
        ylabel="$\Delta I_c / \Sigma I_c$",
        ax=ax,
        stamp=stamp,
    )
    fig.savefig(str(outprefix) + "_ic-diff-norm.png")

plt.show()
