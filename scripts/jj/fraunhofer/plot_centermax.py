"""Plot data extracted by centermax.py.

Expected column names are: ic+, ic-, center+, center-, VARIABLE
VARIABLE is the independent variable; the actual name of that column doesn't matter.
Both + and - data need not be present; e.g. if ic- and center- are missing, that's ok.
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
    "--align",
    "-a",
    default=False,
    action="store_true",
    help="fit fraunhofer center for field alignment angle",
)
args = parser.parse_args()

outdir = Path(args.csv_path).parent
print(f"Output directory: {outdir}")
outpath = outdir / Path(args.csv_path).stem
plt.style.use("fullscreen13")
stamp = Path(args.csv_path).stem

csv = read_csv(args.csv_path)
branch = ""
branch += "+" if "ic+" in csv.columns else ""
branch += "-" if "ic-" in csv.columns else ""
(variable_name,) = set(csv.columns) - set(
    ["datafile", "ic+", "ic-", "center+", "center-"]
)
variable = csv[variable_name].to_numpy()
fraun_center = csv[[f"center{sign}" for sign in reversed(branch)]].to_numpy()
fraun_max = csv[[f"ic{sign}" for sign in reversed(branch)]].to_numpy()

ic_label = ("$I_{c-}$", "$I_{c+}$") if branch == "+-" else f"$I_{{c{branch}}}$"
fig, ax = plot(
    variable,
    fraun_center / 1e-3,
    "o-",
    xlabel=variable_name,
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
fig.savefig(str(outpath) + "_center.png")

fig, ax = plot(
    variable,
    fraun_max / 1e-6,
    "o-",
    color="tab:blue",
    xlabel=variable_name,
    ylabel="critical current (μA)",
    stamp=stamp,
)
for fm in fraun_max.T:
    ax.fill_between(variable, fm / 1e-6, color="tab:blue", alpha=0.5)
ax.set_xlim((variable.min(), variable.max()))
fig.savefig(str(outpath) + "_max.png")

if branch == "+-":
    fig, ax = plot(
        variable,
        np.abs(fraun_max) / 1e-6,
        "o-",
        xlabel=variable_name,
        ylabel="critical current (μA)",
        label=ic_label,
        stamp=stamp,
    )
    ax.fill_between(variable, *np.abs(fraun_max.T / 1e-6), color="tab:gray", alpha=0.5)
    fig.savefig(str(outpath) + "_absmax.png")

    fig, ax = plt.subplots()
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    plot(
        variable,
        np.diff(np.abs(fraun_max)).squeeze() / 1e-9,
        "o-",
        xlabel=variable_name,
        ylabel="$\Delta I_c$ (nA)",
        ax=ax,
        stamp=stamp,
    )
    fig.savefig(str(outpath) + "_deltamax.png")

plt.show()
