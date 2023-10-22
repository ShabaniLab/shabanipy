"""Plot fraunhofer center and optionally fit for alignment.

Expected column names: VARIABLE, center+, center-

Note VARIABLE is the independent variable; the actual name of that column doesn't
matter, but it must be the first column.

Both + and - data need not be present; e.g. if center- is missing, that's ok.
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

csv = read_csv(args.csv_path)
branch = ""
branch += "+" if "center+" in csv.columns else ""
branch += "-" if "center-" in csv.columns else ""
variable_name = csv.columns[0]
variable = csv[variable_name].to_numpy()
center = csv[[f"center{sign}" for sign in branch]].to_numpy()

label = ("$+$branch", "$-$branch") if branch == "+-" else f"${branch}$branch"
fig, ax = plot(
    variable,
    center / 1e-3,
    ".-",
    xlabel=variable_name,
    ylabel="fraunhofer center (mT)",
    label=label,
    stamp=Path(args.csv_path).name,
)
if args.align:
    for c in center.T:
        mask = np.isfinite(c)
        m, b = np.polyfit(variable[mask], c[mask], 1)
        ax.plot(
            variable,
            (m * variable + b) / 1e-3,
            label=f"arcsin$(y/x)$ = {round(np.degrees(np.arcsin(m)), 3)} deg",
        )
ax.legend()
fig.savefig(str(outpath) + "_center.png")

plt.show()
