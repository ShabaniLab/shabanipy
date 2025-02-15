"""Extract background fraunhofer modulation from SQUID oscillations."""

import argparse
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from shabanipy.utils import jy_pink, write_metadata

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "csv_path",
    help="path to .csv file containing (x1, ..., xn, bfield, ic+, ic-, ...) data columns in that order",
)
parser.add_argument(
    "--method",
    "-m",
    choices=["fit", "fitrolling"],
    default="fit",
    help=(
        "how to handle the low-frequency fraunhofer modulation. "
        "`fit` to a quadratic; or"
        "`fitrolling` to fit the quadratic to a rolling average with window-length "
        "specified by command-line `--window-length`"
    ),
)
parser.add_argument(
    "--window-length",
    "-wlen",
    type=int,
    default=None,
    help="window length to use if --fraunhofer=fitrolling; "
    "should match the oscillation period as closely as possible",
)
args = parser.parse_args()
if args.method == "fitrolling" and args.window_length is None:
    raise ValueError("`--window-length` must be specified if `--method=fitrolling`")

jy_pink.register()
plt.style.use(["fullscreen13", "jy_pink"])
OUTDIR = Path(args.csv_path).parent / Path(__file__).stem
OUTDIR.mkdir(exist_ok=True)
print(f"Output directory: {OUTDIR}")
write_metadata(OUTDIR / "args.json", args)

csv = pd.read_csv(args.csv_path)
branch = ""
if "ic+" in csv:
    branch += "+"
if "ic-" in csv:
    branch += "-"
bfield_colidx = csv.columns.get_loc("ic+" if "+" in branch else "ic-") - 1
bfield_colname = csv.columns[bfield_colidx]
group_colnames = csv.columns[:bfield_colidx].to_list()


def extract_background(df, branch):
    bfield = df[bfield_colname]
    ic = df[f"ic{branch} rolling" if args.method == "fitrolling" else f"ic{branch}"]
    bfield_to_fit = bfield[ic.notna()]
    ic_to_fit = ic[ic.notna()]
    poly = np.polynomial.Polynomial.fit(bfield_to_fit, ic_to_fit, 2)
    df[f"ic{pm} background"] = poly(bfield)
    return df


for pm in branch:
    if args.method == "fitrolling":
        csv[f"ic{pm} rolling"] = csv.groupby(group_colnames)[f"ic{pm}"].transform(
            lambda x: x.rolling(args.window_length, center=True).mean()
        )
    csv[f"ic{pm} background"] = (
        csv.groupby(group_colnames)
        .apply(extract_background, pm, include_groups=False)
        .reset_index()[f"ic{pm} background"]
    )
csv.to_csv(OUTDIR / "data.csv")

# plot
for name, group in csv.groupby(group_colnames):
    bfield = group[bfield_colname]
    for pm in branch:
        fig, ax = plt.subplots()
        ax.set_title(f"{group_colnames}={name}", fontsize="xx-small")
        ax.plot(bfield, group[f"ic{pm}"], ".", label="data")
        if args.method == "fitrolling":
            ax.plot(bfield, group[f"ic{pm} rolling"], label="rolling average")
        ax.plot(bfield, group[f"ic{pm} background"], label="background fit")
        ax.legend()
        name_str = reduce(lambda x, y: f"{x}_{y}", name)
        fig.savefig(OUTDIR / f"{name_str}_ic{pm}_background.svg")

plt.show()
