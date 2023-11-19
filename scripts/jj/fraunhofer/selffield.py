"""Plot the fraunhofer tilt due to self-field focusing.

Data input from centermax.py analysis in shabanipy."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "path",
    help="path to .csv file containing fraunhofer center and max in columns: center+, center-, ic+, ic-",
)
parser.add_argument("--bmin", help="minimum B-field to use", type=float)
parser.add_argument("--bmax", help="maximum B-field to use", type=float)
args = parser.parse_args()

outdir = Path(args.path).parent
print(f"Output directory: {outdir}")

df = read_csv(args.path)
field_col = [
    c
    for c in set(df.columns) - {"ic-", "ic+", "center-", "center+"}
    if "field" in c.lower()
]
(field_col,) = field_col
im, ip, cm, cp, b = (
    df["ic- from fit"],
    df["ic+ from fit"],
    df["center-"],
    df["center+"],
    df[field_col],
)
iavg = np.mean([np.abs(im), ip], axis=0)
cdiff = cp - cm
tilt = (ip - im) / (cp - cm)

# filter by magnetic field
mask = [True] * len(b)
if args.bmin is not None:
    mask = mask & (args.bmin <= b)
if args.bmax is not None:
    mask = mask * (b <= args.bmax)
iavg = iavg[mask]
cdiff = cdiff[mask]
tilt = tilt[mask]

plt.style.use("fullscreen13")

fig, ax = plt.subplots()
ax.set_xlabel("critical current (μA)")
ax.set_ylabel("fraunhofer center difference (μT)")
ax.plot(iavg / 1e-6, cdiff / 1e-6, "o")
if args.bmin is not None or args.bmax is not None:
    bstr = f"_{args.bmin}-{args.bmax}"
else:
    bstr = ""
fig.savefig(str(outdir / Path(args.path).stem) + f"_selffield-cdiff{bstr}.png")

fig, ax = plt.subplots()
ax.set_xlabel("critical current (μA)")
ax.set_ylabel("fraunhofer tilt (A/T)")
ax.plot(iavg / 1e-6, tilt, "o")
if args.bmin is not None or args.bmax is not None:
    bstr = f"-{args.bmin}-{args.bmax}"
else:
    bstr = ""
fig.savefig(str(outdir / Path(args.path).stem) + f"_selffield-tilt{bstr}.png")

plt.show()
