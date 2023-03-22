"""Calculate growth rate from a linear fit of thickness vs. time.

The input to this script is a tab-separated file with deposition time, measured
thickness, and thickness uncertainty in the following format:

```
time	thickness	sigma
500	10.0	4.0
1000	20.0	5.0
...
```
"""

import argparse

import numpy as np
from lmfit.models import LinearModel
from matplotlib import pyplot as plt
from pandas import read_csv

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "path", help="path to the .txt histogram data file exported from NMI AFM software"
)
parser.add_argument(
    "--force-zero",
    "-z",
    default=False,
    action="store_true",
    help="force the fit to intersect the origin",
)
parser.add_argument(
    "--error-bars",
    "-e",
    default=False,
    action="store_true",
    help="use error bars in the fit",
)
args = parser.parse_args()
df = read_csv(args.path, sep="\t")
time, thickness, sigma = df.T.to_numpy()

plt.style.use("fullscreen13")
if args.error_bars:
    plt.errorbar(time, thickness, yerr=sigma, fmt="o", capsize=10)
else:
    plt.plot(time, thickness, "o", markersize=10)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.xlim(0, None)
plt.ylim(0, None)

model = LinearModel()
if args.force_zero:
    model.set_param_hint("intercept", value=0, vary=False)
result = model.fit(thickness, x=time, weights=1 / sigma if args.error_bars else None)
time_extrap = np.insert(time, 0, 0)
plt.plot(time_extrap, result.eval(x=time_extrap))
slope, intercept = result.params.valuesdict().values()
plt.title(f"{slope=:.5f}, {intercept=:.1f}")
plt.savefig("growthrate.png")
plt.show()
