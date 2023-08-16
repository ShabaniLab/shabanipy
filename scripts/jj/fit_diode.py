"""Fit the diode effect on the critical current vs. in-plane magnetic field.

This script fits Ic+(B||) and Ic-(B||) according to Eq. (1) of
https://arxiv.org/abs/2303.01902v2.
"""
import argparse
from pathlib import Path
from pprint import pprint
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from pandas import read_csv

# set up command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "datapath",
    help="path to .csv file containing columns 'ic+', 'ic-', and '*[Ff]ield*'"
    "for positive & negative critical current vs. field",
)
parser.add_argument(
    "--bmax",
    "-b",
    type=float,
    help="limit the fit to magnetic fields within +-bmax",
)
# TODO add data symmetrization option as in Alex's jupyter notebook
args = parser.parse_args()

# extract data
df = read_csv(args.datapath)
icp = np.abs(df["ic+"].values)
icm = np.abs(df["ic-"].values)
# field column name might vary
colname = [c for c in df.columns if "field" in c.lower()]
if len(colname) == 1:
    bfield = df[colname[0]].values
else:
    pprint({i: colname for i, colname in enumerate(df.columns)})
    colindex = int(
        input(
            f"Which column has the field data? [select index 0-{len(df.columns) - 1}]: "
        )
    )
    bfield = df.iloc[:, colindex].values

# limit field range
if args.bmax is not None:
    mask = (-args.bmax <= bfield) & (bfield <= args.bmax)
    icm = icm[mask]
    icp = icp[mask]
    bfield = bfield[mask]

# swap Ic+ and Ic- to match sign convention that Ic+ > Ic- for B > 0
# n.b. assume data is sorted by field
if icp[-1] < icm[-1]:
    icp, icm = icm, icp


# build model from https://arxiv.org/abs/2303.01902v2 Eq. (1)
def icp_model(x, imax, b, c, bstar):
    """Positive critical current, Ic+."""
    return imax * (1 - b * (1 + c * np.sign(x - bstar)) * (x - bstar) ** 2)


def icm_model(x, imax, b, c, bstar):
    """Negative critical current, Ic-."""
    return imax * (1 - b * (1 - c * np.sign(x + bstar)) * (x + bstar) ** 2)


def icpm_model(x, imax, b, c, bstar):
    """Positive and negative critical current, Ic+-."""
    return np.concatenate([f(x, imax, b, c, bstar) for f in (icp_model, icm_model)])


model = Model(icpm_model)
model.set_param_hint("imax", value=np.mean([icp.max(), icm.max()]), vary=False)
model.set_param_hint("b", value=25)
model.set_param_hint("c", value=0)
model.set_param_hint("bstar", value=0)
params = model.make_params()

# set up output
outpath = Path(args.datapath)
print(f"Output directory: {outpath.parent}")
outpath = str(outpath.parent / outpath.stem) + f"_fit-{args.bmax}"

# fit and plot
result = model.fit(np.concatenate((icp, icm)), x=bfield)
print(result.fit_report())
print(result.fit_report(), file=open(outpath + ".txt", "w"))
if result.params["b"].value < 0:
    warn("best fit b < 0 but b = (g* μ_B / 4 E_T)^2 > 0")
if result.params["c"].value < 0:
    warn("best fit c < 0 but c = k_so / k_F > 0 (k_so = αm*/hbar^2 > 0)")
if result.params["bstar"].value < 0:
    warn("best fit B* < 0 but B* > 0 by definition")

n = 100
bfield_smooth = np.linspace(bfield.min(), bfield.max(), n)
fit = model.eval(result.params, x=bfield_smooth)
plt.style.use(["fullscreen13"])
plt.plot(bfield / 1e-3, icp / 1e-6, "o", label="$I_{c+}$ data", color="tab:blue")
plt.plot(bfield / 1e-3, icm / 1e-6, "o", label="$I_{c-}$ data", color="tab:orange")
plt.plot(bfield_smooth / 1e-3, fit[:n] / 1e-6, label="$I_{c+}$ fit", color="tab:blue")
plt.plot(bfield_smooth / 1e-3, fit[n:] / 1e-6, label="$I_{c-}$ fit", color="tab:orange")
plt.xlabel("in-plane field (mT)")
plt.ylabel("critical current (μA)")
plt.legend()
plt.savefig(outpath + ".png")
plt.show()
