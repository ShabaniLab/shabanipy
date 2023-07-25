"""Fit the diode effect on the switching current vs. in-plane field.

This script fits Ic+(B||) and Ic-(B||) according to Eq. (1) of
https://arxiv.org/abs/2303.01902v2.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit
from lmfit import Model

# set up command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "datapath", help="path to .csv file containing positive & negative switching current vs. field"
)
parser.add_argument(
    "--bmax",
    "-b",
    type=float,
    help="limit the fit to magnetic fields within +-bmax",
)
args = parser.parse_args()

# extract data
df = read_csv(args.datapath)
icp = np.abs(df['ic+'].values)
icm = np.abs(df['ic-'].values)
bfield = df['b_inplane'].values

# limit field range
if args.bmax is not None:
    mask = (-args.bmax <= bfield) & (bfield <= args.bmax)
    icm = icm[mask]
    icp = icp[mask]
    bfield = bfield[mask]

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
model.set_param_hint("c", value=-0.1)
model.set_param_hint("bstar", value=0.001)
params = model.make_params()

# fit and plot
result = model.fit(np.concatenate((icp, icm)), x=bfield)
print(result.fit_report())

n = 100
bfield_smooth = np.linspace(bfield.min(), bfield.max(), n)
fit = model.eval(result.params, x=bfield_smooth)

plt.scatter(bfield / 1e-3, icm / 1e-6, label='$I_{c-}$ data', color="tab:blue")
plt.scatter(bfield / 1e-3, icp / 1e-6, label='$I_{c+}$ data', color="tab:orange")
plt.plot(bfield_smooth / 1e-3, fit[n:] / 1e-6, label='$I_{c-}$ fit', color="tab:blue")
plt.plot(bfield_smooth / 1e-3, fit[:n] / 1e-6, label='$I_{c+}$ fit', color="tab:orange")

plt.xlabel('in-plane field (mT)')
plt.ylabel('critical current (μA)')
plt.legend()
plt.show()
