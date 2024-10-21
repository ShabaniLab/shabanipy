"""Fit the diode effect on the critical current vs. in-plane magnetic field.

This script fits Ic+(B||) and Ic-(B||) according to Eq. (1) of
https://arxiv.org/abs/2303.01902v2.
"""
import argparse
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from pandas import DataFrame, read_csv

from shabanipy.utils import write_metadata

# set up command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "datapath",
    help="path to .csv file containing columns for Ic+, Ic-, and B//",
)
parser.add_argument(
    "--bcol",
    help=(
        "name of column containing B-field data; "
        "if None, the first column matching *[Ff]ield* is used; "
        "if no match found, the first column is used"
    ),
)
parser.add_argument(
    "--icp_col",
    default="ic+ from fit",
    help="name of column containing Ic+ data",
)
parser.add_argument(
    "--icm_col",
    default="ic- from fit",
    help="name of column containing Ic- data",
)
parser.add_argument(
    "--icp_err_col",
    help="name of column containing Ic+ 1σ uncertainty",
)
parser.add_argument(
    "--icm_err_col",
    help="name of column containing Ic- 1σ uncertainty",
)
parser.add_argument(
    "--bmax",
    default=0.15,
    type=float,
    help="limit the fit to B-fields within +-bmax (T)",
)
parser.add_argument(
    "--bmask",
    type=float,
    help=(
        "mask points with B-fields within +-bmask (T), "
        "e.g. to mask zero-field anomalies"
    ),
)
parser.add_argument(
    "--invert_current",
    action="store_true",
    help="invert current polarity to conform to sign convention (Î x B∥)·growth > 0",
)
parser.add_argument(
    "--remove-zero-field-offset",
    action="store_true",
    help="Symmetrically shift Ic±(B∥) so that Ic+(0) = |Ic-(0)|",
)
# TODO alternative is to symmetrize data as in Alex's jupyter notebook
parser.add_argument(
    "--quiet",
    default=False,
    action="store_true",
    help="do not show plots and suppress console output",
)
args = parser.parse_args()

df = read_csv(args.datapath)
# find bfield column name
if args.bcol is not None:
    bcol = args.bcol
else:
    try:
        bcol = next(c for c in df.columns if "field" in c.lower())
    except StopIteration:
        bcol = df.columns[0]
        warn(f"Can't find field column. Assuming the first column ({bcol}) is field.")
# conform to sign convention (Î x B∥)·growth > 0
if args.invert_current:
    df[[args.icp_col, args.icm_col]] = df[[args.icm_col, args.icp_col]]
    df[[args.icp_err_col, args.icm_err_col]] = df[[args.icm_err_col, args.icp_err_col]]
# remove zero-field offset
if args.remove_zero_field_offset:
    (offset,) = (
        df[df[bcol] == 0][[args.icm_col, args.icp_col]].abs().diff(axis=1).iloc[:, -1]
    )
    df[[args.icp_col, args.icm_col]] -= offset / 2
# limit field range
if args.bmax is not None:
    mask = (-args.bmax <= df[bcol]) & (df[bcol] <= args.bmax)
    df = df[mask]
# mask zero-field anomalies
if args.bmask is not None:
    mask = (-args.bmask <= df[bcol]) & (df[bcol] <= args.bmask)
    df_masked = df[mask]
    df = df[~mask]
# extract data
bfield = df[bcol].values
icp = np.abs(df[args.icp_col].values)
icm = np.abs(df[args.icm_col].values)


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
outdir = Path(args.datapath).parent / Path(__file__).stem
outdir.mkdir(exist_ok=True, parents=True)
print(f"Output directory: {outdir}")
outpath = str(outdir / Path(args.datapath).stem) + "_fit"

# fit and plot
# TODO use errorbars as weights in fit
result = model.fit(np.concatenate((icp, icm)), x=bfield)
if not args.quiet:
    print(result.fit_report())
print(result.fit_report(), file=open(outpath + ".txt", "w"))
if result.params["b"].value < 0:
    warn("best fit b < 0 but b = (g* μ_B / 4 E_T)^2 > 0")

n = 100
bfield_smooth = np.linspace(bfield.min(), bfield.max(), n)
fit = model.eval(result.params, x=bfield_smooth)
plt.style.use(["fullscreen13"])
plt.errorbar(
    bfield / 1e-3,
    icp / 1e-6,
    yerr=df[args.icp_err_col] / 1e-6 if args.icp_err_col is not None else None,
    label="$I_{c+}$ data",
    marker="o",
    color="tab:blue",
    linewidth=0,
    elinewidth=1,
)
plt.errorbar(
    bfield / 1e-3,
    icm / 1e-6,
    yerr=df[args.icm_err_col] / 1e-6 if args.icm_err_col is not None else None,
    label="$I_{c-}$ data",
    marker="o",
    color="tab:orange",
    linewidth=0,
    elinewidth=1,
)
if args.bmask is not None:
    plt.errorbar(
        df_masked[bcol] / 1e-3,
        df_masked[args.icp_col].abs() / 1e-6,
        yerr=df_masked[args.icp_err_col] / 1e-6
        if args.icp_err_col is not None
        else None,
        label="excluded from fit",
        marker="o",
        markeredgecolor="tab:blue",
        markerfacecolor="white",
        linewidth=0,
        elinewidth=1,
    )
    plt.errorbar(
        df_masked[bcol] / 1e-3,
        df_masked[args.icm_col].abs() / 1e-6,
        yerr=df_masked[args.icm_err_col] / 1e-6
        if args.icm_err_col is not None
        else None,
        label="excluded from fit",
        marker="o",
        markeredgecolor="tab:orange",
        markerfacecolor="white",
        linewidth=0,
        elinewidth=1,
    )
plt.plot(bfield_smooth / 1e-3, fit[:n] / 1e-6, label="$I_{c+}$ fit", color="tab:blue")
plt.plot(bfield_smooth / 1e-3, fit[n:] / 1e-6, label="$I_{c-}$ fit", color="tab:orange")
plt.xlabel("in-plane field (mT)")
plt.ylabel("critical current (μA)")
plt.legend()
plt.savefig(outpath + ".png")
write_metadata(outpath + "_metadata.txt", args=args)
df = DataFrame(
    {
        **{k: [v] for k, v in result.best_values.items()},
        **{f"{k}_err": [v.std_dev] for k, v in result.uvars.items()},
        **{k: [result.summary()[k]] for k in ("ndata", "chisqr", "redchi", "rsquared")},
        **{k: [v] for k, v in args.__dict__.items() if k not in ("no_show",)},
    }
)
df.to_csv(outpath + ".csv", index=False)
if not args.quiet:
    plt.show()
