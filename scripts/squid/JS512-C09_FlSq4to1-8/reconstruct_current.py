"""Reconstruct critical current distribution from fraunhofers.

This script depends on data aggregated by ./aggregate.py and subsequent pre-processing
performed by ./summarize.py:
    $ python aggregate.py "fraunhofer" "fraunhofer without curvetracer"
    $ python summarize.py "fraunhofer" "fraunhofer without curvetracer" "fraunhofer (dmm)" "fraunhofer (dmm) without curvetracer"
"""
from pathlib import Path

import numpy as np
from h5py import File
from matplotlib.pyplot import style, subplots
from scipy.constants import physical_constants as phys_c

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
)
from shabanipy.jj.fraunhofer.utils import find_fraunhofer_center
from shabanipy.jj.utils import extract_switching_current
from shabanipy.plotting import jy_pink, plot, plot2d
from shabanipy.plotting.utils import stamp

jy_pink.register()
style.use(["presentation", "jy_pink"])
scans = {
    "WFS02_078": {"jj_width": 4e-6, "jj_length": 1e-6, "threshold": 50},
    "WFS02_073": {"jj_width": 1e-6, "jj_length": 1e-6, "threshold": 50},
    "WFS02_042": {"jj_width": 4e-6, "jj_length": 1e-6, "threshold": 50},
    "WFS02_026": {"jj_width": 2e-6, "jj_length": 1e-6, "threshold": 100},
    "WFS02_146": {"jj_width": 4e-6, "jj_length": 1e-6, "threshold": 50},
    "WFS02_143": {"jj_width": 7e-6, "jj_length": 1e-6, "threshold": 25},
    "WFS-01_026": {"jj_width": 4e-6, "jj_length": 1.5e-6, "threshold": 35},
    "WFS-01_072": {"jj_width": 8e-6, "jj_length": 1.5e-6, "threshold": 25},
}
outdir = Path("./plots/current-reconstruction/")
outdir.mkdir(parents=True, exist_ok=True)


def reconstruct_current(name, obj):
    scan = name.split("/")[-1].split("::")[-1]
    if scan not in scans.keys():
        return None
    print(scan, name)
    params = scans[scan]

    field = obj["x magnet field"]
    ibias = obj["dc current bias"]
    dvdi = np.abs(obj["lock-in (impedance)"])

    # extract fraunhofer from data
    ic = extract_switching_current(ibias, dvdi, threshold=params["threshold"])
    fig, ax = plot2d(field, ibias, dvdi)
    plot(
        np.unique(field),
        ic,
        ax=ax,
        title="Ic extraction",
        color="k",
        lw="0",
        marker=".",
        markersize=5,
    )
    ax.axis("off")
    stamp(ax, scan)
    fig.savefig(outdir / f"{scan}_ic-extraction.png")

    field = np.unique(field)

    # center fraunhofer
    argmax_ic = find_fraunhofer_center(field, ic)
    ic_n = extract_switching_current(
        ibias, dvdi, threshold=params["threshold"], side="negative"
    )
    argmax_ic_n = find_fraunhofer_center(field, np.abs(ic_n))
    field -= (argmax_ic + argmax_ic_n) / 2
    # field = recenter_fraunhofer(np.unique(field), ic) # if just centering at argmax(Ic+)
    fig, ax = plot(
        field / 1e-3,
        np.array([ic, ic_n]).T / 1e-6,
        xlabel="field (mT)",
        ylabel="critical current (μA)",
        title="fraunhofer centered",
    )
    ax.axvline(0, color="k")
    fig.savefig(outdir / f"{scan}_fraunhofer-centered.png")

    # reconstruct current distribution
    x, jx = extract_current_distribution(
        field,
        ic,
        f2k=2 * np.pi * params["jj_length"] / phys_c["mag. flux quantum"][0],
        jj_width=params["jj_width"],
        jj_points=400,
    )
    fig, ax = subplots()
    ax.fill_between(x / 1e-6, jx)
    ax.set_xlabel("position (μA)")
    ax.set_ylabel("critical current density (μA/μm)")
    ax.set_title(f"$\ell_\mathrm{{eff}}$ = {round(params['jj_length'] / 1e-6, 1)}μm")
    fig.savefig(outdir / f"{scan}_current-distribution.png")


if __name__ == "__main__":
    with File("./data_preprocessed.hdf5") as f:
        f.visititems(reconstruct_current)
