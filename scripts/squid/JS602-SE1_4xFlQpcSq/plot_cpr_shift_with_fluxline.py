from pathlib import Path

import numpy as np
from matplotlib.pyplot import style

from shabanipy.jj.utils import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.plotting import jy_pink, plot, plot2d

style.use(["presentation", "jy_pink"])

outdir = Path("plots")
outdir.mkdir(exist_ok=True)

# from vector9 fridge
FILENAME = "JS602-SE1_4xFlQpcSq-v1_N_WFSBHE01-071"
path = get_data_dir() / f"2021/10/Data_1030/{FILENAME}.hdf5"
with LabberData(path) as f:
    bfield = f.get_data("Vector Magnet - Field X")
    ibias, dc_volts = f.get_data("VITracer - VI curve", get_x=True)
    dc_volts /= 100  # amplifier gain 100x
    iflux = f.get_data("circleFL 6 - Source current")

ic = extract_switching_current(ibias, dc_volts, threshold=3.5e-5)
fig, ax = plot(
    np.unique(bfield) / 1e-6,
    ic[::2].T / 1e-6,
    label=[f"{int(i / 1e-6)} μA" for i in np.unique(iflux[::2])],
    xlabel="Vector Magnet Field (μT)",
    ylabel="Current Bias (μA)",
)
#fig.show()
fig.savefig(outdir / f"{FILENAME}_Ic.png")

iflux, bfield, ibias, dc_volts = np.broadcast_arrays(
    iflux[..., np.newaxis], bfield[..., np.newaxis], ibias, dc_volts
)
for idx, i in enumerate(np.unique(iflux)):
    i_uA = int(i / 1e-6)
    fig, ax = plot2d(
        bfield[idx] / 1e-6,
        ibias[idx] / 1e-6,
        np.diff(dc_volts[idx]) / np.diff(ibias[idx]),
        xlabel="Vector Magnet Field (μT)",
        ylabel="Current Bias (μA)",
        zlabel="dV/dI (Ω)",
        title=f"fluxline current = {i_uA} μA",
        vmax=400,
    )
    #plot(np.unique(bfield) / 1e-6, ic[idx] / 1e-6, ax=ax)  # to view Ic extraction
    #fig.show()
    fig.savefig(outdir / f"{FILENAME}_iflux={i_uA}μA.png")
