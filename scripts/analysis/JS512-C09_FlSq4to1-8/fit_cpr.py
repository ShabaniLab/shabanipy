"""Fit current-phase relations to extract transparency.

Wafer:      JS512
Piece:      C09
Layout:     FlSq4to1-8.gds
Devices:    4to1, 4to2, 4to7, 4to8
Cooldown:   WFS-01, WFS02
"""
from h5py import File
import numpy as np
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters

from shabanipy.constants import VECTOR10_AMPS_PER_TESLA_X, VECTOR9_AMPS_PER_TESLA_X
from shabanipy.plotting import plot2d, jy_pink, stamp, plot
from shabanipy.labber import get_data_dir, LabberData
from shabanipy.dvdi import extract_switching_current

plt.style.use(["presentation", "jy_pink"])

FRIDGE = "vector10"
SAMPLE = "JS512-C09_FlSq4to1-8"
DEVICE = "FlSq4to2"
CD_SCAN = "WFS02_125"
PATH = get_data_dir() / f"{FRIDGE}/2021/08/Data_0821/{SAMPLE}_{DEVICE}_{CD_SCAN}.hdf5"
AMPS_PER_T = vars()[f"{FRIDGE.upper()}_AMPS_PER_TESLA_X"]

with LabberData(PATH) as f:
    bfield = f.get_data("X magnet - Source current") / AMPS_PER_T
    ibias, lockin = f.get_data("vicurve - dR vs I curve", get_x=True)
    dvdi = np.abs(lockin)

fig, ax = plot2d(
    *np.broadcast_arrays(bfield[..., np.newaxis] / 1e-3, ibias / 1e-6, dvdi),
    xlabel="x coil field (mT)",
    ylabel="dc bias (μA)",
    zlabel="dV/dI (Ω)",
)
stamp(ax, CD_SCAN)

ic_n, ic_p = extract_switching_current(ibias, dvdi, side='both', threshold=100)
plot(bfield / 1e-3, ic_p / 1e-6, ax=ax, color='w', ls=':')
plot(bfield / 1e-3, ic_n / 1e-6, ax=ax, color='w', ls=':')

params = Parameters()


plt.show()
