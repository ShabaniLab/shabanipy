"Plot/animate shift of CPR with fluxline current from scan WFSBHE01-071."

from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh

from shabanipy.jj.utils import extract_switching_current
from shabanipy.labber import LabberData, get_data_dir
from shabanipy.plotting import jy_pink, plot, plot2d

jy_pink.register()
plt.style.use(["fullscreen13", "jy_pink"])

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)
print(f"Output will be saved to {str(OUTDIR)}/")

# from vector9 fridge
FILENAME = "JS602-SE1_4xFlQpcSq-v1_N_WFSBHE01-071"
PATH = get_data_dir() / f"2021/10/Data_1030/{FILENAME}.hdf5"
with LabberData(PATH) as f:
    bfield = f.get_data("Vector Magnet - Field X")
    ibias, dc_volts = f.get_data("VITracer - VI curve", get_x=True)
    dc_volts /= 100  # amplifier gain 100x
    iflux = f.get_data("circleFL 6 - Source current")

# plot extracted switching current for a few flux current values
ic = extract_switching_current(ibias, dc_volts, threshold=3.5e-5)
fig, ax = plot(
    np.unique(bfield) / 1e-6,
    ic[::2].T / 1e-6,
    label=[f"{int(i / 1e-6)} μA" for i in np.unique(iflux[::2])],
    xlabel="Vector Magnet Field (μT)",
    ylabel="Current Bias (μA)",
)
fig.savefig(OUTDIR / f"{FILENAME}_Ic.png")

# broadcast scalar data over vectorial data to have same shape
iflux, bfield, ibias, dc_volts = np.broadcast_arrays(
    iflux[..., np.newaxis], bfield[..., np.newaxis], ibias, dc_volts
)

# plot the first CPR
fig, ax = plot2d(
    bfield[0] / 1e-6,
    ibias[0] / 1e-6,
    np.diff(dc_volts[0]) / np.diff(ibias[0]),
    xlabel="$B_\perp^\mathrm{ext}$ (μT)",
    ylabel="$I_\\mathrm{bias}$ (μA)",
    zlabel="dV/dI (Ω)",
    vmin=0,
    vmax=500,
    extend_min=False,
)
qmesh = ax.get_children()[0]
assert isinstance(qmesh, QuadMesh), "Violated assumption that first child is QuadMesh"
line = ax.axvline(200, color="black", linestyle="--")
text = ax.text(
    220,
    0,
    r"$I_\mathrm{flux} = " + f"{int(round(iflux[0, 0, 0])) / 1e-6}$ μA",
    color="white",
    size=32,
    ha="center",
    va="center",
)

# define how to update each frame of the animation
def update(i):
    qmesh.set_array(np.diff(dc_volts[i]) / np.diff(ibias[i]))
    text.set_text(r"$I_\mathrm{flux} = " + f"{int(round(iflux[i, 0, 0] / 1e-6))}$ μA")
    return qmesh, line, text


# save the individual frames first
for i in range(iflux.shape[0]):
    update(i)
    fig.savefig(
        OUTDIR / f"cpr-shift-with-fluxline_{int(round(iflux[i, 0, 0] / 1e-6))}uA.png"
    )

# create and save the animation
ani = animation.FuncAnimation(
    fig,
    update,
    # repeat frames to simulate pause
    frames=[0, 0]
    + [i for i in range(iflux.shape[0])]
    + [iflux.shape[0] - 1]
    + [i for i in reversed(range(iflux.shape[0]))],
    interval=500,
    blit=True,
)
ani.save(OUTDIR / "cpr-shift-with-fluxline.gif", dpi=100)

plt.show()
