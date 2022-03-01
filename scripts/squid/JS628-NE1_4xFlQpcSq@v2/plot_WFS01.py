"""Some data from vector9 cooldown WFS01."""

from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from shabanipy.plotting import jy_pink, plot_labberdata

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)
print(f"Output will be saved to {OUTDIR}/")


def savefig(fig, filename):
    fig.savefig((OUTDIR / filename).with_suffix(".png"), format="png", dpi=400)


jy_pink.register()
plot_labberdata = partial(plot_labberdata, style=["fullscreen13", "jy_pink"])

LOCKIN = "VITracer - dR vs I curve"
DMM = "VITracer - VI curve"
BIAS = "Bias current"
BPERP = "VectorMagnet - Field X"

LOCKIN_LABEL = "dV/dI (Ω)"
BIAS_LABEL = r"$I_\mathrm{bias}$ (μA)"
BPERP_LABEL = "$B_\perp$ (mT)"

# fraunhofer
fig, _ = plot_labberdata(
    "2022/02/Data_0207/JS628-NE1_4xFlQpcSq@v2_E_WFS01-033.hdf5",
    x=BPERP,
    xlabel=BPERP_LABEL,
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=LOCKIN,
    zlabel=LOCKIN_LABEL,
    transform=lambda x, y, z: (x / 1e-3, y / 1e-6, np.abs(z)),
    vmin=0,
    vmax=200,
    xlim=(-3, 3),
    ylim=(-2, 2),
)
savefig(fig, "WFS01-033")
fig, _ = plot_labberdata(
    "2022/02/Data_0216/JS628-NE1_4xFlQpcSq@v2_E_WFS01-065.hdf5",
    x=BPERP,
    xlabel=BPERP_LABEL,
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=LOCKIN,
    zlabel=LOCKIN_LABEL,
    transform=lambda x, y, z: (x / 1e-3 + 0.9, y / 1e-6, np.abs(z)),
    vmin=0,
    vmax=200,
    xlim=(-3, 3),
    ylim=(-2, 2),
)
savefig(fig, "WFS01-065")

# gate vs. bias
fig, _ = plot_labberdata(
    "2022/02/Data_0216/JS628-NE1_4xFlQpcSq@v2_E_WFS01-067.hdf5",
    x="qpcJJ gate 32",
    xlabel="$V_{g2} (V)$",
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=LOCKIN,
    zlabel=LOCKIN_LABEL,
    transform=lambda x, y, z: (x, y / 1e-6, np.abs(z)),
    vmin=0,
    vmax=999,
)
savefig(fig, "WFS01-067")
fig, _ = plot_labberdata(
    "2022/02/Data_0206/JS628-NE1_4xFlQpcSq@v2_E_WFS01-005.hdf5",
    x="gate 32 - Source voltage",
    xlabel="$V_g (V)$",
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=DMM,
    zlabel="dV/dI (kΩ)",
    transform=lambda x, y, z: (
        x,
        y / 1e-6,
        savgol_filter(np.diff(z / 100) / np.diff(y), 3, 1) / 1e3,
    ),
    vmin=0,
    vmax=40,
    extend_min=False,
)
savefig(fig, "WFS01-005")
fig, _ = plot_labberdata(
    "2022/02/Data_0227/JS628-NE1_4xFlQpcSq@v2_E_WFS01-128.hdf5",
    x="qpcJJ gate 32",
    xlabel="$V_g (V)$",
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=LOCKIN,
    zlabel=LOCKIN_LABEL,
    transform=lambda x, y, z: (x, y / 1e-6, np.abs(z)),
    vmin=0,
    vmax=800,
)
savefig(fig, "WFS01-128")


plt.show()
