"""Some data from vector9 cooldowns WFS01-02."""

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
plt.style.use(["fullscreen13", "jy_pink"])

LOCKIN = "VITracer - dR vs I curve"
DMM = "VITracer - VI curve"
BIAS = "Bias current"
BPERP = "VectorMagnet - Field X"
BPRLL = "VectorMagnet - Field Y"
ARCFL = "arcFL 20 - Source current"

LOCKIN_LABEL = "dV/dI (Ω)"
DMM_LABEL = "$\Delta V / \Delta I$ (Ω)"
BIAS_LABEL = r"$I_\mathrm{bias}$ (μA)"
BPERP_LABEL = "$B_\perp$ (mT)"
BPRLL_LABEL = "$B_\parallel$ (T)"

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
    stamp="WFS01-033",
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
    stamp="WFS01-065",
)
savefig(fig, "WFS01-065")

# gate
fig, _ = plot_labberdata(
    "2022/02/Data_0206/JS628-NE1_4xFlQpcSq@v2_E_WFS01-005.hdf5",
    x="gate 32 - Source voltage",
    xlabel="$V_g (V)$",
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=DMM,
    zlabel=DMM_LABEL,
    transform=lambda x, y, z: (
        x,
        y / 1e-6,
        savgol_filter(np.gradient(z / 100, axis=-1) / np.gradient(y, axis=-1), 3, 1),
    ),
    vmin=0,
    vmax=400,
    extend_min=False,
    stamp="WFS01-005",
)
savefig(fig, "WFS01-005")
with plt.rc_context({"image.cmap": "viridis", "figure.figsize": (8, 10)}):
    fig, _ = plot_labberdata(
        "2022/02/Data_0212/JS628-NE1_4xFlQpcSq@v2_E_WFS01-051.hdf5",
        x=BIAS,
        xlabel=BIAS_LABEL,
        y="gate 35",
        ylabel="$V_g$ (V)",
        z=LOCKIN,
        zlabel=LOCKIN_LABEL,
        transform=lambda x, y, z: (x / 1e-6, y, np.abs(z)),
        vmin=0,
        vmax=100,
        ylim=(-7, None),
        stamp="WFS01-051",
    )
    savefig(fig, "WFS01-051")
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
    stamp="WFS01-067",
)
savefig(fig, "WFS01-067")
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
    stamp="WFS01-128",
)
savefig(fig, "WFS01-128")


# in-plane
with plt.rc_context({"image.cmap": "viridis", "figure.figsize": (8, 10)}):
    fig, _ = plot_labberdata(
        "2022/02/Data_0209/JS628-NE1_4xFlQpcSq@v2_E_WFS01-043.hdf5",
        x=BIAS,
        xlabel=BIAS_LABEL,
        y=BPRLL,
        ylabel=BPRLL_LABEL,
        z=LOCKIN,
        zlabel=LOCKIN_LABEL,
        transform=lambda x, y, z: (x / 1e-6, y, np.abs(z)),
        vmin=0,
        vmax=100,
        stamp="WFS01-043",
    )
    savefig(fig, "WFS01-043")

# cpr
fig, _ = plot_labberdata(
    "2022/03/Data_0302/JS628-NE1_4xFlQpcSq@v2_E_WFS02-013.hdf5",
    x=BPERP,
    xlabel="$B_\perp$ (μT)",
    y=BIAS,
    ylabel=BIAS_LABEL,
    z=LOCKIN,
    zlabel=LOCKIN_LABEL,
    transform=lambda x, y, z: (x / 1e-6, y / 1e-6, np.abs(z)),
    vmin=0,
    vmax=100,
    xlim=(-20, None),
    ylim=(None, 0),
    stamp="WFS02-013",
)
savefig(fig, "WFS02-013")

# fluxline
for value in (-40, -20, 0, 20, 40):
    fig, ax = plot_labberdata(
        "2022/03/Data_0307/JS628-NE1_4xFlQpcSq@v2_E_WFSBHE02-032.hdf5",
        x=BPERP,
        xlabel="$B_\perp$ (μT)",
        y=BIAS,
        ylabel=BIAS_LABEL,
        z=DMM,
        zlabel=DMM_LABEL,
        transform=lambda x, y, z: (
            x / 1e-6,
            y / 1e-6,
            savgol_filter(
                np.gradient(z / 100, axis=-1) / np.gradient(y, axis=-1), 3, 1
            ),
        ),
        filters={ARCFL: value * 1e-6},
        vmin=0,
        vmax=150,
        stamp=f"WFSBHE02-032_{value}μA",
    )
    ax.axvline(20, color="k", linestyle=":")
    ax.text(
        0.55,
        0.5,
        f"$I_\mathrm{{flux}} = {value}$ μA",
        transform=ax.transAxes,
        color="w",
        size="xx-large",
    )
    savefig(fig, f"WFSBHE02-032_{value}uA")

plt.show()
