"""Studying diode model from arxiv:2303.01902 (Neda's diode paper)."""
import argparse
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def bstar_model(b, c, tau):
    return (1 - tau) ** (1 / 4) * (c / np.sqrt(b))


def icp_model(x, b, c, tau):
    """Positive critical current, Ic+."""
    bstar = bstar_model(b, c, tau)
    return 1 - b * (1 + c * np.sign(x - bstar)) * (x - bstar) ** 2


def icm_model(x, b, c, tau):
    """Negative critical current, Ic-."""
    bstar = bstar_model(b, c, tau)
    return 1 - b * (1 - c * np.sign(x + bstar)) * (x + bstar) ** 2


plt.style.use("fullscreen13")

# set up the command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# parser.add_argument(
#    "--metric",
#    choices=["delta", "normalized", "both"],
#    default="both",
#    help="whether to plot ΔIc, ΔIc/ΣIc (normalized), or both"
# )
parser.add_argument("--bmax", default=0.1, help="x-axis range [-bmax, bmax]")
args = parser.parse_args()

# adjustable parameters
SliderKwargs = namedtuple(
    "SliderKwargs",
    "label valmin valmax valinit valfmt closedmax closedmin handle_style",
    defaults=(None, True, True, {"size": 10}),
)
b = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$b$ (T$^{-2}$)",
        valmin=0,
        valmax=50,
        valinit=10,
        valfmt="%0.2f",
        closedmin=False,
    )
)
c = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$c = \frac{k_\mathrm{SO}}{k_\mathrm{F}}$",
        valmin=0,
        valmax=1,
        valinit=0,
        valfmt="%0.2f",
    )
)
tau = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\tau$",
        valmin=0,
        valmax=1,
        valinit=0,
        valfmt="%0.4f",
        closedmax=False,
    )
)
params = [b, c, tau]

fig = plt.figure()
(fig, slider_fig) = fig.subfigures(1, 2, width_ratios=[4, 1])

slider_axs = slider_fig.subplots(1, len(params))
for param, ax in zip(params, slider_axs):
    param.slider = Slider(ax=ax, orientation="vertical", **param.kwargs._asdict())
    ax.get_children()[4].set_rotation(45)
    # ax.get_children()[4].set_size(10)

ax_ic, ax_diode = fig.subplots(2, 1, sharex=True)
text_bstar = ax_ic.text(
    0.5, 0.05, "", transform=ax_ic.transAxes, va="bottom", ha="center"
)
ax_diode.set_xlabel("in-plane field (T)")
ax_ic.set_ylabel(r"$|I_c|$")
ax_diode.set_ylabel(r"$\Delta|I_c|$")
# ax_ic.set_ylim(0.5, 1)

# initialize Lines
(line_icp,) = ax_ic.plot(0, 0, label=r"$I_{c+}$")
(line_icm,) = ax_ic.plot(0, 0, label=r"$I_{c-}$")
ax_ic.fill_between([0], [0])
ax_ic.legend()
(line_delta,) = ax_diode.plot(0, 0, color="tab:green")

bfield = np.linspace(-args.bmax, args.bmax, 101)


def update(_=None):
    bstar = bstar_model(b.slider.val, c.slider.val, tau.slider.val)
    text_bstar.set_text(f"$B_* = {bstar:0.3f}$ T")

    icp = icp_model(bfield, b.slider.val, c.slider.val, tau.slider.val)
    icm = icm_model(bfield, b.slider.val, c.slider.val, tau.slider.val)

    ax_ic.collections.pop()
    ax_ic.fill_between(bfield, icp, icm, color="tab:green", alpha=0.5)

    line_icp.set_data(bfield, icp)
    line_icm.set_data(bfield, icm)
    line_delta.set_data(bfield, icp - icm)

    ax_diode.relim()
    ax_diode.autoscale()
    fig.canvas.draw_idle()


update()
for ax in (ax_ic, ax_diode):
    ax.relim()
    ax.autoscale()
[p.slider.on_changed(update) for p in params]

plt.get_current_fig_manager().full_screen_toggle()
plt.show()
