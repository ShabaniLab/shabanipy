"""Investigate dependence of SQUID critical behavior on junction and loop parameters.

The junction and loop parameters can be adjusted by sliders.
"""
import argparse
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from shabanipy.jj import transparent_cpr as cpr
from shabanipy.squid import critical_behavior

plt.style.use(
    [
        {
            "figure.constrained_layout.use": True,
            "font.size": 12,
            "lines.markersize": 1,
            "lines.linewidth": 2,
        },
    ]
)


# set up the command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--flux",
    "-f",
    default=False,
    action="store_true",
    help="plot vs. total flux instead of external flux",
)
args = parser.parse_args()

# adjustable parameters
SliderKwargs = namedtuple(
    "SliderKwargs",
    "label valmin valmax valinit valfmt closedmax handle_style",
    defaults=(None, True, {"size": 10}),
)
phi01 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\varphi_0^{(1)}$", valmin=-1, valmax=1, valinit=0, valfmt="%0.2fπ"
    )
)
phi02 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\varphi_0^{(2)}$", valmin=-1, valmax=1, valinit=0, valfmt="%0.2fπ"
    )
)
logratio = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"log$_{10}\frac{I_{c2}}{I_{c1}}$", valmin=-1, valmax=1, valinit=0
    )
)
tau1 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\tau_1$",
        valmin=0,
        valmax=1,
        valinit=0,
        valfmt="%0.4f",
        closedmax=False,
    )
)
tau2 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\tau_2$",
        valmin=0,
        valmax=1,
        valinit=0,
        valfmt="%0.4f",
        closedmax=False,
    )
)
L = SimpleNamespace(
    kwargs=SliderKwargs(
        label="L",
        valmin=0,
        valmax=1,
        valinit=0,
        valfmt=r"%0.3f $\frac{\Phi_0}{I_{c1} + I_{c2}}$",
    )
)
params = [logratio, tau1, tau2, phi01, phi02, L]

fig = plt.figure()
(fig, slider_fig) = fig.subfigures(1, 2, width_ratios=[4, 1])

slider_axs = slider_fig.subplots(1, len(params))
for param, ax in zip(params, slider_axs):
    param.slider = Slider(ax=ax, orientation="vertical", **param.kwargs._asdict())
    ax.get_children()[5].set_rotation(45)
logratio.slider.ax.get_children()[4].set_rotation(45)

((squid_axp, jj_axp), (squid_axn, jj_axn)) = fig.subplots(2, 2, sharex=True)
sqphase_axp = squid_axp.twinx()
sqphase_axn = squid_axn.twinx()
jjphase_axp = jj_axp.twinx()
jjphase_axn = jj_axn.twinx()
squid_axp.set_title("SQUID")
jj_axp.set_title("JJs")
xlabel = "$\Phi_\mathrm{ext}$"
ylabel = "$\Phi$"
if args.flux:
    xlabel, ylabel = ylabel, xlabel
for ax in (squid_axn, jj_axn):
    ax.set_xlabel(xlabel)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(
        [r"$-\Phi_0$", r"$-\frac{\Phi_0}{2}$", "0", r"$\frac{\Phi_0}{2}$", r"$\Phi_0$"]
    )
    ax.set_xlim((-1, 1))
for ax in (squid_axp, squid_axn):
    ax.set_ylabel("$I_c$ ($I_{c1}$)")
for ax in (jj_axp, jj_axn):
    ax.set_ylabel("$I$ ($I_{c1}$)")
for ax in (jjphase_axp, jjphase_axn):
    ax.set_ylabel("phase")
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", "$\pi$", "$2\pi$"])
    ax.set_ylim((0, 2 * np.pi))
for ax in (sqphase_axp, sqphase_axn):
    ax.set_ylabel(ylabel)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{\Phi_0}{2}$", r"$\Phi_0$"])
    ax.set_ylim((0, 1))

# initialize Lines
flux = np.linspace(-1, 1, 501)
phase = flux * 2 * np.pi
(ic_linep,) = squid_axp.plot(0, 0)
(ic_linen,) = squid_axn.plot(0, 0)
squid_axp.fill_between([0], [0])
squid_axn.fill_between([0], [0])
(i1_linep,) = jj_axp.plot(0, 0)
(i1_linen,) = jj_axn.plot(0, 0, label="$I_1$")
(i2_linep,) = jj_axp.plot(0, 0)
(i2_linen,) = jj_axn.plot(0, 0, label="$I_2$")
(p_linep,) = sqphase_axp.plot(0, 0, ".k")
(p_linen,) = sqphase_axn.plot(0, 0, ".k")
(p1_linep,) = jjphase_axp.plot(0, 0, ".", color="tab:green")
(p1_linen,) = jjphase_axn.plot(0, 0, ".", color="tab:green", label=r"$\varphi_1$")
(p2_linep,) = jjphase_axp.plot(0, 0, ".", color="tab:red")
(p2_linen,) = jjphase_axn.plot(0, 0, ".", color="tab:red", label=r"$\varphi_2$")
jj_axn.legend(loc="lower left")
jjphase_axn.legend(loc="lower right")

ic1 = 1


def update(_=None):
    ic2 = 10 ** logratio.slider.val
    for branch, ic_line, i1_line, i2_line, p_line, p1_line, p2_line, sq_ax in zip(
        ("+", "-"),
        (ic_linep, ic_linen),
        (i1_linep, i1_linen),
        (i2_linep, i2_linen),
        (p_linep, p_linen),
        (p1_linep, p1_linen),
        (p2_linep, p2_linen),
        (squid_axp, squid_axn),
    ):
        p_ext, ic, p1, i1, p2, i2 = critical_behavior(
            phase,
            cpr,
            (phi01.slider.val * np.pi, ic1, tau1.slider.val),
            cpr,
            (phi02.slider.val * np.pi, ic2, tau2.slider.val),
            inductance=L.slider.val / (ic1 + ic2),
            branch=branch,
            nbrute=201,
            return_jjs=True,
        )
        xdata = (phase if args.flux else p_ext) / (2 * np.pi)
        sq_ax.collections[-1].remove()
        sq_ax.fill_between(xdata, ic, color="tab:blue")
        ic_line.set_data(xdata, ic)
        i1_line.set_data(xdata, i1)
        i2_line.set_data(xdata, i2)
        p_line.set_data(
            xdata, (p_ext if args.flux else phase) % (2 * np.pi) / (2 * np.pi)
        )
        p1_line.set_data(xdata, p1 % (2 * np.pi))
        p2_line.set_data(xdata, p2 % (2 * np.pi))

        # remove pop and color=... for a cool trip
        for ax in (squid_axp, squid_axn, jj_axp, jj_axn):
            ax.relim()
            ax.autoscale_view()
    fig.canvas.draw_idle()


update()
[p.slider.on_changed(update) for p in params]

plt.get_current_fig_manager().full_screen_toggle()
plt.show()
