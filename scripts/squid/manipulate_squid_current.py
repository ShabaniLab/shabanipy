"""Investigate the SQUID current dependence on junction and loop parameters.

The junction and loop parameters can be adjusted by sliders.
"""
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from shabanipy.squid.cpr import finite_transparency_jj_current as cpr
from shabanipy.squid.squid_model import compute_squid_current

phase = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# adjustable parameters
SliderKwargs = namedtuple(
    "SliderKwargs",
    "label valmin valmax valinit valfmt closedmax",
    defaults=(None, True),
)
phi1 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\varphi_0^{(1)}$", valmin=-2, valmax=2, valinit=0, valfmt="%0.2fπ"
    )
)
phi2 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\varphi_0^{(2)}$", valmin=-2, valmax=2, valinit=0, valfmt="%0.2fπ"
    )
)
logratio = SimpleNamespace(
    kwargs=SliderKwargs(
        label="log$_{10}(I_{c2}/I_{c1})$", valmin=-1, valmax=1, valinit=0
    )
)
tau1 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\tau_1$",
        valmin=0,
        valmax=1,
        valinit=0.5,
        valfmt="%0.4f",
        closedmax=False,
    )
)
tau2 = SimpleNamespace(
    kwargs=SliderKwargs(
        label=r"$\tau_2$",
        valmin=0,
        valmax=1,
        valinit=0.5,
        valfmt="%0.4f",
        closedmax=False,
    )
)
L = SimpleNamespace(kwargs=SliderKwargs(label="L", valmin=0, valmax=1e-9, valinit=0))
params = [logratio, tau1, tau2, phi1, phi2, L]

fig = plt.figure()
# plots on the left, sliders on the right
subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])

axs = subfigs[0].subplots(2, 1, sharex=True)
axs[1].set_xlabel("SQUID phase [2π]")
[ax.set_ylabel("supercurrent [$I_{c1}$]") for ax in axs]

slider_axs = subfigs[1].subplots(1, len(params))
for param, ax in zip(params, slider_axs):
    param.slider = Slider(ax=ax, orientation="vertical", **param.kwargs._asdict())


def update(_=None):
    for ax, branch in zip(axs, [True, False]):
        (line,) = ax.get_lines()
        line.set_ydata(
            compute_squid_current(
                phase,
                cpr,
                (phi1.slider.val * np.pi, 1, tau1.slider.val),
                cpr,
                (phi2.slider.val * np.pi, 10 ** logratio.slider.val, tau2.slider.val,),
                inductance=L.slider.val,
                positive=branch,
            ),
        )
        # remove pop and color=... for a cool trip
        ax.collections.pop()
        poly = ax.fill_between(*line.get_data(), color="tab:blue")
        ax.relim()
        ax.autoscale_view()
    fig.canvas.draw_idle()


[ax.plot(phase / (2 * np.pi), phase) for ax in axs]  # initialize Lines
[ax.fill_between(phase / (2 * np.pi), phase) for ax in axs]  # initialize fill
update()
[p.slider.on_changed(update) for p in params]

plt.get_current_fig_manager().full_screen_toggle()
plt.show()
