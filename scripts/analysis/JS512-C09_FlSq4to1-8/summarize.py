"""Plot data for JS512-C09_FlSq4to1-8, cooldowns WFS-01 and WFS02.

This script depends on data aggregated by `./aggregate.py` which should be run first.

Usage:
    $ python summarize.py [SummarizingStep name] [SummarizingStep name] [...]
If no SummarizingStep names are specified, all will run.

This script will output a lot of errors/warnings, mostly due to hamfisted aggregation
and a few Labber oddities/bugs.  They can be ignored.
"""
import logging
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure

from shabanipy.bulk.data_processing import (
    PreProcessingStep,
    ProcessCoordinator,
    ProcessingStep,
    SummarizingStep,
)
from shabanipy.constants import VECTOR9_AMPS_PER_TESLA_X, VECTOR10_AMPS_PER_TESLA_X
from shabanipy.logging import ConsoleFormatter, configure_logging
from shabanipy.plotting import jy_pink, plot, plot2d
from shabanipy.plotting.utils import stamp

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.formatter = ConsoleFormatter()
logger.addHandler(handler)
configure_logging()

plt.style.use(["presentation", "jy_pink"])

DV_DI = r"$\mathrm{d}V/\mathrm{d}I$ ($\Omega$)"
DELTAV_DELTAI = r"$\Delta V / \Delta I$ ($\Omega$)"

pp_steps = [
    PreProcessingStep(
        name="undo dc voltage amplifier",
        measurements=[],
        input_quantities=["dmm"],
        parameters={"gain": 1e2},
        routine=lambda volt, gain: volt / gain,
        output_quantities=["dc voltage"],
    ),
    PreProcessingStep(
        name="convert magnet current to field",
        measurements=[],
        input_quantities=["x magnet current"],
        parameters={
            "labber_filename": "attrs@__file__",
            "__file__": "attrs@__file__",  # a silly hack
        },
        routine=lambda current, labber_filename, __file__: current
        / amps_per_tesla[re.search("WFS-?0\d", labber_filename).group()],
        output_quantities=["x magnet field"],
    ),
    PreProcessingStep(
        name="convert fluxline voltage to current",
        measurements=["fluxline fraunhofer", "fluxline (voltage source) fraunhofer"],
        input_quantities=["fluxline voltage"],
        parameters={},
        routine=lambda volt: volt / 200e3,
        output_quantities=["fluxline current bias"],
    ),
    PreProcessingStep(
        name="convert yoko voltage to current",
        measurements=[],
        input_quantities=["dc bias (volts)"],
        parameters={},
        routine=lambda volt: volt / 1e6,
        output_quantities=["dc current bias"],
    ),
    PreProcessingStep(
        name="convert lock-in voltage to resistance",
        measurements=[],
        input_quantities=["lock-in (volts)"],
        parameters={},
        routine=lambda volt: np.abs(volt) / 10e-9,
        output_quantities=["lock-in (impedance)"],
    ),
]

s_steps = [
    SummarizingStep(
        name="4um gate leakage",
        input_origin="raw@4um gate leakage",
        input_quantities=["4um gate voltage", "4um gate current", "Xum gate current"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda volts, amps4um, ampsXum, **params: plot_summary(
            volts,
            np.array([amps4um, ampsXum]).T / 1e-12,
            xlabel=r"4μm gate voltage (V)",
            ylabel=r"current (pA)",
            legend=("4μm gate", f"{get_width(params)}μm gate"),
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$",
            **params,
        ),
    ),
    SummarizingStep(
        name="4um gate leakage with fluxline",
        input_origin="raw@4um gate leakage",
        input_quantities=[
            "4um gate voltage",
            "4um gate current",
            "Xum gate current",
            "fluxline current measured",
        ],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda volts, amps4um, ampsXum, amps_fluxline, **params: plot_summary(
            volts,
            np.array([amps4um, ampsXum, amps_fluxline]).T / 1e-12,
            xlabel=r"4μm gate voltage (V)",
            ylabel=r"current (pA)",
            legend=("4μm gate", f"{get_width(params)}μm gate", "fluxline"),
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$",
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate leakage",
        input_origin="raw@Xum gate leakage",
        input_quantities=["Xum gate voltage", "4um gate current", "Xum gate current"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda volts, amps4um, ampsXum, **params: plot_summary(
            volts,
            np.array([amps4um, ampsXum]).T / 1e-12,
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"current (pA)",
            legend=("4μm gate", f"{get_width(params)}μm gate"),
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$",
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate leakage with fluxline",
        input_origin="raw@Xum gate leakage",
        input_quantities=[
            "Xum gate voltage",
            "4um gate current",
            "Xum gate current",
            "fluxline current measured",
        ],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda volts, amps4um, ampsXum, amps_fluxline, **params: plot_summary(
            volts,
            np.array([amps4um, ampsXum, amps_fluxline]).T / 1e-12,
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"current (pA)",
            legend=("4μm gate", f"{get_width(params)}μm gate", "fluxline"),
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$",
            **params,
        ),
    ),
    SummarizingStep(
        name="4um gate vs bias (lock-in)",
        input_origin="raw@4um gate vs bias",
        input_quantities=["4um gate voltage", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, lock_in, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=f"4μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="4um gate vs bias (dmm)",
        input_origin="raw@4um gate vs bias",
        input_quantities=["4um gate voltage", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, volts, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.diff(volts) / np.diff(ibias),
            xlabel=f"4μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="4um gate vs bias (lock-in) without curvetracer",
        input_origin="raw@4um gate vs bias without curvetracer",
        input_quantities=["4um gate voltage", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, lock_in, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=f"4μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="4um gate vs bias (dmm) without curvetracer",
        input_origin="raw@4um gate vs bias without curvetracer",
        input_quantities=["4um gate voltage", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, volts, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.diff(volts) / np.diff(ibias),
            xlabel=f"4μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=f"$V_g^\mathrm{{ {get_width(params)}μm }}={flat_clfs(params)['Xum gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate vs bias (lock-in)",
        input_origin="raw@Xum gate vs bias",
        input_quantities=["Xum gate voltage", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, lock_in, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate vs bias (dmm)",
        input_origin="raw@Xum gate vs bias",
        input_quantities=["Xum gate voltage", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, volts, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.diff(volts) / np.diff(ibias),
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate vs bias (lock-in) without curvetracer",
        input_origin="raw@Xum gate vs bias without curvetracer",
        input_quantities=["Xum gate voltage", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, lock_in, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="Xum gate vs bias (dmm) without curvetracer",
        input_origin="raw@Xum gate vs bias without curvetracer",
        input_quantities=["Xum gate voltage", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda gate, ibias, volts, **params: plot2d_summary(
            gate,
            ibias / 1e-6,
            np.diff(volts) / np.diff(ibias),
            xlabel=f"{get_width(params)}μm gate voltage (V)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=f"$V_g^\mathrm{{4μm}}={flat_clfs(params)['4um gate voltage']}$V",
            pcm_kwargs={"vmin": 0, "vmax": 1000},
            **params,
        ),
    ),
    SummarizingStep(
        name="fraunhofer",
        input_origin="raw@fraunhofer",
        input_quantities=["x magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"applied $\perp$ field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **(
                    {"vmax": 500}
                    if params["labber_filename"].endswith("WFS02_151.hdf5")
                    else {}
                ),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="fraunhofer without curvetracer",
        input_origin="raw@fraunhofer without curvetracer",
        input_quantities=["x magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"applied $\perp$ field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={"vmin": 0},
            **params,
        ),
    ),
    SummarizingStep(
        name="fraunhofer (dmm)",
        input_origin="raw@fraunhofer",
        input_quantities=["x magnet field", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, volt, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.diff(volt) / np.diff(ibias),
            xlabel=r"applied $\perp$ field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **(
                    {"vmax": 999}
                    if np.max(np.diff(volt) / np.diff(ibias)) > 2500
                    else {}
                ),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="fraunhofer (dmm) without curvetracer",
        input_origin="raw@fraunhofer without curvetracer",
        input_quantities=["x magnet field", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, volt, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.diff(volt) / np.diff(ibias),
            xlabel=r"applied $\perp$ field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **(
                    {"vmax": 999}
                    if np.max(np.diff(volt) / np.diff(ibias)) > 2500
                    else {}
                ),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="in-plane vs bias",
        input_origin="raw@in-plane vs bias",
        input_quantities=["y magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"applied $B_y$ (T)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **({"vmax": 500} if np.max(np.abs(lock_in)) > 1000 else {}),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="in-plane vs bias without curvetracer",
        input_origin="raw@in-plane vs bias without curvetracer",
        input_quantities=["y magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"applied $B_y$ (T)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V"
            ),
            append_to_filename=f"(Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **({"vmax": 500} if np.max(np.abs(lock_in)) > 1000 else {}),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="cpr",
        input_origin="raw@cpr",
        input_quantities=["x magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"$x$ coil field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${round(get_num(params, '4um gate voltage'), 2)}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${round(get_num(params, 'Xum gate voltage'), 2)}$V, "
                f"$B_y$={get_num(params, 'y magnet field')}T, "
                f"$T$={get_temp(params)}"
            ),
            append_to_filename=(
                f"(T={get_temp(params)}_"
                f"By={get_num(params, 'y magnet field')}_"
                f"Vg4um={round(get_num(params, '4um gate voltage'), 2)}_"
                f"Vg{get_width(params)}um={round(get_num(params, 'Xum gate voltage'), 2)})"
            ),
            pcm_kwargs={
                "vmin": 0,
                **({"vmax": 1000} if np.max(np.abs(lock_in)) > 2000 else {}),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="cpr without curvetracer",
        input_origin="raw@cpr without curvetracer",
        input_quantities=["x magnet field", "dc current bias", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, ibias, lock_in, **params: plot2d_summary(
            bfield / 1e-3,
            ibias / 1e-6,
            np.abs(lock_in),
            xlabel=r"$x$ coil field (mT)",
            ylabel=r"dc bias (μA)",
            zlabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
                f"$B_y$={get_num(params, 'y magnet field')}T, "
                f"$T$={get_temp(params)}"
            ),
            append_to_filename=f"(T={get_temp(params)}_By={get_num(params, 'y magnet field')}_Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            pcm_kwargs={
                "vmin": 0,
                **({"vmax": 1000} if np.max(np.abs(lock_in)) > 2000 else {}),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="frequency vs amplitude",
        input_origin="raw@frequency vs amplitude",
        input_quantities=[
            "dc current bias",
            "lock-in (impedance)",
            "lock-in frequency",
            "lock-in amplitude",
        ],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, lock_in, freq, ampl, **params: plot_summary(
            np.unique(ibias) / 1e-6,
            np.array([
                np.abs(lock_in[np.where((freq == np.unique(freq)[1]) & (ampl == np.unique(ampl)[0]))]),
                np.abs(lock_in[np.where((freq == np.unique(freq)[1]) & (ampl == np.unique(ampl)[1]))]),
                np.abs(lock_in[np.where((freq == np.unique(freq)[0]) & (ampl == np.unique(ampl)[0]))]),
                np.abs(lock_in[np.where((freq == np.unique(freq)[0]) & (ampl == np.unique(ampl)[1]))]),
            ]).T,
            xlabel="dc bias (μA)",
            ylabel=DV_DI,
            legend=(
                f"{np.round(np.unique(freq)[1], 2)}Hz, {maybe_int(np.round(np.unique(ampl)[0] / 1e6 / 1e-9, 2))}nA",
                f"{np.round(np.unique(freq)[1], 2)}Hz, {maybe_int(np.round(np.unique(ampl)[1] / 1e6 / 1e-9, 2))}nA",
                f"{np.round(np.unique(freq)[0], 2)}Hz, {maybe_int(np.round(np.unique(ampl)[0] / 1e6 / 1e-9, 2))}nA",
                f"{np.round(np.unique(freq)[0], 2)}Hz, {maybe_int(np.round(np.unique(ampl)[1] / 1e6 / 1e-9, 2))}nA",
            ),
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
                f"$B_y$={get_num(params, 'y magnet field')}T, "
                f"$T$={get_temp(params)}"
            ),
            append_to_filename=f"(T={get_temp(params)}_By={get_num(params, 'y magnet field')}_Vg4um={get_num(params, '4um gate voltage')}_VgXum={get_num(params, 'Xum gate voltage')})",
            **params,
        ),
    ),
    SummarizingStep(
        name="hysteresis",
        input_origin="raw@hysteresis",
        input_quantities=[
            "dc current bias",
            "lock-in (impedance)",
            "sweep direction",
            "mc temperature",
        ],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, lock_in, direction, temp, **params: ibias.shape[0] == 2
        and plot_summary(
            np.unique(ibias) / 1e-6,
            np.array(
                [
                    np.abs(lock_in[np.where(direction == np.unique(direction)[0])]),
                    # alternating sweep directions isn't handled in LabberData.get_data
                    np.flip(
                        np.abs(lock_in[np.where(direction == np.unique(direction)[1])])
                    ),
                ]
            ).T,
            xlabel="dc bias (μA)",
            ylabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
                f"$B_y$={get_num(params, 'y magnet field')}T, "
                f"$T$={np.round(np.mean(temp), 2)}K"
            ),
            append_to_filename=(
                f"(T={np.round(np.mean(temp), 2)}_"
                f"By={get_num(params, 'y magnet field')}_"
                f"Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            **params,
        ),
    ),
    SummarizingStep(
        name="hysteresis (dmm)",
        input_origin="raw@hysteresis",
        input_quantities=[
            "dc current bias",
            "dc voltage",
            "sweep direction",
            "mc temperature",
        ],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, volt, direction, temp, **params: ibias.shape[0] == 2
        and plot_summary(
            np.unique(ibias) / 1e-6,
            np.array(
                [
                    volt[np.where(direction == np.unique(direction)[0])],
                    # alternating sweep directions isn't handled in LabberData.get_data
                    np.flip(volt[np.where(direction == np.unique(direction)[1])]),
                ]
            ).T
            / 1e-6,
            xlabel="dc bias (μA)",
            ylabel="un-amplified voltage (μV)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
                f"$B_y$={get_num(params, 'y magnet field')}T, "
                f"$T$={np.round(np.mean(temp), 2)}K"
            ),
            append_to_filename=(
                f"(T={np.round(np.mean(temp), 2)}_"
                f"By={get_num(params, 'y magnet field')}_"
                f"Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            **params,
        ),
    ),
    SummarizingStep(
        name="Rn",
        input_origin="raw@Rn",
        input_quantities=["dc current bias", "lock-in (impedance)", "mc temperature",],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, lock_in, temp, **params: plot_summary(
            np.unique(ibias) / 1e-6,
            np.abs(lock_in),
            xlabel="dc bias (μA)",
            ylabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            ylim=(
                0,
                3000
                if np.max(np.abs(lock_in)) == np.abs(lock_in)[0]
                else 5000
                if np.max(np.abs(lock_in)) > 5000
                else None,
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="Rn without curvetracer",
        input_origin="raw@Rn without curvetracer",
        input_quantities=["dc current bias", "lock-in (impedance)", "mc temperature",],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, lock_in, temp, **params: len(ibias.shape) == 1
        and plot_summary(
            np.unique(ibias) / 1e-6,
            np.abs(lock_in),
            xlabel="dc bias (μA)",
            ylabel=DV_DI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            ylim=(
                0,
                3000
                if np.max(np.abs(lock_in)) == np.abs(lock_in)[0]
                else 5000
                if np.max(np.abs(lock_in)) > 5000
                else None,
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="Rn (dmm)",
        input_origin="raw@Rn",
        input_quantities=["dc current bias", "dc voltage", "mc temperature",],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, volt, temp, **params: plot_summary(
            np.unique(ibias) / 1e-6,
            volt / 1e-3,
            xlabel="dc bias (μA)",
            ylabel="un-amplified voltage (mV)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="Rn (dmm) without curvetracer",
        input_origin="raw@Rn without curvetracer",
        input_quantities=["dc current bias", "dc voltage", "mc temperature",],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, volt, temp, **params: len(ibias.shape) == 1
        and plot_summary(
            np.unique(ibias) / 1e-6,
            volt / 1e-3,
            xlabel="dc bias (μA)",
            ylabel="un-amplified voltage (mV)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="Bc",
        input_origin="raw@Bc",
        input_quantities=["z magnet field", "lock-in (impedance)"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda bfield, lock_in, **params: plot_summary(
            bfield[(bfield > 1.3) & (bfield < 1.6)],
            np.abs(lock_in[(bfield > 1.3) & (bfield < 1.6)]),
            xlabel="z magnet coils (T)",
            ylabel=DV_DI,
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline fraunhofer",
        input_origin="raw@fluxline fraunhofer",
        input_quantities=["fluxline current bias", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda fluxline, ibias, volt, **params: plot2d_summary(
            fluxline / 1e-6,
            ibias / 1e-6,
            np.diff(volt) / np.diff(ibias),
            xlabel="fluxline current (μA)",
            ylabel="dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            pcm_kwargs={
                "vmin": 0,
                **(
                    {"vmax": 500}
                    if np.max(np.diff(volt) / np.diff(ibias)) > 500
                    else {}
                ),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline (voltage source) fraunhofer",
        input_origin="raw@fluxline (voltage source) fraunhofer",
        input_quantities=["fluxline current bias", "dc current bias", "dc voltage"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda fluxline, ibias, volt, **params: plot2d_summary(
            fluxline / 1e-6,
            ibias / 1e-6,
            np.diff(volt) / np.diff(ibias),
            xlabel="fluxline current (μA)*",
            ylabel="dc bias (μA)",
            zlabel=DELTAV_DELTAI,
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            pcm_kwargs={
                "vmin": 0,
                **(
                    {"vmax": 500}
                    if np.max(np.diff(volt) / np.diff(ibias)) > 500
                    else {}
                ),
            },
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline heating",
        input_origin="raw@fluxline fraunhofer",
        input_quantities=["fluxline current bias", "mc temperature"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda fluxline, temp, **params: plot_summary(
            np.mean(fluxline, axis=-1) / 1e-6,
            np.mean(temp, axis=-1) / 1e-3,
            xlabel="fluxline current (μA)",
            ylabel="MC temperature (mK)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline (voltage source) heating",
        input_origin="raw@fluxline (voltage source) fraunhofer",
        input_quantities=["fluxline current bias", "mc temperature"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda fluxline, temp, **params: plot_summary(
            np.mean(fluxline, axis=-1) / 1e-6,
            np.mean(temp, axis=-1) / 1e-3,
            xlabel="fluxline current (μA)*",
            ylabel="MC temperature (mK)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            plot_kwargs={"lw": 2},
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline current",
        input_origin="raw@fluxline fraunhofer",
        input_quantities=["fluxline current bias", "fluxline current measured"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, imeas, **params: plot_summary(
            np.mean(ibias, axis=-1) / 1e-6,
            np.mean(imeas, axis=-1) / 1e-6,
            xlabel="fluxline current bias (μA)",
            ylabel="fluxline current measured (μA)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            **params,
        ),
    ),
    SummarizingStep(
        name="fluxline (voltage source) current",
        input_origin="raw@fluxline (voltage source) fraunhofer",
        input_quantities=["fluxline current bias", "fluxline current measured"],
        parameters={"labber_filename": "attrs@__file__"},
        routine=lambda ibias, imeas, **params: plot_summary(
            np.mean(ibias, axis=-1) / 1e-6,
            np.mean(imeas, axis=-1) / 1e-6,
            xlabel="fluxline current bias (μA)*",
            ylabel="fluxline current measured (μA)",
            title=(
                f"$V_g^\mathrm{{4μm}}$=${get_num(params, '4um gate voltage')}$V, "
                f"$V_g^\mathrm{{ {get_width(params)}μm }}$=${get_num(params, 'Xum gate voltage')}$V, "
            ),
            append_to_filename=(
                f"(Vg4um={get_num(params, '4um gate voltage')}_"
                f"VgXum={get_num(params, 'Xum gate voltage')})"
            ),
            **params,
        ),
    ),
]

# helpers

amps_per_tesla = {
    "WFS-01": VECTOR9_AMPS_PER_TESLA_X,
    "WFS02": VECTOR10_AMPS_PER_TESLA_X,
}


def maybe_int(a):
    return a if isinstance(a, str) else int(a) if a % 1 == 0 else a


def flat_clfs(params) -> dict:
    """Flatten the classifiers for easier access."""
    flat = {}
    for clfs in params["classifiers"].values():
        flat.update(clfs)
    return flat


def get_width(params) -> str:
    """Get the variable width of the second junction in the squid."""
    return re.match("FlSq4to([1-9])", flat_clfs(params)["device"]).group(1)


def get_num(params, key):
    """Get the number corresponding to key from the classifiers."""
    return maybe_int(flat_clfs(params)[key])


def get_temp(params):
    setpoint = flat_clfs(params)["temperature setpoint"]
    return "base" if setpoint <= 0.3 else setpoint


def out_dir(params) -> Path:
    """Create the output directory and return its Path."""
    out_dir = Path(params["directory"], flat_clfs(params)["device"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_summary(
    x,
    y,
    xlabel,
    ylabel,
    legend=None,
    title=None,
    append_to_filename="",
    ylim=(None, None),
    plot_kwargs={},
    **params,
):
    """Plot 1-dimensional data and save figure."""
    if len(x.shape) > 1:
        logger.warning(
            f"Failed to plot {params['labber_filename']}: "
            f"data is more than 1D ({x.shape=})"
        )
        return
    if x.shape[0] != y.shape[0]:
        logger.warning(
            f"Failed to plot {params['labber_filename']}: "
            f"first dimensions of data don't match ({x.shape=}, {y.shape=})"
        )
        return
    try:
        fig = Figure()
        ax = fig.add_subplot()
        plot(
            x,
            y,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ax=ax,
            label=legend,
            **plot_kwargs,
        )
    except Exception as e:
        logger.warning(
            f"Failed to plot data from {params['labber_filename']}: caught {repr(e)}"
        )
        logger.exception("ERROR")
        return
    ax.set_ylim(ylim)
    scan_id = flat_clfs(params)["scan"]
    stamp(ax, scan_id)
    fig.savefig(out_dir(params) / (scan_id + append_to_filename + ".png"), format="png")


def plot2d_summary(
    x,
    y,
    z,
    xlabel,
    ylabel,
    zlabel,
    title=None,
    append_to_filename="",
    pcm_kwargs={},
    rc_params={},
    **params,
):
    """Plot 2-dimensional data and save figure."""
    rcParams.update(rc_params)
    if any(len(a.shape) > 2 for a in (x, y, z)):
        logger.warning(
            f"Failed to plot {params['labber_filename']}: "
            f"data is more than 2D ({x.shape=} {y.shape=} {z.shape=})"
        )
        return
    if len(set([x.shape[0], y.shape[0], z.shape[0]])) > 1:
        logger.warning(
            f"Truncating {params['labber_filename']} along first axis: "
            f"scan may have been interrupted ({x.shape=} {y.shape=} {z.shape=})."
        )
        n = min(x.shape[0], y.shape[0], z.shape[0])
        x, y, z = x[:n], y[:n], z[:n]
    min_z = np.min(z)
    try:
        fig = Figure()
        ax = fig.add_subplot()
        plot2d(
            x,
            y,
            z,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title,
            ax=ax,
            **pcm_kwargs,
        )
    except Exception as e:
        logger.warning(
            f"Failed to plot data from {params['labber_filename']}: caught {repr(e)}"
        )
        return
    scan_id = flat_clfs(params)["scan"]
    stamp(ax, scan_id)
    fig.savefig(out_dir(params) / (scan_id + append_to_filename + ".png"), format="png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        s_steps = [ss for ss in s_steps if ss.name in sys.argv[1:]]
    required_inputs = set.union(*[set(ss.input_quantities) for ss in s_steps])
    pp_steps = [pps for pps in pp_steps if set(pps.output_quantities) & required_inputs]
    pc = ProcessCoordinator(
        archive_path="data_aggregated.hdf5",
        duplicate_path="data_preprocessed.hdf5",
        processing_path="data_processed.hdf5",
        summary_directory="plots",
        preprocessing_steps=pp_steps,
        processing_steps=[],
        summary_steps=s_steps,
    )
    pc.run_preprocess()
    pc.run_process()
    pc.run_summary()
