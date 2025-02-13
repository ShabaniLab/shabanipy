"""SQUID model functions to fit against experimental data.

These functions are used to construct an lmfit.Model."""

from functools import partial
from inspect import signature
from typing import Literal

import numpy as np
from lmfit import Model, Parameters
from scipy.constants import elementary_charge as e
from scipy.constants import physical_constants

from shabanipy.jj import transparent_cpr as tcpr

from .estimate import (
    estimate_bfield_offset,
    estimate_critical_current,
    estimate_frequency,
)
from .squid import critical_behavior, critical_control

PHI0 = physical_constants["mag. flux quantum"][0]


def squid_model(
    bfield: np.ndarray,
    bfield_offset: float,
    loop_area: float,
    anomalous_phase1: float,
    anomalous_phase2: float,
    critical_current1: float,
    critical_current2: float,
    transparency1: float,
    transparency2: float,
    inductance: float,
    temperature: float,
    gap: float,
    branch: Literal["+", "-", "+-"],
    nbrute: int = 101,
    ninterp: int = 101,
):
    """Model of the critical current of a dc SQUID with transparent junctions.

    Parameters
    ----------
    bfield
        External magnetic field (T) at which the critical current was measured.
    bfield_offset
        Magnetic field offset (T) due to hysteresis in the vector magnet coils.
    loop_area
        Effective area A (m^2) of the SQUID loop, used to convert the external field B
        to phase via 2πAB/Φ0.
    anomalous_phase1, anomalous_phase2
        Anomalous shifts in the junctions' phases (rad).
    critical_current1, critical_current2
        Junctions' critical currents (A).
    transparency1, transparency2
        Junctions' transparencies, on the interval [0, 1).
    inductance
        SQUID loop inductance (H).
    temperature
        Mixing chamber temperature (K), or electron temperature if known.
    gap
        Superconducting gap Δ (eV).
    branch
        Which branch of the SQUID critical current to model: positive (+), negative (-),
        or both (+-).
    nbrute
        Number of points to use in the brute-force optimization of the SQUID current.
    ninterp
        Number of points in total phase Φ ~ [0, 2π] used to interpolate the SQUID
        behavior as a function of Φ_ext(Φ).

    Returns
    -------
    squid_ic
        The SQUID critical current Ic(B_ext) for the requested `branch`es.
        If both are requested, squid_ic[0] is the positive branch and squid_ic[1] is the
        negative branch.
    """
    critical = (
        critical_behavior
        if inductance == 0
        else partial(critical_control, ninterp=int(ninterp))
    )
    phase_ext = 2 * np.pi * loop_area * (bfield - bfield_offset) / PHI0
    squid_ic = [
        critical(
            phase_ext,
            tcpr,
            (anomalous_phase1, critical_current1, transparency1, temperature, gap * e),
            tcpr,
            (anomalous_phase2, critical_current2, transparency2, temperature, gap * e),
            inductance=inductance / PHI0,
            branch=b,
            nbrute=int(nbrute),
        )[1]
        for b in branch
    ]
    return np.squeeze(squid_ic)


class SquidModel(Model):
    """Model of the critical current of a dc SQUID with transparent junctions."""

    # default parameter specifications
    param_specs = {
        "anomalous_phase1": {"value": 0, "vary": False},
        "anomalous_phase2": {"value": 0, "vary": False},
        "transparency1": {"min": 0, "max": 1},
        "transparency2": {"min": 0, "max": 1},
        "inductance": {"value": 0, "vary": False},
        "temperature": {"vary": False},
        "gap": {"vary": False},
    }

    def __init__(self, *args, **kwargs):
        param_names = [
            p
            for p in list(signature(squid_model).parameters.keys())[1:]
            if p not in kwargs
        ]
        super().__init__(
            squid_model,
            *args,
            param_names=param_names,
            **kwargs,
        )
        for pname in [p for p in param_names if p in self.param_specs]:
            self.set_param_hint(pname, **self.param_specs[pname])

    def guess(self, ic: np.ndarray, bfield: np.ndarray, **kwargs) -> Parameters:
        """Guess initial values for the fit parameters.

        Only parameters without initial values are guessed.

        Parameters
        ----------
        ic
            SQUID critical current data Ic(B_ext).  If shape (2, N), assumes ic[0] is
            the positive branch and ic[1] the negative branch.
        bfield
            External magnetic field B_ext at which `ic` was measured.  Must be 1d.

        Returns
        -------
        Initialized Parameters.
        """
        for pname in self.param_names:
            if pname not in self.param_hints or "value" not in self.param_hints[pname]:
                getattr(self, f"_guess_{pname}")(ic, bfield, **kwargs)
        return self.make_params()

    def _guess_bfield_offset(self, ic, bfield, **kwargs):
        ic = np.atleast_2d(ic)
        bfield_offset, idxs = estimate_bfield_offset(
            bfield,
            ic[0] if self.opts["branch"] in {"+", "+-"} else None,
            ic[-1] if self.opts["branch"] in {"-", "+-"} else None,
        )
        self.set_param_hint("bfield_offset", value=bfield_offset)

    def _guess_loop_area(self, ic, bfield, **kwargs):
        cycles_per_tesla, (freqs, fft) = estimate_frequency(bfield, ic)
        self.loop_area_estimate = (cycles_per_tesla, (freqs, fft))
        self.set_param_hint("loop_area", value=cycles_per_tesla * PHI0)

    def _guess_critical_current1(self, ic, bfield, **kwargs):
        self._guess_critical_current(1, ic, **kwargs)

    def _guess_critical_current2(self, ic, bfield, **kwargs):
        self._guess_critical_current(2, ic, **kwargs)

    def _guess_critical_current(self, jj, ic, **kwargs):
        # if present, smaller_ic_jj is 1 or 2
        smaller_ic_jj = kwargs.get("smaller_ic_jj")
        if smaller_ic_jj is not None:
            smaller_ic_jj = jj == smaller_ic_jj
        self.set_param_hint(
            f"critical_current{jj}", value=estimate_critical_current(ic, smaller_ic_jj)
        )

    def _guess_transparency1(self, ic, bfield, **kwargs):
        self.set_param_hint("transparency1", value=0)

    def _guess_transparency2(self, ic, bfield, **kwargs):
        self.set_param_hint("transparency2", value=0)

    def _guess_temperature(self, ic, bfield, **kwargs):
        self.set_param_hint(
            "temperature", value=round(np.mean(kwargs["temp"]), 3), vary=False
        )
