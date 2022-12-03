"""Plot the SQUID critical current for the given parameters."""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import e, physical_constants

from shabanipy.jj import transparent_cpr
from shabanipy.squid import critical_behavior

PHI0 = physical_constants["mag. flux quantum"][0]
plt.style.use("fullscreen13")

# set up the command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--ic1", type=float, default=1e-6, help="Critical current (A) of junction 1"
)
parser.add_argument(
    "--ic2", type=float, default=1e-6, help="Critical current (A) of junction 2"
)
parser.add_argument(
    "--t1", type=float, default=0, help="Transparency [0, 1) of junction 1"
)
parser.add_argument(
    "--t2", type=float, default=0, help="Transparency [0, 1) of junction 2"
)
parser.add_argument(
    "--phi1", type=float, default=0, help="Anomalous phase of junction 1"
)
parser.add_argument(
    "--phi2", type=float, default=0, help="Anomalous phase of junction 2"
)
parser.add_argument("--T", type=float, default=0, help="Temperature (K)")
parser.add_argument(
    "--gap", type=float, default=200e-6, help="Superconducting gap Δ (eV)"
)
parser.add_argument("--L", type=float, default=0, help="Loop inductance (H)")
parser.add_argument("--xres", type=int, default=501, help="Resolution in x")
parser.add_argument("--yres", type=int, default=501, help="Resolution in y")
parser.add_argument(
    "--flux",
    action="store_true",
    default=False,
    help="Plot vs. total flux instead of applied flux",
)
parser.add_argument(
    "--points", action="store_true", default=False, help="Plot points, not lines"
)
args = parser.parse_args()
print(args)

phase = np.linspace(-2 * np.pi, 2 * np.pi, args.xres)

fig, axs = plt.subplots(3, sharex=True)
xlabel = "total flux" if args.flux else "external flux"
axs[-1].set_xlabel(xlabel + " ($\Phi_0$)")
axs[0].set_ylabel("SQUID $I_c$ (μA)")
axs[1].set_ylabel("JJ1 $I$ (μA)")
axs[2].set_ylabel("JJ2 $I$ (μA)")

phase_ext, squid_ic, phase1, current1, phase2, current2 = critical_behavior(
    phase,
    transparent_cpr,
    (args.phi1, args.ic1, args.t1, args.T, args.gap * e),
    transparent_cpr,
    (args.phi2, args.ic2, args.t2, args.T, args.gap * e),
    inductance=args.L / PHI0,
    nbrute=args.yres,
    return_jjs=True,
)
x = phase if args.flux else phase_ext
kwargs = {"marker": ".", "markersize": 1, "lw": 0} if args.points else {}
axs[0].plot(x / (2 * np.pi), squid_ic / 1e-6, **kwargs)
axs[1].plot(x / (2 * np.pi), current1 / 1e-6, **kwargs)
axs[2].plot(x / (2 * np.pi), current2 / 1e-6, **kwargs)

plt.show()
