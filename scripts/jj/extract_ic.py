"""Extract critical current as a function of some variable."""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import ConfArgParser, get_output_dir, git_hash, plot2d

p = ConfArgParser(description=__doc__)
# required arguments
p.add_argument("--datapath", "-dp", help="path to the datafile", required=True)
p.add_argument(
    "--ch_measured",
    help="channel name for the measured voltage (a.c. or d.c.)",
    required=True,
)
p.add_argument("--ch_source_dc", help="channel name for the d.c. source", required=True)
p.add_argument(
    "--ch_variable",
    help="channel name for the independent variable (e.g. gate or field)",
    required=True,
)
# optional arguments
p.add_argument(
    "--threshold",
    "-t",
    help="voltage threshold distinguishing superconducting and normal states",
    type=float,
)
p.add_argument(
    "--offset_npoints",
    help="number of points around 0 bias that are averaged to compute and remove any offset",
    type=int,
)
p.add_argument(
    "--ignore_npoints",
    help="number of points around 0 bias to ignore (e.g. when lock-in hasn't settled)",
    type=int,
)
p.add_argument(
    "--branch",
    "-b",
    help="branch of critical current to extract",
    choices=["+", "-", "+-"],
    default="+",
)
p.add_argument(
    "--resistor_dc",
    "-r_dc",
    help="resistor at d.c. source output, used to convert bias from volts to amps",
    type=float,
)
p.add_argument(
    "--resistor_ac",
    "-r_ac",
    help="resistor at a.c. source output, used to convert measured volts to ohms",
    type=float,
)
p.add_argument(
    "--ch_source_ac",
    help="channel name for the a.c. source, used to convert measured volts to ohms",
)
g = p.add_argument_group(title="Plotting")
g.add_argument("--vmin", help="min value of colorbar scale", type=float, default=None)
g.add_argument("--vmax", help="max value of colorbar scale", type=float, default=None)
g.add_argument(
    "--no-show",
    help="do not plt.show() plots (e.g. for bulk processing)",
    action="store_true",
)
args = p.parse_args()

plt.style.use(["fullscreen13"])

outdir = get_output_dir() / Path(__file__).stem
if args.config_path:
    outdir /= Path(args.config_path).stem
    outprefix = outdir / args.config_section
else:
    outprefix = outdir / Path(args.datapath).stem
outdir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {outdir}")

with ShaBlabberFile(args.datapath) as f:
    var, bias, meas = f.get_data(
        args.ch_variable,
        args.ch_source_dc,
        args.ch_measured,
        order=(args.ch_variable, args.ch_source_dc),
    )
    if args.resistor_dc:
        bias /= args.resistor_dc
    if args.resistor_ac or args.ch_source_ac:
        if not (args.resistor_ac and args.ch_source_ac):
            raise ValueError(
                f"Both resistor_ac ({getattr(args, 'resistor_ac')}) "
                f"and ch_source_ac ({getattr(args, 'ch_source_ac')}) "
                "are required to convert a.c. rms volts to ohms"
            )
        ac_volt = f.get_channel(args.ch_source_ac).instrument.config["Output amplitude"]
        ohms = meas / (ac_volt / args.resistor_ac)

sides = {"+": "positive", "-": "negative", "+-": "both"}
ic = extract_switching_current(
    bias,
    np.abs(meas),
    side=sides[args.branch],
    threshold=args.threshold,
    offset_npoints=args.offset_npoints,
    ignore_npoints=args.ignore_npoints,
)

fig, ax = plot2d(
    var,
    bias / 1e-6 if args.resistor_dc else bias,
    np.abs(ohms) if args.resistor_ac else np.abs(meas),
    xlabel=args.ch_variable,
    ylabel="bias current (μA)" if args.resistor_dc else args.ch_source_dc,
    zlabel="|dV/dI| (Ω)" if args.resistor_ac else args.ch_measured,
    vmin=args.vmin,
    vmax=args.vmax,
    stamp=Path(args.datapath).name,
)
fig.savefig(str(outprefix) + "_rawdata.png")

ax.plot(var[:, 0], ic.T / 1e-6 if args.resistor_dc else ic, "k")
fig.savefig(str(outprefix) + "_rawdata+ic.png")

data = {args.ch_variable: var[:, 0]}
if args.branch == "+" or args.branch == "-":
    data[f"ic{args.branch}"] = ic
else:  # args.branch == "+-"
    data[f"ic-"] = ic[0]
    data[f"ic+"] = ic[1]
data["datafiles"] = [Path(args.datapath).stem] * var.shape[0]
data = DataFrame(data)
data.to_csv(str(outprefix) + ".csv", index=False)

metadata = {"git_commit": git_hash(), "command": " ".join(sys.argv), "args": vars(args)}
with open(str(outprefix) + "_metadata.txt", "w") as f:
    f.write(json.dumps(metadata, indent=4))

if not args.no_show:
    plt.show()
