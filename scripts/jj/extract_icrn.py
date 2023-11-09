"""Extract critical current, normal resistance, and excess current from d.c. VI curve.

At the moment, only 2D data is supported, i.e. V(I,x) where x is some other variable."""
import json
import sys
from pathlib import Path
from pprint import pformat

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from shabanipy.dvdi import extract_iexrn, extract_switching_current
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import ConfArgParser, get_output_dir, git_hash, plot, plot2d

p = ConfArgParser(description=__doc__)
p.add_argument("--datapath", "-dp", help="path to the datafile", required=True)
p.add_argument(
    "--ch_measured",
    help="channel name for the measured d.c. voltage",
    required=True,
)
p.add_argument(
    "--offset",
    help="d.c. voltage offset [raw units] of V(I) curve from 0 V; overrides --offset-npoints",
    type=float,
)
p.add_argument(
    "--offset_npoints",
    help="number of points near 0 bias to average to compute d.c. voltage offset; "
    "not used if --offset is given",
    type=int,
    default=10,
)
p.add_argument(
    "--threshold",
    "-t",
    help="d.c. voltage threshold [raw units, relative to 'offset'] distinguishing superconducting and normal states",
    type=float,
    default=3e-4,
)
p.add_argument(
    "--gain",
    help="gain of amplifier before d.c. voltmeter, used to compute normal resistance",
    type=float,
    default=100,
)
p.add_argument(
    "--ch_source_dc", help="channel name for the d.c. bias source", required=True
)
p.add_argument(
    "--resistor_dc",
    "-r_dc",
    help="resistor [ohms] at d.c. source output, used to convert bias from volts to amps",
    type=float,
    default=1e6,
)
p.add_argument(
    "--bias_min",
    help="current bias [amps] above which V(I) curve is considered ohmic/linear",
    type=float,
    default=15e-6,
)
p.add_argument(
    "--ch_x",
    help="channel name for another independent variable (e.g. gate or field)",
    required=True,  # TODO support single VI traces
)
p.add_argument(
    "--branch",
    "-b",
    help="branch of the VI curve to analyze",
    choices=["+", "-", "+-"],
    default="+-",
)
g = p.add_argument_group(title="Plotting")
g.add_argument(
    "--vmin", help="min value of colorbar scale [ohms]", type=float, default=None
)
g.add_argument(
    "--vmax", help="max value of colorbar scale [ohms]", type=float, default=None
)
args = p.parse_args()
print(f"Running {__file__} with arguments:\n{pformat(vars(args))}")

plt.style.use(["fullscreen13"])

outdir = get_output_dir() / Path(__file__).stem
if args.config_path:
    outdir /= Path(args.config_path).stem
    prefix = args.config_section
else:
    prefix = Path(args.datapath).stem
outdirvv = outdir / "plots"
outdirvv.mkdir(parents=True, exist_ok=True)
outprefix = outdir / prefix
outprefixvv = outdirvv / prefix
print(f"Output directory: {outdir}")

with ShaBlabberFile(args.datapath) as f:
    x, bias, meas = f.get_data(
        args.ch_x,
        args.ch_source_dc,
        args.ch_measured,
        order=(args.ch_x, args.ch_source_dc),
    )
    if args.resistor_dc:
        bias /= args.resistor_dc

args.bias_min = abs(args.bias_min)
if ("+" in args.branch and args.bias_min > bias.max()) or (
    "-" in args.branch and -args.bias_min < bias.min()
):
    raise ValueError(
        f"The given --bias_min ({args.branch}{args.bias_min}) is out of range {bias.min(), bias.max()}"
    )

sides = {"+": "positive", "-": "negative", "+-": "both"}
ic = extract_switching_current(
    bias,
    meas,
    side=sides[args.branch],
    threshold=args.threshold,
    offset=args.offset if args.offset else 0,
    offset_npoints=args.offset_npoints,
    interp=True,
)

stamp = Path(args.datapath).name
fig, ax = plot2d(
    x,
    bias / 1e-6,
    np.gradient(meas / args.gain, axis=-1) / np.gradient(bias, axis=-1),
    xlabel=args.ch_x,
    ylabel="bias current (μA)",
    zlabel="$\Delta V / \Delta I$ (Ω)",
    stamp=stamp,
    vmin=args.vmin,
    vmax=args.vmax,
)
fig.savefig(str(outprefixvv) + "_rawdata.png")
ax.plot(x[:, 0], ic.T / 1e-6 if args.resistor_dc else ic, "k")
fig.savefig(str(outprefixvv) + "_rawdata+ic.png")

# normal resistance and excess current
iex, rn = extract_iexrn(
    bias,
    meas,
    side=sides[args.branch],
    offset=args.offset,
    bias_min=args.bias_min,
)
rn /= args.gain


sign = {"-": -1, "+": 1}
text_kwargs = {"-": {"ha": "left", "va": "top"}, "+": {"ha": "right", "va": "bottom"}}


def plot_analysis(
    ax: plt.Axes,
    i: np.ndarray,  # 1d array of current bias
    v: np.ndarray,  # 1d array of voltage measured
    ic: float,  # critical current
    iex: float,  # excess current
    rn: float,  # normal resistance
    branch: str,  # branch (+ or -)
):
    s = sign[branch]
    i_extreme = s * max(s * i)
    v_extreme = s * min(s * v)
    ax.plot([iex, i_extreme], [0, (i_extreme - iex) * rn], "k--")
    ax_width = np.diff(ax.get_xlim())[0]
    ax_height = np.diff(ax.get_ylim())[0]
    ax.arrow(
        ic,
        -s * ax_height / 6,
        0,
        s * ax_height / 6,
        length_includes_head=True,
        head_width=ax_width / 100,
        head_length=ax_height / 50,
        color="k",
    )
    summary = (
        f"$I_{{ex{branch}}} = {round(iex, 2)}$ μA\n"
        f"$I_{{c{branch}}} = {round(ic, 2)}$ μA\n"
        f"$R_{{n{branch}}} = {round(rn, 2)}$ kΩ\n"
        f"$I_{{ex{branch}}}R_{{n{branch}}} = {round(iex * rn * 1e3)}$ μV\n"
        f"$I_{{c{branch}}}R_{{n{branch}}} = {round(ic * rn * 1e3)}$ μV\n"
    )
    ax.text(i_extreme, v_extreme, summary, **text_kwargs[branch])


for x_, i, v, iex_, rn_, ic_ in zip(
    x, bias / 1e-6, meas / args.gain / 1e-3, iex.T / 1e-6, rn.T / 1e3, ic.T / 1e-6
):
    (x_,) = np.unique(x_)
    fig, ax = plot(
        i,
        v,
        xlabel="bias current (μA)",
        ylabel="voltage (mV)",
        title=f"{args.ch_x} = {x_}",
        stamp=stamp,
    )
    for _ in zip(*[np.atleast_1d(_) for _ in (ic_, iex_, rn_)], args.branch[::-1]):
        plot_analysis(ax, i, v, *_)
    fig.savefig(str(outprefixvv) + f"_{args.ch_x}={x_}_iex-rn.png")

data = {args.ch_x: x[:, 0]}
if args.branch == "+" or args.branch == "-":
    data[f"ic{args.branch}"] = ic
    data[f"iex{args.branch}"] = iex
    data[f"rn{args.branch}"] = rn
else:  # args.branch == "+-"
    data[f"ic-"] = ic[0]
    data[f"ic+"] = ic[1]
    data[f"iex-"] = iex[0]
    data[f"iex+"] = iex[1]
    data[f"rn-"] = rn[0]
    data[f"rn+"] = rn[1]
data["datafiles"] = [Path(args.datapath).stem] * x.shape[0]
data = DataFrame(data)
data.to_csv(str(outprefix) + ".csv", index=False)

metadata = {"git_commit": git_hash(), "command": " ".join(sys.argv), "args": vars(args)}
with open(str(outprefix) + "_metadata.txt", "w") as f:
    f.write(json.dumps(metadata, indent=4))

plt.show()
