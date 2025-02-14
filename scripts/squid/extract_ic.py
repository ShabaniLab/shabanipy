"""Extract critical current as a function of some variable(s).

No unit conversions are performed: everything is in terms of raw data units.
"""
from functools import reduce
from pathlib import Path
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from shabanipy.dvdi import extract_switching_current
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import (
    ConfArgParser,
    get_output_dir,
    jy_pink,
    load_config,
    plot2d,
    write_metadata,
)

p = ConfArgParser(description=__doc__)
# required arguments
p.add_argument("config_path", help="path to .ini config file, relative to this script")
p.add_argument("config_section", help="section of the .ini config file to use")

g = p.add_argument_group(title="Plotting")
g.add_argument("--vmin", help="min value of colorbar scale", type=float, default=None)
g.add_argument("--vmax", help="max value of colorbar scale", type=float, default=None)
g.add_argument(
    "--no-show",
    help="do not plt.show() plots (e.g. for bulk processing)",
    action="store_true",
)
args = p.parse_args()
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

outdir = (
    get_output_dir()
    / Path(__file__).stem
    / f"{config['WAFER']}-{config['PIECE']}_{config['LAYOUT']}"
    / config["DEVICE"]
    / f"{config['COOLDOWN']}-{config['SCAN']}"
)
outdir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {outdir}")
outprefix = str(outdir / f"{Path(args.config_path).stem}_{args.config_section}")

write_metadata(outprefix + "_args.json", args)

CH_IVAR = config["CH_IVAR"]  # independent variable x w.r.t. which Ic±(x) is plotted
CH_CURRENT = config["CH_CURRENT"]  # current bias
CH_THRESHOLD = config["CH_THRESHOLD"]  # measured channel to threshold for Ic
with ShaBlabberFile(config["DATAPATH"]) as f:
    ch_swept = [ch for ch in f._sweep_channel_names if ch not in {CH_IVAR, CH_CURRENT}]
    *swept, ivar, ibias, meas = f.get_data(
        *ch_swept, CH_IVAR, CH_CURRENT, CH_THRESHOLD, order=(..., CH_IVAR, CH_CURRENT)
    )
branch = ""
branch += "+" if np.any(ibias > 0) else ""
branch += "-" if np.any(ibias < 0) else ""
if np.any(np.iscomplex(meas)):
    meas = np.real(meas)

sides = {"+": "positive", "-": "negative", "+-": "both"}
ic = extract_switching_current(
    ibias,
    meas,
    side=sides[branch],
    threshold=config.getfloat("THRESHOLD"),
    offset_npoints=config.getint("OFFSET_NPOINTS"),
    ignore_npoints=config.getint("IGNORE_NPOINTS"),
    interp=True,
)
if branch == "+-":
    ic = np.moveaxis(ic, 0, -1)

jy_pink.register()
plt.style.use(["fullscreen13", "jy_pink"])
# TODO support 2d data as well; currently only tested for 3d+
for index in np.ndindex(*ibias.shape[:-2]):
    swept_values = tuple(np.unique(ch[index]).squeeze() for ch in swept)
    swept_str = reduce(lambda x, y: f"{x}_{y}", swept_values, "")
    fig, ax = plot2d(
        ivar[index],
        ibias[index],
        meas[index],
        xlabel=CH_IVAR,
        ylabel=CH_CURRENT,
        zlabel=CH_THRESHOLD,
        vmin=args.vmin,
        vmax=args.vmax,
        stamp=config["DATAPATH"],
        rasterized=True,
    )
    ax.plot(np.unique(ivar[index]), ic[index], "k", lw=0, marker="_", ms=2)
    ax.set_title(f"{ch_swept}={swept_values}", fontsize="xx-small")
    fig.savefig(outprefix + swept_str + "_rawdata+ic.svg")


def collapse(data):
    """Eliminate current bias on the last axis of `data`.

    Check that the sweeps are all identical in the process.
    """
    if np.diff(data, axis=-1).any():
        warn("Collapsing non-indentical sweeps, data will be lost")
    return data[..., 0]


data = {ch: collapse(data).flatten() for ch, data in zip(ch_swept, swept)}
data[CH_IVAR] = collapse(ivar).flatten()
for pm, i in zip(branch, (1, 0)):
    data[f"ic{pm}"] = ic[..., i].flatten()
data["datapath"] = [config["DATAPATH"]] * np.prod(ivar.shape[:-1])
data = DataFrame(data)
data.to_csv(outprefix + "_ic.csv", index=False)

if not args.no_show:
    plt.show()
