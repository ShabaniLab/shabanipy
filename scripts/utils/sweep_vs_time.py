"""Plot the evolution of a swept variable over time.

For a given step channel (e.g. magnetic field), plot the channel's value as a function
of time (in arbitrary "step" units).
"""
import argparse
from datetime import datetime
from itertools import chain
from pathlib import Path
from pprint import pformat

import pandas as pd
from matplotlib import pyplot as plt

from shabanipy.labber import ShaBlabberFile, get_data_dir
from shabanipy.utils import get_output_dir, stamp

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "channel",
    help="name of the channel to plot vs. time",
)
parser.add_argument(
    "tstart",
    help="initial time in ISO format: 'YYYY-MM-DD HH:MM:SS'",
)
parser.add_argument(
    "tstop",
    help="final time in ISO format: 'YYYY-MM-DD HH:MM:SS'",
)
args = parser.parse_args()
t0, tf = map(datetime.fromisoformat, (args.tstart, args.tstop))
t0, tf = map(lambda minmax: minmax(t0, tf), (min, max))

plt.style.use("fullscreen13")

outdir = get_output_dir() / Path(__file__).stem
outdir.mkdir(parents=True, exist_ok=True)
print(f"output directory: {outdir}")

root = get_data_dir()
print(f"collecting datafiles between {t0} and {tf} in {root}")
datafiles = []
for t in pd.date_range(t0, tf, freq="D"):
    for path in (root / t.strftime("%Y/%m/Data_%m%d")).glob("*.hdf5"):
        datafiles.append(ShaBlabberFile(path))

datafiles = filter(
    lambda f: t0 <= f._creation_time and f._creation_time <= tf, datafiles
)
datafiles = sorted(datafiles, key=lambda f: f._creation_time)

print(
    f"processing datafiles in datetime order:\n"
    f"{pformat([Path(f.filename).name for f in datafiles])}"
)
data = []
for f in datafiles:
    try:
        (d,) = f.get_data(args.channel, sort=False, flatten=True)
    except ValueError:
        d = [f.get_fixed_value(args.channel)]
    data.append(d)

fig, ax = plt.subplots()
ax.set_xlabel("time (steps)")
ax.set_ylabel(args.channel)
ax.set_xticklabels([])
stamp(ax, f"{root.stem}/{t0} $-$ {tf}")
ax.plot(list(chain(*data)), ".-", markersize=5, lw=1)
fmt = "%Y%m%dT%H%M"
fig.savefig(
    outdir / f"{root.stem}_{args.channel}_{t0.strftime(fmt)}-{tf.strftime(fmt)}.png"
)
plt.show()
