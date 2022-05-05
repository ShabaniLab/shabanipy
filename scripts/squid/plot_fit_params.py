"""Plot the fit parameters of the SQUID model.

This requires the output from `fit_squid_oscillations.py`.
"""
import argparse
import configparser
import json
from pathlib import Path

import numpy as np
from lmfit.model import load_modelresult
from matplotlib import pyplot as plt

from shabanipy.squid.lmfitmodels import squid_model

# set up the command-line interface
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--branch",
    "-b",
    choices=["+", "-", "+-"],
    default="+",
    help="plot params for positive (+), negative (-), or simultaneous (+-) branch fits",
)
args = parser.parse_args()

# load the config file
with open(Path(__file__).parent / args.config_path) as f:
    print(f"Using config file `{f.name}`")
    ini = configparser.ConfigParser()
    ini.read_file(f)
    config = ini[args.config_section]

# input/output dir here is output dir of fit_squid_oscillations.py
IODIR = (
    f"{config['WAFER']}-{config['PIECE']}_{config['LAYOUT']}/fits/{config['DEVICE']}"
)
print(f"All output will be saved to `{IODIR}`")
assert Path(IODIR).exists(), "There are no fits to work on..."
OUTPATH = Path(IODIR) / args.config_section

x = []
modelresults = []
for section in json.loads(config["sections"]):
    if "FILTER_VALUE" in ini[section]:
        value = ini.getfloat(section, "FILTER_VALUE")
        filter_str = f"_{value}"
    else:
        value = ini.getfloat(section, "SWEEP_VALUE")
        filter_str = ""
    x.append(value)
    path = Path(IODIR) / (
        f"{ini.get(section, 'COOLDOWN')}-"
        f"{ini.get(section, 'SCAN')}"
        f"{filter_str}_{args.branch}_modelresult.json"
    )
    modelresults.append(load_modelresult(path, funcdefs={"squid_model": squid_model}))

for pname in modelresults[0].params.keys():
    y = [mr.params[pname].value for mr in modelresults]
    yerr = [mr.params[pname].stderr for mr in modelresults]
    yerr = [np.nan if err is None else err for err in yerr]
    fig, ax = plt.subplots()
    ax.set_xlabel(config.get("XLABEL"))
    ax.set_ylabel(pname)
    ax.errorbar(x, y, yerr=yerr, fmt=".--")
    fig.savefig(str(OUTPATH) + f"_{pname}.png")

plt.show()
