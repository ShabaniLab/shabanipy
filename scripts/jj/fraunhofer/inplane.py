import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from shabanipy.dvdi import extract_switching_current
from shabanipy.jj import find_fraunhofer_center
from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import load_config, plot, plot2d

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "config_path", help="path to .ini config file, relative to this script"
)
parser.add_argument("config_section", help="section of the .ini config file to use")
parser.add_argument(
    "--align",
    "-a",
    default=False,
    action="store_true",
    help="plot fraunhofer center vs. in-plane for field alignment",
)
parser.add_argument(
    "--debug",
    "-d",
    default=False,
    action="store_true",
    help="show additional plots for debugging purposes",
)
args = parser.parse_args()
_, config = load_config(Path(__file__).parent / args.config_path, args.config_section)

plt.style.use(["fullscreen13"])

b_inplane = []
fraun_center = []
i = 1
while config.get(f"DATAPATH{i}"):
    ch_bias = config.get(f"CH_BIAS{i}", config["CH_BIAS"])
    ch_meas = config.get(f"CH_MEAS{i}", config["CH_MEAS"])
    with ShaBlabberFile(config[f"DATAPATH{i}"]) as f:
        filter_val = config.getfloat(f"FILTER_VAL{i}")
        if filter_val is not None:
            filters = [
                (
                    config.get(f"FILTER_CH{i}", config["CH_FIELD_INPLANE"]),
                    np.equal,
                    filter_val,
                )
            ]
            inplane = filter_val
        else:
            filters = None
            inplane = f.get_channel(config["CH_FIELD_INPLANE"]).instrument.config[
                config["CH_FIELD_INPLANE"].split(" - ")[-1]
            ]
        b_inplane.append(inplane)
        print(f"Processing {inplane=}")
        b_perp, ibias, meas = f.get_data(
            config["CH_FIELD_PERP"],
            ch_bias,
            ch_meas,
            order=(config["CH_FIELD_PERP"], ch_bias),
            filters=filters,
        )
        if ch_meas.endswith(("VI curve", "SingleValue")):  # DC volts from DMM
            volts = meas / config.getfloat("DC_AMP_GAIN")
            dvdi = np.gradient(volts, axis=-1) / np.gradient(ibias, axis=-1)
            dvdi_label = "ΔV/ΔΙ (Ω)"
        else:  # differential Ω from lock-in
            dvdi = np.abs(meas)
            dvdi_label = "|dV/dI| (Ω)"
            if ch_meas.endswith("Value"):  # lock-in directly (not VICurveTracer)
                ibias_ac = f.get_channel(ch_meas).instrument.config[
                    "Output amplitude"
                ] / config.getfloat("R_AC_OUT")
                dvdi /= ibias_ac
                ibias /= config.getfloat("R_DC_OUT")

    Path("output").mkdir(exist_ok=True)
    outpath = f"output/{Path(config[f'DATAPATH{i}']).stem}_{inplane=}"
    fig, ax = plot2d(
        b_perp / 1e-3,
        ibias / 1e-6,
        dvdi,
        xlabel="out-of-plane field (mT)",
        ylabel="current bias (μA)",
        zlabel=dvdi_label,
        title=f"in-plane field = {round(inplane / 1e-3)} mT",
        stamp=f"{config['FRIDGE']}/{config[f'DATAPATH{i}']}",
        vmin=config.getfloat("VMIN"),
        vmax=config.getfloat("VMAX"),
        extend_min=False,
    )

    b_perp = np.unique(b_perp)

    threshold = config.getfloat(f"THRESHOLD{i}", config.getfloat("THRESHOLD"))
    if config.getboolean(f"THRESHOLD_DC{i}", config.getboolean(f"THRESHOLD_DC")):
        ic_signal = volts
    else:
        ic_signal = dvdi
    ic_p = extract_switching_current(
        ibias, ic_signal, side="positive", threshold=threshold
    )
    savgol_wlen = config.getint(f"SAVGOL_WLEN{i}", config.getint(f"SAVGOL_WLEN", None))
    if savgol_wlen is not None:
        ic_p = savgol_filter(ic_p, savgol_wlen, 2)
    ax.plot(b_perp / 1e-3, ic_p / 1e-6, "k-", lw=1)

    if args.debug:
        plt.show()

    field_lim = config.get(f"FIELD_LIM{i}")
    if field_lim is not None:
        field_lim = tuple(json.loads(field_lim))
    center = find_fraunhofer_center(b_perp, ic_p, field_lim=field_lim, debug=args.debug)
    fraun_center.append(center)
    ax.axvline(center / 1e-3, color="k")
    fig.savefig(outpath + "_fraun-center.png")

    i += 1
b_inplane = np.array(b_inplane)
fraun_center = np.array(fraun_center)

# field alignment
if args.align:
    m, b = np.polyfit(b_inplane, fraun_center, 1)
    fig, ax = plot(
        b_inplane / 1e-3,
        fraun_center / 1e-6,
        "o-",
        xlabel="in-plane field (mT)",
        ylabel="fraunhofer center (μT)",
    )
    ax.plot(
        b_inplane / 1e-3,
        (m * b_inplane + b) / 1e-6,
        label=f"arcsin$(B_\perp/B_\parallel)$ = {round(np.degrees(np.arcsin(m)) / 1e-3, 2)} mdeg",
    )
    ax.legend()
    fig.savefig(outpath + "_field-alignment.png")

plt.show()
