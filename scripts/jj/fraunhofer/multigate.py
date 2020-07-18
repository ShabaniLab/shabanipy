"""Plot multigate Fraunhofers and guess corresponding current densities."""
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast
from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
        extract_theta, extract_current_distribution)
from shabanipy.jj.utils import extract_switching_current
from shabanipy.utils.labber_io import LabberData


LABBER_DATA_DIR = os.environ['LABBER_DATA_DIR']

# Fraunhofer data for multigates -1-2-3-4-5-
# Gates 2 and 4 are fixed at Vg2 = Vg4 = constant while gates 1, 3, and 5 are
# swept together.
# A Fraunhofer pattern is measured for 6 values of Vg1 = Vg3 = Vg5.
DATA_FILE_PATH = (Path(LABBER_DATA_DIR) /
        '2019/11/Data_1104/JS123A_BM003_054.hdf5')

# channel names
CH_MAGNET = 'Keithley Magnet 1 - Source current'
CH_BIAS = 'Yoko 1 - Voltage'    # current bias
CH_GATE = 'SRS - Aux 2 output'  # gate voltage (gates 1, 3, and 5)
CH_RESIST = 'SRS - Value'

# conversion factors
CURR_TO_FIELD = 1e3 / 18.2  # coil current to B-field (in mT)
VOLT_TO_RESIST = 1 / 10e-9  # lock-in voltage to resistance (inverse current)

resist = []
ic = []
with LabberData(str(DATA_FILE_PATH)) as f:
    channels = f.list_channels()

    field = np.unique(f.get_data(CH_MAGNET))[:-10] * CURR_TO_FIELD

    gate = np.unique(f.get_data(CH_GATE))
    for g in gate:
        bias = f.get_data(CH_BIAS, filters={CH_GATE: g})[:-10]
        resist.append(np.abs(np.real(VOLT_TO_RESIST *
            f.get_data(CH_RESIST, filters={CH_GATE: g})[:-10])))
        ic.append(extract_switching_current(bias, resist[-1], 5, 'positive'))

for i, g in enumerate(gate):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(r'$V_{g,odd}$ = ' + f'{g}')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Bias current (µA)')
    im = ax.imshow(resist[i].T, origin='lower',
            extent=(field[0], field[-1], 0, 2))
    cb = fig.colorbar(im)
    cb.ax.set_ylabel('Differential resistance (Ω)')
    ax.plot(field, ic[i], color='white')
    fig.show()
