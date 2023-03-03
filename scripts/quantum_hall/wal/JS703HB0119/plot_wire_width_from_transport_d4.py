import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from scipy import constants

conductance_quantum = constants.e ** 2 / constants.h

from load_data import Loader

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True

gate, ryy = Loader.load_depletion_sweep_d4()
geometric_factor, w, l = Loader.load_geometry_micron_d4()

# remove gate values above -3 V
resistance_0 = ryy[0]
idx = np.argmin(abs(gate - (-2.75)))
gate = gate[idx:]
ryy = ryy[idx:]

# load density and mobility 
with open("density_mobility_d4.txt", "r") as fp:
    data = json.load(fp)
density = float(data['density'])
mobility = float(data['mobility'])

# ballistic 
k_fermi = np.sqrt(2*np.pi*density)
width_ballistic_nm = (1/conductance_quantum/2)/(ryy - resistance_0)*np.pi/k_fermi/1e-9
# diffusive 
resistivity = resistance_0*geometric_factor
width_diffusive_nm = resistivity*l*1e-6/(ryy - resistance_0)/1e-9
# mean free path 
mean_free_path_nm = constants.hbar*k_fermi*mobility/constants.e/1e-9
# plot 
fig, ax = plt.subplots(1, 1, dpi=600)
ax.plot(gate, width_diffusive_nm,
            color='k',
            linestyle='-', 
            linewidth=1.0, 
            label=r'Diffusive $w\gg \ell$')
ax.plot(gate, width_ballistic_nm,
            color='k',
            linestyle='-.', 
            linewidth=1.0, 
            label=r'Ballistic $w\ll \ell$')
ax.plot(gate, gate*0 + mean_free_path_nm,
            color='k',
            linestyle=':', 
            alpha=0.5,
            linewidth=1.0,
            label=r'Mean free path $\ell$')
ax.set_xlabel(r"$V_D$ (V)")
ax.set_ylabel(r"$w$ (nm)")
ax.set_xlim([gate[np.argmin(gate)], gate[np.argmax(gate)]])
ax.set_title("D4, Wire width estimates")
ax.legend(frameon=False)
fig.set_size_inches(3.33, 3.33)
path = "d4_width_from_transport.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
