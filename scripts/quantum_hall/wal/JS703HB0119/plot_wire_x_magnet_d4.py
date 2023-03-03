import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import constants

from load_data import Loader

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True

gate, x_field, ryy = Loader.load_wire_x_field_d4()
# choose every other field value (the ones with the same sweep direction)
gate = gate[:, ::2]
x_field = x_field[::2]
ryy = ryy[:, ::2]
# remove beyond -1.15 V
idx = np.argmin(abs(gate[:, 0] - (-1.15)))
gate = gate[:idx, :]
x_field = x_field[:idx]
ryy = ryy[:idx, :]

fig, ax = plt.subplots(1, 1, dpi=600)
shift = 0.04
for i in range(len(x_field)): 
    ax.plot(gate[:, i], ryy[:, i]/1e3 + i*shift,
                color='k',
                linestyle='-', 
                linewidth=0.25)

ax.set_xlabel(r"$V_w$ (V)")
ax.set_ylabel(r"R (k$\Omega$)")
ax.set_xlim([-1.15, 0])
ax.set_title(r"D4, Resistance vs $x$ Magnetic Field")
# ax.legend(frameon=False)
fig.set_size_inches(3.33, 3.33)
path = "d4_wire_x_field.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
