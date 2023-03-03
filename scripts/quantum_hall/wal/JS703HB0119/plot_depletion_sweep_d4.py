import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm

from shabanipy.utils.plotting import jy_pink
jy_pink.register()
from load_data import Loader

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True
cmap = cm.get_cmap('jy_pink')
colors = cmap([0, 0.5, 1])

gate, ryy = Loader.load_depletion_sweep_d4()

# plot full range
fig, ax = plt.subplots(1, 1, dpi=600)
ax.scatter(gate, ryy/1e3,
            color='#541688',
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0)
ax.set_xlabel(r"$V_D$ (V)")
ax.set_ylabel(r"Resistance $R_{4t}$ (k$\Omega$)")
ax.set_xlim([gate[np.argmin(gate)], gate[np.argmax(gate)]])
ax.set_title("D4, Depletion gate sweep")

fig.set_size_inches(3.33, 3.33)
path = "d4_depletion_sweep.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")

# plot zoomed in
idx_start = np.argmin(abs(gate - (-1.5)))
idx_finish = np.argmin(abs(gate - (0)))
ax.set_xlabel(r"")
ax.set_ylabel(r"")
ax.set_xlim([gate[idx_start], gate[idx_finish]])
ax.set_ylim([ryy[idx_finish]/1e3, ryy[idx_start]/1e3])
ax.set_title("Linear regime")
fig.set_size_inches(2, 2)
path = "d4_depletion_sweep_linear.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
