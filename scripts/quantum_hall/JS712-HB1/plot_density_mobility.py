import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colormaps
from scipy import constants
from shabanipy.utils.plotting import jy_pink
jy_pink.register()
from load_data import Loader
from shabanipy.quantum_hall.wal.utils import recenter_wal_data, compute_dephasing_time, compute_linear_soi
from shabanipy.quantum_hall.conversion import htr_from_mobility_density, mean_free_time_from_mobility, kf_from_density, diffusion_constant_from_mobility_density
from get_density_mobility import get_density_mobility

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True
cmap = colormaps['jy_pink']

# load data 
z_field, gate, ryy, ryx = Loader.load_density()

density, _, mobility_yy = get_density_mobility(z_field.T, ryx.T, ryy.T, ryy.T, Loader.GEOMETRIC_FACTOR, z_field_range=[0, z_field[-1, 0]])

num_traces = np.shape(gate)[1]
colors = cmap(np.linspace(0, 1, num_traces))
fig, ax = plt.subplots(1, 1, dpi=600)
# density 
ax.scatter(gate[0, :], density/1e16, 
    color=colors[0],
    linestyle='None', 
    marker='^',
    s=8,
    edgecolors='none',
    alpha=1.0, 
    label=r"Density")
ax.scatter([], [], marker='o', s=7, color=colors[-1], label="Mobility")
ax.legend(loc=4)
ax2 = ax.twinx()
ax2.scatter(gate[0, :], mobility_yy*1e4,
    color=colors[-1],
    linestyle='None', 
    marker='o',
    s=8,
    edgecolors='none',
    alpha=1.0)

ax.set_xlabel(r"$V_{\mathrm{g}}$ (V)")
ax.set_ylabel(r"$n$ (10$^{12}$ 1/cm$^2$)")
ax2.set_ylabel(r"$\mu$ (cm$^2$/V.s)")
ax.set_title(r"JS712-HB1")
# ax.set_xlim([gate[0, 0], gate[0, -1]])
fig.set_size_inches(3.33, 3.33)
path = "density_mobility.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)

idx = np.argmin(np.abs(gate[0, :] - 0.0))
print(f"n = {density[idx]/1e16:.2f} 1e12/cm^2")
print(f"mu = {mobility_yy[idx]*1e4:.2f} cm^2/V.s")