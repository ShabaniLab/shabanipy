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
z_field, gate, ryy, ryx = Loader.load_fan()

density, _, mobility_yy = get_density_mobility(z_field.T, ryx.T, ryy.T, ryy.T, Loader.GEOMETRIC_FACTOR, z_field_range=[0, 2])
# for i in range(np.shape(gate)[1]):
#     print(r"gate = {0:.2f} (V)".format(gate[0, i]))
#     print(r"density = {0:.4f} (10e12 1/cm2)".format(density[i]/1e16))
#     print(r"mobility_yy = {0:.0f} (cm2/Vs)".format(mobility_yy[i]*1e4))

num_traces = np.shape(gate)[1]
colors = cmap(np.linspace(0, 1, num_traces))
# plot ryy
fig, ax = plt.subplots(1, 1, dpi=600)
shift = 1.0
for i in range(num_traces):
    # normalize 
    r_range = np.max(ryy[:, i]) - np.min(ryy[:, i])
    r = (ryy[:, i] - ryy[0, i]) / r_range + i*shift
    ax.plot(z_field[:, i], r, 
        color=colors[i],
        linestyle='-', 
        linewidth=0.5)
    ax.annotate(r" {0:.2f} V".format(gate[0, i]), xy=(0, r[0]))

ax.set_xlabel(r"$B_z$ (T)")
ax.set_ylabel(r"Normalized $R_{yy}$")
ax.set_title(r"JS712-HB1, Gate dependeence")
plt.yticks([])
ax.set_xlim([z_field[0, 0], z_field[-1, 0]])
fig.set_size_inches(3.33, 6)
path = "fan_chart_yy.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)

# plot ryx
fig, ax = plt.subplots(1, 1, dpi=600)
shift = 1.0
for i in range(num_traces):
    ax.plot(z_field[:, i], (ryx[:, i] - ryx[0, i])/1e3, 
        color=colors[i],
        linestyle='-', 
        linewidth=0.5)

ax.set_xlabel(r"$B_z$ (T)")
ax.set_ylabel(r"$R_{yx}$ (k$\Omega$)")
ax.set_title(r"JS712-HB1, Gate dependeence")
ax.set_xlim([z_field[0, 0], z_field[-1, 0]])
fig.set_size_inches(3.33, 3.33)
path = "fan_chart_yx.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
