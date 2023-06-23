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
z_field_xx, rxx, ryx = Loader.load_rxx_ryx()
z_field_yy, ryy, rxy = Loader.load_ryy_rxy()
# trim data 
max_field_clean_data = 9
idx_xx = np.argmin(abs(z_field_xx - max_field_clean_data))
idx_yy = np.argmin(abs(z_field_yy - max_field_clean_data))
z_field_xx = z_field_xx[:idx_xx]
rxx = rxx[:idx_xx]
ryx = ryx[:idx_xx]
z_field_yy = z_field_yy[:idx_yy]
ryy = ryy[:idx_yy]
rxy = rxy[:idx_yy]

density_yx, mobility_xx, _ = get_density_mobility(z_field_xx, ryx, rxx, rxx, Loader.GEOMETRIC_FACTOR, z_field_range=[0, 2])
density_xy, mobility_yy, _ = get_density_mobility(z_field_yy, rxy, ryy, ryy, Loader.GEOMETRIC_FACTOR, z_field_range=[0, 2])
print(r"density_yx = {0:.4f} (10e12 1/cm2)".format(density_yx/1e16))
print(r"density_xy = {0:.4f} (10e12 1/cm2)".format(density_xy/1e16))
print(r"mobility_xx = {0:.0f} (cm2/Vs)".format(mobility_xx*1e4))
print(r"mobility_yy = {0:.0f} (cm2/Vs)".format(mobility_yy*1e4))

num_traces = 4
colors = cmap(np.linspace(0, 1, num_traces))
# plot rxx and ryy 
fig, ax = plt.subplots(1, 1, dpi=600)
ax.plot(z_field_xx, rxx,
    color=colors[0],
    linewidth=1.0,
    alpha=1.0, 
    label=r"$R_{xx}$, $\mu_{xx}=$" + r"{0:.3e} (cm$^2$/V.s)".format(mobility_xx*1e4))
ax.plot(z_field_yy, ryy,
    color=colors[3],
    linewidth=1.0,
    alpha=1.0,
    label=r"$R_{yy}$, $\mu_{yy}=$" + r"{0:.3e} (cm$^2$/V.s)".format(mobility_yy*1e4))

ax.set_xlabel(r"$B_z$ (T)")
ax.set_ylabel(r"$R_{4\mathrm{t}}$ ($\Omega$)")
ax.set_title(r"JS712-HB5, Longitudinal resistance")
ax.set_xlim([z_field_xx[0], z_field_xx[-1]])
ax.legend()
fig.set_size_inches(5, 3.33)
path = "large_field_sweep_longitudinal.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)

# plot rxy and ryx
fig, ax = plt.subplots(1, 1, dpi=600)
ax.plot(z_field_yy, rxy/1000,
    color=colors[1],
    linewidth=1.0,
    alpha=1.0, 
    label=r"$R_{xy}$, " + r"$n = ${0:.2e} (1/cm$^2$)".format(density_xy/1e4))
ax.plot(z_field_xx, ryx/1000,
    color=colors[2],
    linewidth=1.0,
    alpha=1.0,
    label=r"$R_{yx}$, " + r"$n = ${0:.2e} (1/cm$^2$)".format(density_yx/1e4))

ax.set_xlabel(r"$B_z$ (T)")
ax.set_ylabel(r"$R_{4\mathrm{t}}$ ($k\Omega$)")
ax.set_title(r"JS712-HB5, Hall resistance")
ax.set_xlim([z_field_xx[0], z_field_xx[-1]])
ax.legend()
fig.set_size_inches(5, 3.33)
path = "large_field_sweep_hall.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
