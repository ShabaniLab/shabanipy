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
z_field_xx, rxx = Loader.load_rxx()
z_field_yy, ryy = Loader.load_ryy()
z_field_xy, rxy = Loader.load_rxy()
z_field_yx, ryx = Loader.load_ryx()

density, mobility_xx, mobility_yy = get_density_mobility(z_field_xy, rxy, rxx, ryy, Loader.GEOMETRIC_FACTOR, z_field_range=[0, 2])
density_yx, _, _ = get_density_mobility(z_field_yx, ryx, rxx, ryy, Loader.GEOMETRIC_FACTOR, z_field_range=[0, 2])
print(r"density = {0:.4f} (10e12 1/cm2)".format(density/1e16))
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

ax.set_xlabel(r"$H_z$ (T)")
ax.set_ylabel(r"$R_{4\mathrm{t}}$ ($\Omega$)")
ax.set_title(r"JS712-HB4, Longitudinal resistance")
ax.set_xlim([z_field_xx[0], z_field_xx[-1]])
ax.legend()
fig.set_size_inches(5, 3.33)
path = "large_field_sweep_longitudinal.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)

# plot rxy and ryx
fig, ax = plt.subplots(1, 1, dpi=600)
ax.plot(z_field_xy, rxy/1000,
    color=colors[1],
    linewidth=1.0,
    alpha=1.0, 
    label=r"$R_{xy}$, " + r"$n = ${0:.2e} (1/cm$^2$)".format(density/1e4))
ax.plot(z_field_yx, ryx/1000,
    color=colors[2],
    linewidth=1.0,
    alpha=1.0,
    label=r"$R_{yx}$, " + r"$n = ${0:.2e} (1/cm$^2$)".format(density_yx/1e4))

ax.set_xlabel(r"$H_z$ (T)")
ax.set_ylabel(r"$R_{4\mathrm{t}}$ ($k\Omega$)")
ax.set_title(r"JS712-HB4, Hall resistance")
ax.set_xlim([z_field_xy[0], z_field_xy[-1]])
ax.legend()
fig.set_size_inches(5, 3.33)
path = "large_field_sweep_hall.png"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
