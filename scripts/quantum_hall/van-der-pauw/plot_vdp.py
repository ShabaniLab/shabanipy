import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colormaps
from scipy import constants
from shabanipy.utils.plotting import jy_pink
jy_pink.register()
from loader import Loader
from shabanipy.quantum_hall.wal.utils import recenter_wal_data, compute_dephasing_time, compute_linear_soi
from shabanipy.quantum_hall.conversion import htr_from_mobility_density, mean_free_time_from_mobility, kf_from_density, diffusion_constant_from_mobility_density, GEOMETRIC_FACTORS
from get_density_mobility import get_density_mobility

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
# rcParams['text.usetex'] = True
cmap = colormaps['jy_pink']

# load data 
z_field_xx, rxx = Loader.get_data(tag='VXX')
z_field_yy, ryy = Loader.get_data(tag='VYY')
z_field_xy, rxy = Loader.get_data(tag='VXY')

G = GEOMETRIC_FACTORS['Van der Pauw']
density, mobility_xx, mobility_yy = get_density_mobility(z_field_xx.T, z_field_yy.T, z_field_xy.T, rxx.T, ryy.T, rxy.T, G, z_field_range=[2.0, 4.0])

colors = cmap(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 1, dpi=600)
# density 
ax.plot(z_field_xx, rxx/1e3, 
    color=colors[0],
    linewidth=1,
    label=r"$R_{xx}$" + r", $\mu_{xx} = $" + r"{0:.2e}".format(mobility_xx[0]*1e4) + " cm$^2$/Vs")
ax.plot(z_field_yy, ryy/1e3, 
    color=colors[1],
    linewidth=1,
    label=r"$R_{yy}$" + r", $\mu_{yy} = $" + r"{0:.2e}".format(mobility_yy[0]*1e4) + " cm$^2$/Vs")
ax.plot([], [], 
    color=colors[-1],
    linewidth=1,
    label=r"$R_{xy}$" + r", $n$ = {0:.2e}".format(density[0]/1e4) + " cm$^{-2}$")
ax2 = ax.twinx()
ax2.plot(z_field_xy, rxy/1e3, 
    color=colors[-1],
    linewidth=1)
ax.legend(loc=2)
ax.yaxis.label.set_color(colors[0]) 
ax2.yaxis.label.set_color(colors[-1]) 
# ax2.tick_params(colors=colors[-1])
ax.set_xlim(z_field_xx[0], z_field_xx[-1])
ax.set_xlabel(r"$B_z$ (T)")
ax.set_ylabel(r"$R_{xx}, R_{yy}$ (k$\Omega$)")
ax2.set_ylabel(r"$R_{xy}$ (k$\Omega$)")
name = Loader.get_name()
ax.set_title(name)
fig.set_size_inches(5, 5)
path = f"{name}.pdf"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
