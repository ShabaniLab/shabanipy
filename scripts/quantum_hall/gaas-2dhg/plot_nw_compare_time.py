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
cmap = colormaps['jy_pink']

# load data 
z_field_xx, rxx = Loader.get_data(tag='NW-VXX')
z_field_xx_november, rxx_november = Loader.get_data(tag='NW-VXX-NOVEMBER')

colors = cmap(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 1, dpi=600)
# density 
ax.plot(z_field_xx, rxx/1e3, 
    color=colors[0],
    linewidth=0.5,
    label=r"$I=20$ nA, rate $=-500$ $\mu$T/s, June 2023")
ax.plot(z_field_xx_november, rxx_november/1e3, 
    color=colors[-1],
    linewidth=0.5,
    label=r"$I=20$ nA, rate $=-500$ $\mu$T/s, November 2022")
ax.legend()
ax.set_xlim(z_field_xx[0], z_field_xx[-1])
ax.set_xlabel(r"$H_z$ (T)")
ax.set_ylabel(r"$R_{xx}$ (k$\Omega$)")
name = Loader.get_name()
ax.set_title(name + " Device NW")
fig.set_size_inches(5, 5)
path = f"{name.lower()}-nw-compare-time.pdf"
plt.savefig(path, bbox_inches='tight') 
print(f"figure saved: {path}")
plt.close(fig)
