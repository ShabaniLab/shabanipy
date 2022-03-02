import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from shabanipy.plotting import jy_pink

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True

class Template(object):
    
    def __init__(self, sample_name=""):
        self.fig = None
        self.ax = None
        self.cmap = cm.get_cmap('jy_pink')
        self.sample_name = sample_name
        self.make_new_figure()

    def make_new_figure(self, sample_name=None):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(1, 1, dpi=600)
        if sample_name is not None:
            self.sample_name = sample_name

    def plot_rashba(self, density, rashba, path):
        colors = self.cmap([0, 0.5, 1.0])
        self.ax.scatter(density/1.0e16, rashba,
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0)
        self.ax.set_xlabel(r"Density n (10$^{12}$/cm$^2$)")
        self.ax.set_ylabel(r"Rashba strength $\alpha$ (meV.$\AA$)")
        self.ax.set_xlim([density[0]/1e16, density[-1]/1e16])
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)    

    def plot_rashba_dresselhaus(self, density, rashba, linear_dresselhaus, path):
        colors = self.cmap([0, 0.5, 1.0])
        self.ax.scatter(density/1.0e16, rashba,
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0,
            label='Rashba')
        self.ax.scatter(density/1.0e16, linear_dresselhaus,
            color=colors[1],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0,
            label='linear Dresselhaus')
        self.ax.legend()
        self.ax.set_xlabel(r"Density n (10$^{12}$/cm$^2$)")
        self.ax.set_ylabel(r"Spin-orbit strength $\alpha$ (meV.$\AA$)")
        self.ax.set_xlim([density[0]/1e16, density[-1]/1e16])
        self.ax.set_ylim([0, max(rashba)])
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)    

    def plot_transport(self, gate, density, mobility, path):
        colors = self.cmap([0, 0.5, 1.0])
        self.ax.scatter(gate, density/1.0e16, 
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0)
        ax2 = self.ax.twinx()
        ax2.scatter(gate, mobility*10,
            color=colors[2],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0)
        self.ax.set_xlabel(r"Gate Voltage $V_g$ (V)")
        self.ax.set_ylabel(r"Density n (10$^{12}$/cm$^2$)")
        ax2.set_ylabel(r"Mobility $\mu$ (10$^3$ cm$^2$/V.s)")
        self.ax.set_xlim([gate[0], gate[-1]])
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)    

    def plot_parallel_zeeman(self, electric_field, parallel_zeeman, path, parallel_zeeman_fit=None, z_41=0):
        colors = self.cmap([0, 0.5, 1.0])
        if parallel_zeeman_fit is not None:
            self.ax.plot(electric_field, parallel_zeeman_fit, 
                color=colors[1],
                linewidth=1.0, 
                label=r"fit, $z_{41}^{6c6c}=$" + r"{0:.2f} ($\AA^3/\Omega$)".format(z_41))
        self.ax.scatter(electric_field, varialbe_zeeman,
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0,
            label=r"estimated from WAL")
        # self.ax.set_xlim([density[0], density[-1]])
        self.ax.legend(frameon=False, fontsize=7, loc=2)
        self.ax.set_xlabel(r"Electric field $\mathcal{E}_z$ (V/$\AA$)")
        self.ax.set_ylabel(r"$g_\parallel \mu_B$")
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)    


    def plot_variable_zeeman(self, electric_field, varialbe_zeeman, path, variable_zeeman_fit=None, z_41=0):
        colors = self.cmap([0, 0.5, 1.0])
        if variable_zeeman_fit is not None:
            self.ax.plot(electric_field, variable_zeeman_fit, 
                color=colors[1],
                linewidth=1.0, 
                label=r"fit, $z_{41}^{6c6c}=$" + r"{0:.2f} ($\AA^3/\Omega$)".format(z_41))
        self.ax.scatter(electric_field, varialbe_zeeman,
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0,
            label=r"estimated from WAL")
        # self.ax.set_xlim([density[0], density[-1]])
        self.ax.legend(frameon=False, fontsize=7, loc=2)
        self.ax.set_xlabel(r"Electric field $\mathcal{E}_z$ (V/$\AA$)")
        self.ax.set_ylabel(r"$z_{41}^{6c6c}\mathcal{E}_z$ (meV/T)")
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)    
    
    def plot_gfactor_angles(self, density, gfactor, angles, path):
        colors = self.cmap(np.linspace(0, 1, len(angles)))
        for i, angle in enumerate(angles):
            self.ax.scatter(density/1.0e16, gfactor[:, i],
                color=colors[i], 
                linestyle='solid',
                marker='o',
                s=5,
                alpha=1.0,
                label=r"{0:.0f}".format(angle))
        self.ax.legend()
        self.ax.set_xlabel(r"Density $10^{12}$ 1/cm$^{2}$")
        self.ax.set_ylabel(r'effective $g$ factor')
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)

    def plot_gfactor(self, density, gfactor, path):
        colors = self.cmap([0, 0.5, 1.0])
        self.ax.scatter(density/1.0e16, gfactor,
            color=colors[0],
            linestyle='None', 
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0)
        # self.ax.set_xlim([density[0], density[-1]])
        self.ax.set_xlabel(r"Density $10^{12}$ 1/cm$^{2}$")
        self.ax.set_ylabel(r'effective $g$ factor')
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)
    
    def plot_zeeman_angle(self, angle, zeeman, gfactor, field_magnitude, density, path, zeeman_fit=None):
        colors = self.cmap([0, 0.5, 1.0])
        if zeeman_fit is not None:
            self.ax.plot(angle, zeeman_fit, 
                color=colors[1],
                linewidth=1.0, 
                label=r"Anisotropic Zeeman Model")
        self.ax.scatter(angle, zeeman, 
            color=colors[0],
            marker='o',
            s=7,
            edgecolors='none',
            alpha=1.0,
            label=r"Zeeman from WAL fits")    
        ax2 = self.ax.twinx()
        ax2.scatter(angle, gfactor, alpha=0.0)
        ax2.set_ylabel(r"Effective $g$ factor")
        min_y = np.min(zeeman)*0.9
        max_y = np.max(zeeman)*1.2
        self.ax.set_xlim([angle[0], angle[-1]])
        self.ax.set_ylim([min_y*0.9, max_y])
        self.ax.legend(frameon=False, fontsize=7, loc=1)
        self.ax.set_xlabel(r"$\angle B_\parallel$ (deg)")
        self.ax.set_ylabel(r"Zeeman energy $g\mu_\mathrm{B}B_\parallel$ (meV)")
        self.ax.set_title(r"{0}, $B_\parallel=${1:.2f} T, n = {2:.2e} 1/cm$^2$".format(self.sample_name, field_magnitude, density/1e4))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)


    def plot_zeeman_angle_deprecated(self, angle, zeeman, gfactor, field_magnitude, density, path, zeeman_fit=None, g_inplane=0, g_variable=0, z_41=0):
        colors = self.cmap([0, 0.5, 1.0])
        if zeeman_fit is not None:
            self.ax.plot(angle, zeeman_fit, 
                color=colors[1],
                linewidth=1.0, 
                label=r"Anisotropic Zeeman Model")
        self.ax.scatter(angle, zeeman, 
            color=colors[0],
            marker='o',
            s=7,
            edgecolors='none',
            alpha=1.0,
            label=r"Zeeman from WAL fits")    
        ax2 = self.ax.twinx()
        ax2.scatter(angle, gfactor, alpha=0.0)
        ax2.set_ylabel(r"Effective $g$ factor")
        min_y = np.min(zeeman)*0.9
        max_y = np.max(zeeman)*1.2
        self.ax.annotate(r"$g_\parallel$={0:.1f}".format(g_inplane) + r", $g_\varepsilon=z_{41}\varepsilon_z/\mu_B=$" + r" {0:.1f}".format(g_variable) + r", $z_{41}=$" + r"{0:.2f} $\AA^3/\Omega$".format(z_41), xy=(0, min_y))
        self.ax.set_xlim([angle[0], angle[-1]])
        self.ax.set_ylim([min_y*0.9, max_y])
        self.ax.legend(frameon=False, fontsize=7, loc=1)
        self.ax.set_xlabel(r"$\angle B_\parallel$ (deg)")
        self.ax.set_ylabel(r"Zeeman energy $g\mu_\mathrm{B}B_\parallel$ (meV)")
        self.ax.set_title(r"{0}, $B_\parallel=${1:.2f} T, n = {2:.2e} 1/cm$^2$".format(self.sample_name, field_magnitude, density/1e4))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)

    def plot_zeeman(self, inplane, zeeman, zeeman_std, zeeman_fit, gfactor, path):
        colors = self.cmap([0, 0.5, 1.0])
        self.ax.errorbar(inplane, zeeman, zeeman_std,
            color=colors[0],
            linestyle='None', 
            marker='o',
            # s=3,
            # edgecolors='none',
            alpha=1.0,
            label=r"WAL fits")
        self.ax.plot(inplane, zeeman_fit,
            color=colors[1],
            linewidth=1.0, 
            label=r"$g=${0:.2f}".format(gfactor))
        self.ax.legend(frameon=False, fontsize=7, loc=1)
        self.ax.set_xlim([0, inplane[-1]])
        # self.ax.set_ylim([min_y, max_y])
        self.ax.set_xlabel(r'$B_\parallel$ (T)')
        self.ax.set_ylabel(r'Zeeman energy $g\mu_\mathrm{B}B_\parallel$ (meV)')
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)

    def plot_single(self, field, conductivity, density, mobility, mean_free_path, transport_field, phi, alpha, beta, gamma, path, conductivity_fit=None, inplane=0.0, angle_label=False):
        colors = self.cmap([0, 0.5, 1.0])
        min_y = -5
        max_y = 5
        coordinate_x = field[0] * 1000 + 10*(field[-1] - field[0])
        if conductivity_fit is not None:
            self.ax.plot(field * 1000, conductivity_fit,
                color=colors[1],
                linewidth=1.0, 
                label=r"model")
        self.ax.scatter(field * 1000, conductivity, 
            color=colors[0],
            marker='o',
            s=3,
            edgecolors='none',
            alpha=1.0,
            label=r"data")
        self.ax.axvline(transport_field * 1000, min_y, max_y, color='gray', alpha=0.4, label=r'$B_\mathrm{tr}$')
        self.ax.axvline(-transport_field * 1000, min_y, max_y, color='gray', alpha=0.4)
        self.ax.annotate(r"$n = {0:.2e}$".format(density/1e4) + r" cm$^{-2}$", xy=(coordinate_x, max_y - 0.5))
        self.ax.annotate(r"$\mu = {0:.2e}$ cm$^2$/V.s".format(mobility*1e4), xy=(coordinate_x, max_y - 1))
        self.ax.annotate(r"$\ell = {0:.0f}$ nm".format(mean_free_path/1e-9), xy=(coordinate_x, max_y - 1.5))
        self.ax.annotate(r"$\ell_\phi = {0:.2f} \mu$m".format(phi), xy=(coordinate_x, max_y - 2))
        self.ax.annotate(r'$\alpha = {0:.2f}$ meV.$\AA$'.format(alpha), xy=(coordinate_x, max_y - 2.5))
        self.ax.annotate(r'$\beta = {0:.2f}$ meV.$\AA$'.format(beta), xy=(coordinate_x, max_y - 3))
        self.ax.annotate(r'$\gamma = {0:.3f}$ eV.$\AA^3$'.format(gamma), xy=(coordinate_x, max_y - 3.5))
        if angle_label:
            self.ax.annotate(r'$\angle B_\parallel = {0:.0f}$'.format(inplane) + r"$^{\circ}$", xy=(coordinate_x, max_y - 4))
        else:    
            self.ax.annotate(r'$B_\parallel = {0:.3f}$ T'.format(inplane), xy=(coordinate_x, max_y - 4))
        self.ax.legend(frameon=False, fontsize=7, loc=1)
        self.ax.set_xlim([field[0] * 1000, field[-1] * 1000])
        self.ax.set_ylim([min_y, max_y])
        self.ax.set_xlabel(r'$B_\perp$ (mT)')
        self.ax.set_ylabel(r'MagnetoConductivity $\Delta \sigma$ (e$^2/4\pi^2\hbar$)')
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.fig.set_size_inches(3.33, 3.33)
        self._save_figure(path)
                
    
    def plot_waterfall(self, field, conductivity, density_or_inplane, path, conductivity_fit=None, third_variable='density'):
        range_conductivity = np.max(conductivity, axis=1) - np.min(conductivity, axis=1)
        colors = self.cmap(np.linspace(0, 1, int(len(density_or_inplane))))
        for i, n in enumerate(density_or_inplane):
            f = field[i, :]
            s = conductivity[i, :]
            shift = - min(s) + np.sum(0.4*range_conductivity[0:i])
            s += shift
            if conductivity_fit is not None:
                if i == 0:
                    self.ax.plot(f, conductivity_fit[i, :] + shift,
                        color='k',
                        linewidth=1.0, 
                        label=r"model")
                else:
                    self.ax.plot(f, conductivity_fit[i, :] + shift,
                        color='k',
                        linewidth=1.0)
            if i == 0 or i == len(density_or_inplane) - 1:
                if third_variable == 'density':    
                    self.ax.scatter(f, s, 
                        color=colors[i],
                        marker='o',
                        s=5,
                        edgecolors='none',
                        alpha=1.0,
                        label=r"$n = ${0:.2e}".format(n/1e4) + " cm$^{-2}$")
                elif third_variable == 'inplane':
                    self.ax.scatter(f, s, 
                        color=colors[i],
                        marker='o',
                        s=5,
                        edgecolors='none',
                        alpha=1.0,
                        label=r"$B_\parallel = ${0:.2f} T".format(n))
                else:
                    self.ax.scatter(f, s, 
                        color=colors[i],
                        marker='o',
                        s=5,
                        edgecolors='none',
                        alpha=1.0,
                        label=r"$\angle B_\parallel = ${0:.2f}".format(n) + r"$^{\circ}$")
            else:
                self.ax.scatter(f, s, 
                    color=colors[i], 
                    marker='o',
                    s=5,
                    edgecolors='none',
                    alpha=1.0)
        
        self.ax.legend(frameon=False, fontsize=7)
        self.ax.set_xlabel(r'$B_\perp$ (mT)')
        self.ax.set_ylabel(r'MagnetoConductivity $\Delta \sigma$ (e$^2/4\pi^2\hbar$)')
        self.ax.set_title(r"{0}".format(self.sample_name))
        self.ax.set_xlim([-35, 35])
        self.fig.set_size_inches(3.33, 5.39)
        self._save_figure(path)
        
    def _save_figure(self, path):
        plt.savefig(path, bbox_inches='tight') 
        print(f"figure saved: {path}")
        plt.close(self.fig)