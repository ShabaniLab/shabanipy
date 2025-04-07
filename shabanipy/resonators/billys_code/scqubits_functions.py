import inspect
from numpy import diff

import sys
sys.path.append('/Users/billystrickland/Documents/code/resonators')
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.resonator_functions as rf
from matplotlib.patches import Rectangle
import matplotlib as mpl

from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
plt.style.use(jy_mpl_rc)

import scqubits as scq

def C_to_Ec(x):
    print('EC = ', 19370/x/1000) # gigahertz
    return 19370/x/1000 # gigahertz
def Ec_to_C(x):
    return 1000*x/19370 # fF
def I_to_Ej(x):
    return x*496.7 #gigahertz
def Ej_to_I(x):
    return x/496.7 # microamp

def get_nmatelem(fl):
    n = fl.matrixelement_table('n_operator', evals_count=100)
    return abs(n[0][1])

def get_dEdphi(fl, flux = 0.25):
    flux_list = np.linspace(flux-.00000001, flux+.00000001, 3)
    spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=9, subtract_ground=True);
    E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T
    E_phi_01 = E_phi[1]-E_phi[0]
    dE_dphi = np.diff(E_phi_01)
    return dE_dphi[int(len(dE_dphi)/2)]

def get_enhancements(Ej_h, Ej_l, Ec, El, flux_list = np.linspace(0, 1, 151)): 

    fl_h = scq.Fluxonium(
            EJ=Ej_h,
            EC=Ec,
            EL=El,
            cutoff = 100,
            flux = 0.45
        )
    dE_dphi_h = get_dEdphi(fl_h)
    nh = get_nmatelem(fl_h)
    fl_l = scq.Fluxonium(
            EJ=Ej_l,
            EC=Ec,
            EL=El,
            cutoff = 100,
            flux = 0.5
        )
    nl = get_nmatelem(fl_l)
    dE_dphi_l = get_dEdphi(fl_l)
    print('heavy: <0|n|1>', nh)
    print('light: <0|n|1>', nl)
    print('dE/dphi_h', dE_dphi_h)
    print('dE/dphi_l', dE_dphi_l)
    print('heavy to light: dE/dphi enhancement = ',dE_dphi_h/dE_dphi_l)
    print('heavy to light: <0|n|1> enhancement',nl/nh)
    delta_de = dE_dphi_h/dE_dphi_l
    delta_n = nl/nh
    return delta_de, delta_n

def get_fl(flux_extent, El, Ej, Ec, flux=0):
    flux_list = np.linspace(flux_extent[0],flux_extent[-1], 10)
    fl= scq.Fluxonium(
        EJ=Ej,
        EC=Ec,
        EL=El,
        cutoff = 110,
        flux = flux)
    return fl, flux_list

def plot_spectrum(flux_extent, El, Ej, Ec, ylim = [4, 6], zero2 = True, zero3 = False):

    fl, flux_list = get_fl(flux_extent, El, Ej, Ec)
    spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=20, subtract_ground=True);
    E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T

    f01 = E_phi[1]-E_phi[0]
    f02 = E_phi[2]-E_phi[0]
    f03 = E_phi[3]-E_phi[0]
    
    plt.plot(flux_list, f01, color=line_colors[1], label='$f_{01}$', linestyle='--', dashes=(5, 5))
    if zero2:
        plt.plot(flux_list, f02, color=line_colors[8], label='$f_{02}$', linestyle='--', dashes=(5, 5))
    if zero3:    
        plt.plot(flux_list, f03, color=line_colors[3], label='$f_{03}$', linestyle='--', dashes=(5, 5))
    plt.xlabel('$\Phi/\Phi_0$')
    plt.legend()
    

def get_f01(El, Ej, Ec, phi):
    fl, flux_list = get_fl([phi, phi+.000001], El, Ej, Ec)
    spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=20, subtract_ground=True);
    E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T

    f01 = E_phi[1]-E_phi[0]
    f02 = E_phi[2]-E_phi[0]
    f03 = E_phi[3]-E_phi[0]
    f04 = E_phi[4]-E_phi[0]
    return f01[0], f02[0], f03[0], f04[0]
    
def save_fig(fig, name, format='eps'):
    fig.tight_layout()
    folder_path = 'figs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/{name}.{format}', format=format)
    
def plot_energies(flux_extent, El, Ej, Ec, 
                  phi = 0.5,
                  ylim = [4, 6], 
                  zero2 = True, 
                  zero3 = False,
                 figsize = (6, 4),
#                  linestyle = '--'),
                dashes = (5, 5),
                 eval_num = 20,
                 flux_res = 151):
    flux_list = np.linspace(flux_extent[0],flux_extent[-1], flux_res)
    fl= scq.Fluxonium(
        EJ=Ej,
        EC=Ec,
        EL=El,
        cutoff = 110,
        flux = phi)

    spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=eval_num, subtract_ground=True);
    E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T
    plt.plot(flux_list, E_phi[0])
    plt.plot(flux_list, E_phi[1])
    plt.plot(flux_list, E_phi[2])
    plt.plot(flux_list, E_phi[3])
    
    
def plot_spectrum(flux_extent, El, Ej, Ec, 
                  phi = 0.5,
                  ylim = [4, 6], 
                  zero2 = True, 
                  zero3 = False,
                 figsize = (6, 4),
#                  linestyle = '--'),
                dashes = (5, 5),
                 eval_num = 20,
                 flux_res = 151):
    flux_list = np.linspace(flux_extent[0],flux_extent[-1], flux_res)
    fl= scq.Fluxonium(
        EJ=Ej,
        EC=Ec,
        EL=El,
        cutoff = 110,
        flux = phi)

    spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=eval_num, subtract_ground=True);
    E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T

    f01 = E_phi[1]-E_phi[0]
    f02 = E_phi[2]-E_phi[0]
    f03 = E_phi[3]-E_phi[0]
    fig, ax = plt.subplots(figsize = figsize)
    plt.plot(flux_list, f01, color=line_colors[1], label='$f_{01}$', linestyle='--', dashes=dashes)
    if zero2:
        plt.plot(flux_list, f02, color=line_colors[8], label='$f_{02}$', linestyle='--', dashes=dashes)
    if zero3:    
        plt.plot(flux_list, f03, color=line_colors[3], label='$f_{03}$', linestyle='--', dashes=dashes)
    plt.xlabel('$\Phi/\Phi_0$')
    plt.ylabel('$f_{ij}$ (GHz)')
    ax.set_xlim([-1, 1])
    plt.legend()
    return fl, fig, ax

def anharm(El, Ej, Ec, phi):
    f01, f02, _ = get_f01(El, Ej, Ec, phi)
    return 2*np.array(f01)-np.array(f02)

# anharm(El, Ec, Ej, 0)
def get_Ej(freq, freq_array, Ej_array):
    """
    Given a frequency freq and two arrays freq_array and Ej_array, 
    returns the corresponding Ej value based on the frequency.
    """
    # Find the index of the closest frequency in freq_array
    closest_index = min(range(len(freq_array)), key=lambda i: abs(freq_array[i] - freq))
    
    # Return the corresponding Ej value
    return Ej_array[closest_index]

def plot_energies(flux_extent, El, Ej, Ec, 
                  phi = 0.5,
                  ylim = [4, 6], 
                  zero2 = True, 
                  zero3 = False,
                 figsize = (6, 4),
#                  linestyle = '--'),
                dashes = (5, 5),
                 eval_num = 20,
                 flux_res = 151):
    flux_list = np.linspace(flux_extent[0],flux_extent[-1], flux_res)
#     Ej = np.logspace(-1, 1, 5)
    Ej = np.linspace(.1, 10, 5)
    Ej = Ej[::-1]
    colors = ['#FF9999', '#DC143C', '#8B0000', '#DC143C', '#800000']
    for i, x in enumerate(Ej):
        fl= scq.Fluxonium(
            EJ=x,
            EC=Ec,
            EL=El,
            cutoff = 110,
            flux = phi)

        spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=eval_num, subtract_ground=True);
        E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T
        plt.plot(flux_list, E_phi[1]-E_phi[0], label = r'$E_J$ = '+str(round(x, 1)) +' GHz', color = colors[i])
#     plt.ylim(0, 10)
    plt.legend(bbox_to_anchor=(1, .8))
    return fl, fig, ax
