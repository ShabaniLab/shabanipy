import sys
sys.path.append('/Users/billystrickland/Documents/code/resonators')
import matplotlib.pyplot as plt
from shabanipy.labber import LabberData
import numpy as np
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors

from scipy.optimize import curve_fit
import cmath
import utils.resonator_functions as rf
import utils.qubit_functions as qf
import scqubits as scq
import os
from scipy import linalg as la
from tqdm.notebook import tqdm
from typing import List, Tuple
import os
import inspect


def I_to_Ej(x):
    return x*496.7 #gigahertz

def save_fig(fig, name):
    fig.tight_layout()
    folder_path = 'figs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(f'{folder_path}/{name}.eps', format='eps')

plt.style.use(jy_mpl_rc)

def extract_data(sample, file_num, month, day, freq_slice=3):
    savepath = '/Users/billystrickland/Documents/code/resonators/data/'+sample+'/'
    os.makedirs(savepath, exist_ok=True)
    year = '2024'
    root = f'/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/{year}/'+month+'/Data_'+month+day
    FILE = root+'/'+sample+'-'+file_num+'.hdf5'
    P_CH, S21_CH, DF_CH = ['Qcage - Magnet - Source voltage', 'VNA - S21', 'SC1 - Frequency']

    with LabberData(FILE) as f:
        magnet = f.get_data(P_CH)
        freq, data = f.get_data(S21_CH, get_x=True)

    magnet = np.unique(magnet)
    signal = abs(data[:,:])
    return freq, signal, magnet

def plot_data(signal, 
              zlim=[0,0],
              xlim=[0, 1],
              ylim = [0,1],
              figsize = (6, 8),
              xlabel = '$\Phi/\Phi_0$',
              ylabel = '$f_\mathrm{drive}$ (GHz)',
              zlabel='$|S_{21}|$ (arb.)',     
              cbar = True
             ):

    fig, ax = plt.subplots(figsize = figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    img = ax.imshow(signal, cmap = 'viridis', 
           extent =[xlim[0],xlim[-1], ylim[0], ylim[-1]], aspect='auto',
            interpolation='nearest')
    if cbar:
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(zlabel)
    if zlim != [0,0]:
        img.set_clim(vmin=zlim[0], vmax=zlim[1])
    
    return fig, ax

def save_fig(fig, name):
    fig.tight_layout()
    folder_path = 'figs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/{name}.eps', format='eps', 
               transparent=True)
    
def plot_spectrum(flux_extent, El, Ej, Ec, ylim = [4, 6], zero2 = True,
                 dashes = (2, 6)):
        flux_list = np.linspace(flux_extent[0],flux_extent[1], 151)
        fl= scq.Fluxonium(
            EJ=Ej,
            EC=Ec,
            EL=El,
            cutoff = 110,
            flux = 0)
        eval_num = 20
        spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=eval_num, subtract_ground=True);
        E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T
        f01 = E_phi[1]-E_phi[0]
        f02 = E_phi[2]-E_phi[0]
        plt.plot(flux_list, f01, color=line_colors[2], label='$f_{01}$', linestyle='--', dashes = dashes)
        if zero2:
            plt.plot(flux_list, f02, color=line_colors[1], label='$f_{02}$', linestyle='--', dashes = dashes)
        plt.xlabel('$\Phi/\Phi_0$')
#         plt.title('$E_L$ = '+str(round(El, 2))+' GHz'+', $E_J$ = '+str(round(Ej, 2))+' GHz'+ ', $E_C$ = '+str(round(Ec, 2))+' GHz')
        plt.legend()
    
    
def get_data(file_num,
             month,
             day,
             freq_slice=3,
             sample='JS681-4fluxonium-005-002-BS', 
             year = '2023', 
             channels = ['SC1 - Frequency', 'QCage Magnet - Source voltage', 'VNA - S21'],
             mag=True
            ):

    savepath = '/Users/billystrickland/Documents/code/resonators/data/'+sample+'/'
    os.makedirs(savepath, exist_ok=True)
    root = f'/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/{year}/{month}/Data_{month}{day}'
    FILE = f'{root}/{sample}-{file_num}.hdf5'
    DF_CH, P_CH, S21_CH = channels
    with LabberData(FILE) as f:
        magnet = f.get_data(P_CH)
        freq, signal = f.get_data(S21_CH, get_x=True)
        drive = f.get_data(DF_CH)
    
    magnet = np.unique(magnet)
    drive = np.unique(drive)

    if mag:
        signal = abs(signal)

    return freq, signal, magnet, drive # signal is complete complex S21 data, magnet is unique, drive is unique

def dynamic_slice(signal, dispersive=2, switch_indices=False):
    
    # For dealing with two tone data
    # Takes in complex signal of shape (magnet, drive, probe freq) and spits out an array of shape (magnet, drive) at the resonance (minus dispersive)
    fr = []
    if switch_indices:
        for i,x in enumerate(signal[:,0]):
            fr.append(np.argmin(x)-dispersive)
        fr = np.array(fr)
        result_array = signal[np.arange(len(signal[:,0])), :, fr.astype(int)]
    else:
        for i,x in enumerate(signal[:,0]):
            fr.append(np.argmin(x)-dispersive)
        fr = np.array(fr)
        result_array = signal[np.arange(len(signal[:,0])), :, fr.astype(int)]

    result_array = rf.average_data(result_array)
    
    return result_array.T[::-1]

# def plot_data(signal, clim=[0,0], flux_extent=[-1, 1], drive = [3e9, 7e9], figsize = (6, 8)):
#     y_label='$f_\mathrm{drive}$ (GHz)'    
#     fig, ax = rf.make_plot_pre('$\Phi/\Phi_0$', y_label, figsize = figsize)
#     img = ax.imshow(signal, cmap = 'viridis', 
#            extent =[flux_extent[0],flux_extent[-1], drive[0]*1e-9, drive[-1]*1e-9], aspect='auto',
#             interpolation='nearest')
#     cbar = fig.colorbar(img, ax=ax)
#     if clim != [0,0]:
#         img.set_clim(vmin=clim[0], vmax=clim[1])
#     cbar.ax.set_ylabel('$|S_{21}|$ (arb.)')
#     return fig, ax

# def plot_spectrum(flux_extent, El, Ej, Ec, ylim = [4, 6], zero2 = True, zero3 = False):
#     flux_list = np.linspace(flux_extent[0],flux_extent[-1], 151)
#     fl= scq.Fluxonium(
#         EJ=Ej,
#         EC=Ec,
#         EL=El,
#         cutoff = 110,
#         flux = 0)
#     eval_num = 20
#     spectrum = fl.get_spectrum_vs_paramvals('flux', flux_list, evals_count=eval_num, subtract_ground=True);
#     E_phi = inspect.getmembers(spectrum)[5][1]["energy_table"].T

#     f01 = E_phi[1]-E_phi[0]
#     f02 = E_phi[2]-E_phi[0]
#     f03 = E_phi[3]-E_phi[0]
#     plt.plot(flux_list, f01, color=line_colors[1], label='$f_{01}$', linestyle='--', dashes=(5, 5))
#     if zero2:
#         plt.plot(flux_list, f02, color=line_colors[8], label='$f_{02}$', linestyle='--', dashes=(5, 5))
#     if zero3:    
#         plt.plot(flux_list, f03, color=line_colors[3], label='$f_{03}$', linestyle='--', dashes=(5, 5))
#     plt.xlabel('$\Phi/\Phi_0$')
#     plt.legend()
    

from scipy.signal import find_peaks

def find_troughs(array_of_arrays, prominence=1):
    troughs = []
    for arr in array_of_arrays:
        peaks, _ = find_peaks(-np.array(arr), prominence=prominence, )
        if len(peaks) > 0:
            # Select the trough with the minimum value among the peaks
            min_trough_index = min(peaks, key=lambda x: arr[x])
            troughs.append(min_trough_index)
    return troughs
    
    
def I_to_Ej(x):
    return x*496.7 #gigahertz

def Ej_to_I(x):
    return x/496.7 #gigahertz

def save_fig(fig, name, format='eps'):
    fig.tight_layout()
    folder_path = 'figs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/{name}.{format}', format=format)
    
def normalize(signal):
    return signal*(1/max(signal[0]))

def mag_to_flux(magnet):
    return -0.266467 + 0.00299401*magnet*1e3


def extract_data(sample, file_num, month, day, year = '2024', channels = ['Qcage - Magnet - Source voltage', 'VNA - S21', 'SC1 - Frequency']):
    root = f'/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/{year}/'+month+'/Data_'+month+day
    FILE = root+'/'+sample+'-'+file_num+'.hdf5'
    with LabberData(FILE) as f:
        data=[]
        for i in channels:
            data.append(f.get_data(i))
    return data

def plot_t1_decay(time, signal, a, perr, phimax = 1, voltage=False, us=False):
    if voltage:
        fig, ax = rf. make_plot_pre(r'$\tau$ (ns)', r'$V_H$ (V)', figsize = (5, 4))
    else:
        fig, ax = rf. make_plot_pre(r'$\tau$ (ns)', r'phase (deg)', figsize = (5, 4))
    
    plt.plot(time, np.array((abs(exp_decay(time, a[0], a[1], a[2])-a[2])))/phimax)
    plt.plot(time, np.array(abs(signal-a[2]))/phimax, marker='.', linestyle='None')
    if us == False:
        plt.annotate('$T_1$ = ' + str(round(a[0], 2))+' $\pm$ '+str(round(perr[0], 2))+' ns', 
                     xycoords = 'figure fraction', xy=(.2, .8))
    else:
        plt.annotate('$T_1$ = ' + str(round(a[0]*1e-3, 2))+' $\pm$ '+str(round(perr[0]*1e-3, 2))+' µs', 
                     xycoords = 'figure fraction', xy=(.2, .8))

def fit_decay(time, signal, plot_fit=True, voltage=False, p0=[100, 1, 1]):
    popt, pcov = curve_fit(exp_decay, time, signal, p0=p0,maxfev=500000000)
    a = popt
    perr = np.sqrt(np.diag(pcov))
    if plot_fit:
        plot_t1_decay(time, signal, a, perr, voltage=voltage)
    return a, perr


def get_contrast(sample, file_num, month='04', day='09', window = [50,100], year='2024'):

    I, Q = extract_data(sample, file_num, month, day, year='2024',
                                    channels = [
                                        'Digitizer ShabLab - Ch3 - Signal',
                                        'Digitizer ShabLab - Ch4 - Signal',
                                               ])
    time = np.linspace(0, len(I[0])*2, len(I[0]))
    
    fig, ax = plt.subplots()
    plt.plot(time, I[0]*1e3, label = 'off')
    plt.plot(time, I[1]*1e3, label = 'on')
    plt.ylabel('$V_H$ (mV)')
    plt.xlabel('time (ns)')
    
    ax.fill_between(time, min(I[0])*1e3, max(I[1])*1e3, where=(time >= window[0]*2) & (time <= window[1]*2),
                    facecolor='green', alpha=.5,
                   label = 'integration window')
    
    averages_off = np.mean(I[0, window[0]:window[-1]])
    averages_on = np.mean(I[1, window[0]:window[-1]])
    
    plt.legend()
    
    return averages_off, averages_on

def get_T1(sample, file_num, month, day, on=0, off=0, window = [50,100], year='2024'):

    delay, I, Q = extract_data(sample, file_num, month, day,
                                    channels = [
                                        'MQPG - Readout delay',
                                        'Digitizer ShabLab - Ch3 - Signal',
                                        'Digitizer ShabLab - Ch4 - Signal',
                                               ])

    averages = np.mean(I[:, window[0]:window[-1]], axis = 1)

    a, pcov = curve_fit(qf.exp_decay, delay, averages,
                            maxfev=500000000,
                            bounds = [[10e-9, 0, averages[-1]-1], [100000e-9, 100, averages[-1]+1]]
                          )

    perr = np.sqrt(np.diag(pcov))
    print('T1, A, b = ', a)

    fig, ax = plt.subplots()
    if on!=0:
        plt.axhline(y = off*1e3, linestyle = '--', color = 'gray')
        plt.annotate('Off',xy=(delay[int(len(delay)-len(delay)/10)], off*1e3), color = 'gray')

        plt.axhline(y = on*1e3, linestyle = '--', color = 'gray')
        plt.annotate('On', xy = (delay[int(len(delay)-len(delay)/10)], on*1e3), color = 'gray')

    plt.plot(delay, qf.exp_decay(delay, *a)*1e3, label = 'Fit, $T_1$ = ' + str(round(a[0]*1e9))+' $\pm$ '+str(round(perr[0]*1e9))+' ns')
    plt.plot(delay, averages*1e3, marker ='.', linestyle = 'None', label = 'Data')
    
    #     plt.axhline(y = 0, linestyle = '--', color = 'gray')
#     plt.annotate('Off',xy=(delay[int(len(delay)-len(delay)/10)], .03), color = 'gray')

#     plt.axhline(y = 1, linestyle = '--', color = 'gray')
#     plt.annotate('On', xy = (delay[int(len(delay)-len(delay)/10)], .93), color = 'gray')

#     plt.plot(delay, (qf.exp_decay(delay, *a)-off)/(on-off), label = 'Fit, $T_1$ = ' + str(round(a[0]*1e9))+' $\pm$ '+str(round(perr[0]*1e9))+' ns')
#     plt.plot(delay, (averages-off)/(on-off), marker ='.', linestyle = 'None', label = 'Data')
    print('on, off = ', on, off)
    plt.ylabel('$V_H$ (mV)')
    plt.xlabel(r'$\tau$ (s)')
    plt.legend(fontsize = 14)
    
def generate_H(phi_ax: np.ndarray, 
               VJ_Fr, 
               EJ1: float,
               EJ2: float,
               EC: float, 
               EL: float, 
               ng: float, 
               phi_ext: float) -> Tuple[np.ndarray, np.ndarray]:
    
    phi_N = len(phi_ax)
    dphi = phi_ax[1] - phi_ax[0]
    
    # Calculate first and second derivatives of phase variable
    dphi_dx = (np.diag(np.ones(phi_N - 1), 1) 
               - np.diag(np.ones(phi_N - 1), -1)) / (2 * dphi)
    
    d2phi_dx2 = (-2 * np.diag(np.ones(phi_N), 0) + 
                 np.diag(np.ones(phi_N - 1), 1) + 
                 np.diag(np.ones(phi_N - 1), -1)) / dphi**2
    
    # Calculate the potential energy term 'V'
    V = EL/2 * (phi_ax)**2 + VJ_Fr(phi_ax - phi_ext, EJ1, EJ2)
    
    # Construct the Hamiltonian 'H' for the system
    H = -4 * EC * (d2phi_dx2 - 2 * ng * dphi_dx + ng**2 * np.ones(phi_N)) + np.diag(V)
    
    return H, phi_ax

def VJ_Fr(phi, EJ1, EJ2):
    return - EJ1*np.cos(phi) - EJ2*np.cos(2*phi)

def VJ_Sm(phi, K1, T1, K2, T2):
    return -K1*np.sqrt(1-T1*np.sin(phi/2)**2) -K2*np.sqrt(1-T2*np.sin(phi/2)**2)

def get_nonsine_freqs(EJ1, EJ2, EL = 2.8,
                    EC = .8, flux=[-.5, .5], step=61):
    ng = 0
    
    phi_N = 1001  # Minimum ~ 1000
    phi_max = 10*np.pi  # Min 10, further increase if the wavefunction tails are too close to the boundary
    phi_ax = np.linspace(-phi_max, phi_max, phi_N)
    
    # Spectrum as function of external flux (reduced coordinate phi_ext = 2 pi Phi / Phi_0)
    M = step
    phi_ext_ax = np.linspace(flux[0]*2*np.pi, flux[-1]*2*np.pi, M)

    ws = np.zeros((M, phi_N))
    vs = np.zeros((M, phi_N, phi_N))

    for i, phi_ext in tqdm(enumerate(phi_ext_ax), total=len(phi_ext_ax)):

        EJ = np.cos(phi_ax - phi_ext)
        H, phi_ax = generate_H(phi_ax, VJ_Fr,
                               EJ1=EJ1,
                               EJ2=EJ2,
                               EC=EC, 
                               EL=EL, 
                               ng=ng, 
                               phi_ext=phi_ext)
        ws[i], vs[i] = la.eigh(H)
        
    omegat = []
    label = []
    for i in range(4):
        omegat.append(ws[:, i+1]-ws[:, 0])    
        label.append(r'$f0$'+str(i+1))
    omegat = np.array(omegat)
    return omegat, phi_ext_ax

def plot_spectrum_AM(flux_extent, EJ1, EJ2, EL, EC, step=61):
    
    omegat, phi_ext_ax = get_nonsine_freqs(flux=flux_extent, EJ1=EJ1, EJ2=EJ2, EL=EL, EC=EC, step = step)

    for i,x in enumerate(omegat):
        plt.plot(phi_ext_ax/(2*np.pi), x, color = 'black', linestyle = '--', dashes=[2, 4])

    plt.plot(phi_ext_ax/(2*np.pi), omegat[1]-omegat[0], color = 'black', linestyle = '--', dashes=[2, 4])
    plt.plot(phi_ext_ax/(2*np.pi), omegat[2]-omegat[0], color = 'black', linestyle = '--', dashes=[2, 4])
