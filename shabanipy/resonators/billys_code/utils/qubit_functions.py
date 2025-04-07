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

def get_lamb_shift(freq, data_lowpower, data_highpower, plot_traces = False):
    if plot_traces:
        fig, ax = rf.make_plot_pre('$f$ (GHz)', '$S_{21}$ (dB)')
        plt.plot(freq, abs(data_lowpower), linestyle = 'None', marker = 'o', label='high power')
        plt.plot(freq, abs(data_highpower)+.6, linestyle = 'None', marker = 'o', label='low power') 
    mini_0 = get_freq(freq, data_lowpower)
    mini_1 = get_freq(freq, data_highpower)
    return (mini_0-mini_1)*1e-6

def phase_func(I, Q):
    phase1 = []
    for h in range(len(I)):
        z_i = complex(I[h], Q[h])*360/(2*np.pi)
        phase1.append(cmath.phase(z_i))
    return np.array(phase1)*360/(2*np.pi)

def sqrtfit(x, y, z):
    return z*np.sqrt((x-y))

def decaying_sin(t, A, T, freq, m, b, p):
    return -A*np.exp(-t/T)*np.cos(freq*2*np.pi*t+p)+m*t+b

def db_to_watts(x):
    return 10**(x/(10))*0.001

def exp_decay(t, T1, A, b):
    return A*np.exp(-t/T1)+b

def sin_simple(t, A, freq, b):
    return A*np.cos(freq*2*np.pi*t)+b

def decay(t, A, T, m, b):
    return -A*np.exp(-t/T)+m*t+b

def decay_ref(t,A, T, m, b):
    return A*np.exp(-t/T)+m*t+b

def sin_simple(t, A, freq, b):
    return A*np.cos(freq*2*np.pi*t)+b

def get_photons_in_resonator(kc, ki, fr,power,unit='dBm'):
    if unit=='dBm':
        power = dBm_to_watts(power)
    return 4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2) * power

def average_data(signal):
    row_averages = np.mean(signal, axis=1)
    result_array = signal - row_averages[:, np.newaxis]
    signal = result_array
    return signal

def plot_t1_decay(time, signal, a, perr, phimax = 1, voltage=False, us=False, figsize = (5,4)):
    if voltage:
        fig, ax = rf. make_plot_pre(r'$\tau$ (ns)', r'$V_H$ (mV)', figsize = figsize)
    else:
        fig, ax = rf. make_plot_pre(r'$\tau$ (ns)', r'phase (deg)', figsize = figsize)
    
    plt.plot(time, np.array((abs(exp_decay(time, a[0], a[1], a[2])-a[2])))/phimax)
    plt.plot(time, np.array(abs(signal-a[2]))/phimax, marker='.', linestyle='None')
    if us == False:
        plt.annotate('$T_1$ = ' + str(round(a[0], 2))+' $\pm$ '+str(round(perr[0], 2))+' ns', 
                     xycoords = 'figure fraction', xy=(.4, .7))
    else:
        plt.annotate('$T_1$ = ' + str(round(a[0]*1e-3, 2))+' $\pm$ '+str(round(perr[0]*1e-3, 2))+' µs', 
                     xycoords = 'figure fraction', xy=(.2, .8))

def fit_decay(time, signal, plot_fit=True, voltage=False, p0=[100, 1, 1], figsize = (6, 4)):
    popt, pcov = curve_fit(exp_decay, time, signal, p0=p0, maxfev=500000000)
    a = popt
    perr = np.sqrt(np.diag(pcov))
    if plot_fit:
        plot_t1_decay(time, signal, a, perr, voltage=voltage, figsize = figsize)
    return a, perr

def get_contrast_schuster(g, kappa, omegar, omegaq):
    phi = np.arctan(2.*g**2/(kappa*(omegar-omegaq)))*360/(2.*np.pi)
    return 2*phi

def get_contrast(omegaq):
    kappa = 3.45*1e6          # in GHz
    omegar = 7.094*1e9             # in GHz
    g = 64.75*1e6               # in GHz
    phi = get_contrast_schuster(g, kappa, omegar, omegaq)*27/360
    return 2*phi

def readout_delay(signal, delay):
    timed_signal = []
    for j, y in enumerate(signal):
        timer = y
        timed_signal.append(timer[j+delay])
    return timed_signal

def find_max_contrast(I_vs_time, Q_vs_time):
    contrast = []
    for i in range(len(I_vs_time[0])):
        contrast_i = qf.phase_func(I_vs_time[:,i] , Q_vs_time[:,i])[0]-phase_func(I_vs_time[:,i] , Q_vs_time[:,i])[-1]
        contrast.append(contrast_i)
    return np.argmax(contrast)

def fit_rabi_linecut(time, signal,
                     bounds=[[-15, 20e-9, 0, 1e-9, -360, 0],[-5, 100e-9, .35e9, .5e9,360, 2 * np.pi]],
                     linestyle = 'None',
                    ylabel = '$V_H$ (mV)',
                    xlabel = r'$\tau_\mathrm{Rabi}$ (ns)',
                    figsize = (6,4)):
    # bounds = amplitude A, time constant T, Rabi frequency freq, slope m, y offset b, phase p
    time_hr = np.linspace(time[0], time[-1], 1000)
    with plt.rc_context(jy_mpl_rc):
        fig, ax = rf.make_plot_pre(xlabel,ylabel, figsize = figsize)
        a, pcov = curve_fit(decaying_sin, time, signal,
                        bounds=(bounds[0], bounds[1]),
                        maxfev=500000000)
        print('A, T, freq, m, b, p = ', a)
        
        perr = np.sqrt(np.diag(pcov))
        plt.plot(time*1e9, signal, marker = '.', linestyle = linestyle)
        plt.plot(time_hr*1e9, decaying_sin(time_hr, *a))
        ax.plot(time_hr*1e9, decay(time_hr, a[0], a[1], a[3], a[4]), color = 'black', linestyle = '--')
        ax.plot(time_hr*1e9, decay_ref(time_hr, a[0], a[1], a[3], a[4]), color = 'black', linestyle = '--')
        meter_min = sin_simple(0, a[0], a[2], a[4])
        meter_max = sin_simple(1/a[2]/2, a[0], a[2], a[4])
        swing = meter_max-meter_min
        print('measured meter swing = ', round(swing, 2), 'deg')
#         print('expected meter swing= ', round(get_contrast(qubit_freq), 1), 'deg')
#         fig, ax = rf.make_plot_pre(r'$\tau_\mathrm{Rabi}$ (ns)',r'$P_{|1\rangle}$' )
#         plt.plot(time*1e9, (signal-meter_min)/swing, marker = 'o')
        plt.annotate('$T_2$ = ' + str(round(a[1]*1e9, 2))+' $\pm$ '+str(round(perr[1]*1e9, 2))+' ns', 
                     xycoords = 'figure fraction', xy=(.5, .8))
        return a, perr, fig, ax

def get_data_rabiplus1(day,  month='01', year = '2023', file_num='001', freq=True, sample = 'JS681-4fluxonium-005-003-BS'):
    root = '/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/'
    FILE = f"{root}{year}/{month}/Data_{month}{day}/{sample}-{file_num}.hdf5"
    if freq:
        channels = ['Digitizer ShabLab - Ch3 - Signal', 'Digitizer ShabLab - Ch4 - Signal', 'SC1 - Drive - Frequency']
    else:
        channels = ['Digitizer ShabLab - Ch3 - Signal', 'Digitizer ShabLab - Ch4 - Signal', 'SC1 - Drive - Amplitude']
    I_CH, Q_CH, D_CH = channels
    with LabberData(FILE) as f:
        I, Q, drive = f.get_data(I_CH), f.get_data(Q_CH), f.get_data(D_CH)
    return I, Q, drive

def plot_rabi_2d(signal, 
                 xlabel='', 
                 ylabel='', 
                 zlabel='',
                 extent=[0,1,0,1],
                 zlim = [0,0]
                ):
    with plt.rc_context(jy_mpl_rc):
        fig, ax = rf.make_plot_pre(xlabel,ylabel )
        pos = ax.imshow(signal, cmap = 'viridis', interpolation='none', aspect='auto',
                       extent = extent)
        cbar = fig.colorbar(pos, ax=ax)
        cbar.ax.set_ylabel(zlabel)
    if zlim != [0,0]:
        pos.set_clim(vmin=zlim[0], vmax=zlim[1])
    return fig, ax

def decaying_sin_simple(t, A, T, freq, b):
    return A*np.exp(-t/T)*np.cos(freq*2*np.pi*t)+b

def sin_simple(t, A, freq, b):
    return A*np.cos(freq*2*np.pi*t)+b
        
def get_timed_phase2d(I, Q, delay):
    phase_2d = []
    for i, x in enumerate(Q):
        phase = phase_func(readout_delay(x,delay), readout_delay(I[i], delay))
        phase_2d.append(phase)
    return phase_2d

def readout_delay(signal, delay):
    timed_signal = []
    for j, y in enumerate(signal):
        timer = y
        timed_signal.append(timer[j+delay])
    return timed_signal

    
