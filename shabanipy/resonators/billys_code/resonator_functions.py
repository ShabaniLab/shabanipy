from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import math
import csv
import os
import matplotlib.pyplot as plt
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.helper import *
from shabanipy.jy_mpl_settings.colors import line_colors
from shabanipy.labber import LabberData


def make_savepath(sample, ID):
    root = '/Users/billystrickland/Documents/code/resonators/data/'
    savepath = root+sample+'/results'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    newpath = savepath+'/fits/'+ID
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return savepath, newpath

def make_plot_pre(xlabel, ylabel, xscale = 'linear', figsize = (8,5)):
    with plt.rc_context(jy_mpl_rc):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        fig.tight_layout()
    return fig, ax

def to_db(value_raw,reference=1.):
    # take in magnitude of complex transmission data abs(S_21) and plot it as dB
    return 10*np.log10(value_raw/reference)

def db_to_watts(x):
    return 10**(x/(10))*0.001

def average_data(signal):
    row_averages = np.mean(signal, axis=1)
    return signal - row_averages[:, np.newaxis]

def make_plot_post(fig, savename, legend = False):
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    fig.tight_layout()
    fig.savefig(savename+'.eps', transparent=True)
    fig.savefig(savename+'.png')
    plt.show()

def proc_csv(FILES):
    results = []
    for f in FILES:
        with open(f, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                results.append(list(map(float,row)))
    return results

def to_db(value_raw,reference=1.):
    return 10*np.log10(value_raw/reference)

def from_db(value_raw):
    return np.exp(value_raw/10)

def get_results(results, err_thresh):
    def filter_results(index):
        return [r[index] for i, r in enumerate(results) if err[i] < err_thresh]
    
    err = [r[14] for i, r in enumerate(results)]
    filtered_results = filter_results
    
    return np.array(
        [
            filtered_results(-1),
            filtered_results(-2),
            filtered_results(0),
            filtered_results(14),
            filtered_results(3),
            filtered_results(10),
            filtered_results(4),
            filtered_results(9),
            filtered_results(5),
            filtered_results(11),
        ]
    )
def f_to_l(freq, capacitance):
    return np.array([((2*math.pi*x)**2*capacitance)**(-1) for x in freq])

def f_to_l_err(freq, freq_err, ls):
    return [r*freq_err[i]/freq[i] for i, r in enumerate(ls)]

def f_to_lj(freq, capacitance, lest):
    return np.array([((2*math.pi*x)**2*capacitance)**(-1)-lest for x in freq])

def alpha(freq_geo, freq_meas):
    return 1-(freq_meas/freq_geo)**2

def freq(L, C):
    return 1/(2*np.pi*np.sqrt(L*C))

def mfq():
    return 2.068*10**(-15)

def lj_to_ic(lj):
    return mfq()/(2*math.pi*lj)

def ic_to_lj(ic):
    mfq = 2.068*10**(-15)
    return mfq/(2*math.pi*ic)

def two_fluid_model(T, alpha_K, Tc):
    return -alpha_K/(2-2*np.power(np.multiply(T, 1/Tc), 4))+alpha_K/2

def fplus_fit(vg, f1, m, b, g):
    return 0.5*(f1+m*vg+b) + ((g)**2+0.25*(m*vg+b-5.425*10**9)**2)**0.5

def fminus_fit(vg, f1, m, b, g):
    return 0.5*(f1+m*vg+b) - ((g)**2+0.25*(-(m*vg+b)+5.425*10**9)**2)**0.5


def plot_f_qint(FILES, x_label, savepath, traces, mod=1, scale='linear', err_thresh=1000, figsize = (6,4), yscale = 'linear', qext = False):
    y_label='$Q_\mathrm{int}$'
    fig, ax = make_plot_pre(x_label, y_label, xscale = scale, figsize = figsize)
    ax.set_yscale(yscale)
    for i in range(len(FILES)):
        results = proc_csv(FILES[i])            
        photon, power, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err = get_results(results, err_thresh) 
        if scale == 'log':
            ax.errorbar(np.array(photon)*mod,qi_diacorr,yerr=qi_diacorr_err,linestyle = 'None',color=line_colors[i+1], marker = '.', label = traces[i])
        elif scale == 'linear':
            ax.errorbar(np.array(power)*mod,qi_diacorr,yerr=qi_diacorr_err,linestyle = 'None',color=line_colors[i+1], marker = '.', label = traces[i])
    savename = savepath+'_Qint'
    make_plot_post(fig, savename, legend=True)

    y_label='$f_r$ (GHz)'
    fig, ax = make_plot_pre(x_label, y_label, xscale = scale, figsize = figsize)

    for i in range(len(FILES)):
        results = proc_csv(FILES[i])            
        photon, power, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err = get_results(results, err_thresh) 
        if scale =='log':
            ax.errorbar(np.array(photon)*mod, np.array(freq)*1e-9, yerr=np.array(freq_err)*1e-9
                    , color = line_colors[i+1],linestyle = 'None', marker = '.', label = traces[i])
        if scale =='linear':
            ax.errorbar(np.array(power)*mod, np.array(freq)*1e-9, yerr=np.array(freq_err)*1e-9
                    , color = line_colors[i+1],linestyle = 'None', marker = '.', label = traces[i])

    savename = savepath+'_fr'
    make_plot_post(fig, savename, legend=True)
    if qext == True:
        y_label='$Q_\mathrm{ext}$'
        fig, ax = make_plot_pre(x_label, y_label, xscale = scale, figsize = figsize)

        for i in range(len(FILES)):
            results = proc_csv(FILES[i])            
            photon, power, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err = get_results(results, err_thresh) 
            if scale =='log':
                ax.errorbar(np.array(photon)*mod, np.array(qc), yerr=np.array(qc_err),
                        color = line_colors[i+1],linestyle = 'None', marker = '.', label = traces[i])
            if scale =='linear':
                ax.errorbar(np.array(power)*mod, np.array(qc), yerr=np.array(qc_err),
                        color = line_colors[i+1],linestyle = 'None', marker = '.', label = traces[i])

        savename = savepath+'_Qext'
        make_plot_post(fig, savename, legend=True)

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

    return signal, magnet, drive # signal is complete complex S21 data, magnet is unique, drive is unique

def save_fig(fig, name):
    fig.tight_layout()
    folder_path = 'figs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/{name}.eps', format='eps')
