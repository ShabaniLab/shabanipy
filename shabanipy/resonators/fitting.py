from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.helper import *
from shabanipy.jy_mpl_settings.colors import line_colors

import csv
import matplotlib.pyplot as plt
from shabanipy.shabanipy.labber import LabberData
from shabanipy.resonators.notch_geometry import fit_complex, notch_from_results
import numpy as np
import os

from scipy.optimize import curve_fit

def linear(x, m, b):
    return m*x + b

def lin_fit(x,y):
    return curve_fit(linear, x, y)

def substract_delay(freq, phase):
    xf = np.append(freq[:10],freq[-10:])
    yf = np.append(phase[:10],phase[-10:])
    p, _ = lin_fit(xf,yf)
    yf = linear(freq, *p)
    return phase - yf

def to_db(x):
    return 20*np.log10(x)

################################## CHANGE HERE #####################################################
sample = 'JS626-4TR-Noconst-1-BSBHE-001'
root = '/Users/billystrickland/Documents/code/resonators/data/'
file_num = [
    '050',
    '051',
    '052'
    ]
FILES = []
for num in file_num:
    FILES.append(root+sample+'/'+str(sample)+'-'+num+'.hdf5')
    
att = [
    00,
    20,
    40,
]

ID = 'vg-6_att'+str(att[0])

fridge_att = 56

res_index = 0
gui = True
err = True

################################## CHANGE HERE #####################################################

P_CH, S21_CH = ['VNA - Output power', 'VNA - S21']
        
power = None
freq  = None
data  = None

for i, FILE in enumerate(FILES[:3]):
    with LabberData(FILE) as f:
        _p = f.get_data(P_CH) - att[i] - fridge_att
        _f, _d = f.get_data(S21_CH, get_x=True)
        _p = _p[::-1]
        _f = _f[::-1]
        _d = _d[::-1]
        
        if power is None:
            power = _p
            freq, data = _f, _d
        else:
            power = np.append(power, _p, axis=0)
            freq = np.append(freq, _f, axis=0)
            data = np.append(data, _d, axis=0)
                        
with LabberData(FILES[-1]) as f:
    
    _p = f.get_data(P_CH) - att[-1] - fridge_att
    _f, _d = f.get_data(S21_CH, get_x=True)
    _p = _p[:3:-1]
    _f = _f[:3:-1]
    _d = _d[:3:-1]
    
    power = np.append(power, _p, axis=0)
    freq = np.append(freq, _f, axis=0)
    data = np.append(data, _d, axis=0)

##
##freq = freq[:,res_index]
##data = data[:,res_index]
##power = power[:,res_index]

newpath = root+sample+'/results/fits/res'+str(res_index)
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(type(freq), type(data), type(power))
results, fdata = fit_complex(freq, data, powers=power,gui_fit=gui, #delay=.098833, 
                             return_fit_data=True, delay_range=(-.005,+0.005), save_gui_fits = True,
                             save_gui_fits_path= root+sample+'/results/fits/res'+str(res_index),
                             save_gui_fits_filetype='.eps')
data_columns = 'Qi_dia_corr, Qi_no_corr, absQc, Qc_dia_corr, Ql, fr, theta0, phi0, phi0_err, Ql_err, absQc_err, fr_err, chi_square, Qi_no_corr_err, Qi_dia_corr_err, prefactor_a, prefactor_alpha, baseline_slope, baseline_intercept, Power, Photon'

np.savetxt(root+sample+'/results/'+ID+'.csv', results, delimiter=',', header = data_columns)

