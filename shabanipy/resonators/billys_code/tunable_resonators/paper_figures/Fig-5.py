import sys
sys.path.append('/Users/billystrickland/Documents/code/resonators')

from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
from shabanipy.labber import LabberData
from shabanipy.resonators.notch_geometry import fit_complex
import numpy as np
import os
from scipy.optimize import curve_fit

sample = 'JS626-4TR-Noconst-1-BSBHE-001'
root = '/Users/billystrickland/Documents/code/resonators/data/'
file_num = '066'
FILES = [root+sample+'/'+str(sample)+'-'+file_num+'.hdf5']
res_index = 0
gui = True
err = True
p = 1
VG_CH, P_CH, S21_CH = ['Gate - Source voltage', 'VNA - Output power', 'VNA - S21']

power, freq, vg, data = None, None, None, None

for i, FILE in enumerate(FILES[:3]):
    with LabberData(FILE) as f:
        _p = f.get_data(P_CH) - 40 - 56
        _f, _d = f.get_data(S21_CH, get_x=True)
        _v = f.get_data(VG_CH)
        _p = _p[::-1]
        _f = _f[::-1]
        _d = _d[::-1]
        _v = _v[::-1]

        if power is None:
            power, freq, data, vg = _p, _f, _d, _v
        else:
            power = np.append(power, _p, axis=0)
            freq = np.append(freq, _f, axis=0)
            data = np.append(data, _d, axis=0)
            vg = np.append(vg, _v, axis=0)

newpath = root+sample+'/results'
if not os.path.exists(newpath):
    os.makedirs(newpath)

with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1,1, figsize=(9 ,6))
    img = ax.imshow(abs(data[p]*1e3), cm.get_cmap('viridis'), aspect = 'auto',
                    extent=[freq[p][0][0]*1e-9, freq[p][0][-1]*1e-9, vg[p][-1], vg[p][0]],)
    ax.set_xlabel('$f$ (GHz)')
    ax.set_ylabel('$V_G$ (V)')
    cbar = plt.colorbar(img)
    cbar.set_label('$|S_{21}|$ (a.u.)')
    fig.savefig(newpath+'/anti'+str(power[p][0])+'.eps', transparent=True, bbox_inches='tight')
    fig.savefig(newpath+'/anti'+str(power[p][0])+'.png', bbox_inches='tight')

mini = np.where(abs(data[p][0]) == abs(data[p][0]).min())

fr = []
for v in range(len(freq[p])):
    mini = np.where(abs(data[p][v]) == abs(data[p][v]).min())
    fr.append(freq[p][v][mini[0][0]])

cutoff = 63
fplus = fr[cutoff:]
vgplus = vg[p][cutoff:]
fminus = fr[:cutoff-1]
vgminus = vg[p][:cutoff-1]
results = np.array([fminus])
comboY = np.concatenate([fplus, fminus])
comboX = np.concatenate([vgplus, vgminus])
x1, x2 = vgplus, vgminus

def fplus_fit(vg, f1, m, b, g):
    return 0.5*(f1+m*vg+b) + ((g)**2+0.25*(m*vg+b-f1)**2)**0.5
def fminus_fit(vg, f1, m, b, g):
    return 0.5*(f1+m*vg+b) - ((g)**2+0.25*(-(m*vg+b)+f1)**2)**0.5
def combinedFunction(comboData, f1, m, b, g):
    extract1 = comboData[:len(vgplus)] # first data
    extract2 = comboData[len(vgplus):] # second data
    result1 = fplus_fit(extract1, f1, m, b, g)
    result2 = fminus_fit(extract2, f1, m, b, g)
    return np.append(result1, result2)

popt, pcov = curve_fit(combinedFunction, comboX, comboY,
                        p0 = np.array([ 5.42703146e+09,  6.28304336e+08,  1.36426134e+10, 4.12031653e+07]),maxfev=5000000)
f, a, b, c = popt
perr = np.sqrt(np.diag(pcov))
y_fit_1 = np.array(fplus_fit(vg[0], f, a, b, c))
y_fit_2 = np.array(fminus_fit(vg[0], f, a, b, c))

with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1,1, figsize=(7 ,6))
    ax.plot(comboY*1e-9,comboX, 'D', label = 'Data') # plot the raw data
    ax.plot(y_fit_1*1e-9, vg[0],color = line_colors[1], label = 'Fit') # plot the equation using the fitted parameters
    ax.plot(y_fit_2*1e-9, vg[0],color=line_colors[1]) # plot the equation using the fitted parameters
    ax.plot(1e-9*popt[1]*vg[0]+1e-9*popt[2], vg[0], linestyle='dashed', color='gray', label = '$f_1$, $f_2$')
    ax.axvline(x=popt[0]*1e-9, linestyle='dashed', color='gray')
    ax.set_xlim(5.37, 5.47)
    ax.set_ylabel('$V_G$ (V)')
    ax.set_xlabel('$f$ (GHz)')
    ax.annotate('$g/2\pi$ = '+str(round(popt[3]*10**(-6), 3))+' MHz', xycoords = 'figure fraction', xy = (0.3, 0.2), fontsize = 15)
    plt.legend(bbox_to_anchor = (0.4, 1), frameon = False)
    fig.tight_layout()
    fig.savefig(newpath+'/anti_fits'+str(power[p][0])+'.eps', transparent=True, bbox_inches='tight')
    fig.savefig(newpath+'/anti_fits'+str(power[p][0])+'.png', bbox_inches='tight')
    np.savetxt(newpath+'/066_res_freqs.csv', results, delimiter=',')
    plt.show()
