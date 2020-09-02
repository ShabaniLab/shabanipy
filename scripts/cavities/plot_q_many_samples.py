import numpy as np
import csv
from os import path as pth
import matplotlib.pyplot as plt

BASEPATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper'
DATAPATH = pth.join(BASEPATH,'data')
CSV_PATH = pth.join(BASEPATH,'fits')
IMAGE_DIR = pth.join(BASEPATH,'images')


symbols = [
    ('.',18),
    ('P',12),
    ('^',12),
    ('*',12),
    ('*',12),
    ('o',12)
]

sample_info = {
    'JS314' : {
        'name'    : 'InAs-Al (lattice-matched buffer)',
        'indices' : [3],
        'files'   : [
            'JS314_CD1_att20_004',
            'JS314_CD1_att40_006',
            'JS314_CD1_att60_007'
        ],
        'att'     : [-20,-40,-60]
    },
    'JS200' : {
        'name'    : 'InAs-Al (lattice-mismatched buffer)',
        'indices' : [0],
        'files'   : ['JS200_JY001_008'],
        'att'     : [0]
    },
    'IBM' : {
        'name'    : 'Nb on Si',
        'indices' : [0],
        'files'   : ['IBM_resonators_007'],
        'att'     : [0]
    },
    'InP_F' : {
        'name'    : 'Al on InP',
        'indices' : [0],
        'files'   : [
            'InPF_JY001_003',
            'InPF_JY001_001',
            'InPF_JY001_002',
            'InPF_JY001_004'
        ],
        'att'     : [0,20,40,60]
    },
#    'InP_G' : {
#        'name': 'Sputtered Al on InP (G)',
#        'indices': [0],#,1,2,3,4,5],
#        'files': [
#            'InPG_JY001_004',
#            'InPG_JY001_005',
#            'InPG_JY001_006'
#        ],
#    'att': [0, 20, 40]
#    },
}

res_freq_array = []

samples = {
    sample : {
            }
    for sample in sample_info
}

fieldnames = [
    'total_power',
    'photon_num',
    'qi',
    'qi_err',
    'qc',
    'ql',
    'ql_err',
    'chi_square'
]

for sample in sample_info:
    for res_index in sample_info[sample]['indices']:
        if res_index not in samples[sample]:
            samples[sample][res_index] = {
                'total_power': [],
                'qi': [],
                'qi_err': [],
                'ql': [],
                'ql_err': [],
                'qc': [],
                'chi_square': [],
                'photon_num': [],
                'res_freq': np.nan,
            }
        for filename in sample_info[sample]['files']:
            res_result_path = pth.join(CSV_PATH,sample,
                filename + '-' + str(res_index) + '.csv')
            temps = []
            if pth.exists(res_result_path):
                print(res_result_path)
                with open(res_result_path, mode='a+') as data_base:
                    data_base.seek(0)
                    csv_reader = csv.DictReader(data_base)
                    for row in csv_reader:
                        for fieldname in fieldnames:
                            try:
                                samples[sample][res_index][fieldname].append(
                                    float(row[fieldname])
                                )
                            except ValueError as e:
                                print(e)
                        try:
                            samples[sample][res_index]['res_freq'] = float(
                                row['res_freq']
                            )
                        except ValueError as e:
                            print(e)

for sample in samples:
    for res_index in samples[sample]:
        for fieldname in samples[sample][res_index]:
            samples[sample][res_index][fieldname] = np.array(
                samples[sample][res_index][fieldname]
            )

symbols = {sample : symbol for sample,symbol in zip(samples,symbols)}

markersize   = 15
labelsize    = 16
tickfontsize = 12
legendsize   = 16

bottom = .12
top    = .93
right  = .98
left   = .15

plot_info = [
    ['total_power','qi',"Power Vs. Q$_i$", "linear", "log", "Power [dB]",
        "Internal Quality Factor", 'Qi'],
    ['photon_num','qi',"Photon Number Vs. Q$_i$", "log", "log", "Photon Number",
        "Internal Quality Factor", 'Qi_photon'],
    ['total_power','qc',"Power Vs. Q$_C$", "linear", "log", "Power [dB]",
        "Coupled Quality Factor", 'Qc'],
    ['photon_num','qc',"Photon Number Vs. Q$_C$", "log", "log", "Photon Number",
        "Coupled Quality Factor", 'Qc_photon'],
]

def moving_average(values,index,pts):
    if pts%2 == 0:
        step = pts//2
    else:
        step = (pts-1)//2
    L = max(0, index - step)
    R = max(len(values)-1, index + step)
    return np.average(values[L:R])

# Work in progress

def filter_points(xdata,ydata,chi2,box_average_pts=3,filter_multiple=10):
    return xdata,ydata
    filtered_x,filtered_y = [],[]
    for i in range(len(xdata)):
        avg_q = moving_average(ydata,i,box_average_pts)
        if avg_q*filter_multiple > ydata[i] > avg_q/filter_multiple:
            filtered_x.append(xdata[i])
            filtered_y.append(ydata[i])
    return filtered_x,filtered_y
        
for (xdata_key, ydata_key, title, xscale, yscale, xlabel, ylabel, 
        filename) in plot_info:
    f,ax = plt.subplots(1,1,figsize=(16,9))
    for sample in samples:
        for res_index in samples[sample]:
            xdata = samples[sample][res_index][xdata_key]
            ydata = samples[sample][res_index][ydata_key]
            res_freq = samples[sample][res_index]['res_freq']
            labelstring = sample_info[sample]['name'] + ', $f_r$ = ' + f'{res_freq/1e9:.1f} GHz' 
            ax.plot(xdata, ydata, '.', label=labelstring, ms=markersize)
    ax.set_title(title,fontsize=labelsize)
    ax.set_xlabel(xlabel,fontsize=labelsize)
    ax.set_ylabel(ylabel,fontsize=labelsize)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
        
    for _ax in [ax.xaxis,ax.yaxis]:
        for tick in _ax.get_major_ticks():
            tick.label.set_fontsize(tickfontsize)
        
    ax.legend(fontsize=legendsize)
    plt.tight_layout()
    f.savefig(pth.join(IMAGE_DIR,'combined',filename + '.png'))


photon_figure, axs = plt.subplots(1,2,figsize=(12,6))

for sample in samples:
    for res_index in samples[sample]:
        xdata = samples[sample][res_index]['photon_num']
        ydata = samples[sample][res_index]['qc']
        res_freq = samples[sample][res_index]['res_freq']
        #labelstring = sample + ' $f_r$ = ' + f'{res_freq/1e9:.3f} GHz'
        labelstring = sample_info[sample]['name'] + ', $f_r$ = ' + f'{res_freq/1e9:.1f} GHz' 
        symbol, size = symbols[sample]
        axs[0].plot(xdata, ydata, symbol, label=labelstring, ms=size)

axs[0].set_title('External Quality Factor',fontsize=labelsize)
axs[0].set_xlabel('Photon Number',fontsize=labelsize)
axs[0].set_ylabel('$Q_{ext}$',fontsize=labelsize)

axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[0].legend(ncol=1,prop={'size': 12})
axs[0].set_ylim([None,1e6])

for sample in samples:
    for res_index in samples[sample]:
        xdata = samples[sample][res_index]['photon_num']
        ydata = samples[sample][res_index]['qi']
        res_freq = samples[sample][res_index]['res_freq']
        #labelstring = sample + ' $f_r$ = ' + f'{res_freq/1e9:.3f} GHz'
        labelstring = sample_info[sample]['name'] + ', $f_r$ = ' + f'{res_freq/1e9:.1f} GHz' 
        symbol, size = symbols[sample]
        axs[1].plot(xdata, ydata, symbol, label=labelstring, ms=size)

axs[1].set_title('Internal Quality Factor',fontsize=labelsize)
axs[1].set_xlabel('Photon Number',fontsize=labelsize)
axs[1].set_ylabel('$Q_{int}$',fontsize=labelsize)

axs[1].set_xscale('log')
axs[1].set_yscale('log')

#axs[1].legend(ncol=2)
axs[1].set_ylim([None,1e7])

#handles, labels = axs[1].get_legend_handles_labels()
#photon_figure.legend(handles, labels, loc='upper center')

photon_figure.tight_layout()
photon_figure.savefig(pth.join(IMAGE_DIR,'combined_photon.png'))
photon_figure.savefig(pth.join(IMAGE_DIR,'fig3.eps'))

#plt.show()

