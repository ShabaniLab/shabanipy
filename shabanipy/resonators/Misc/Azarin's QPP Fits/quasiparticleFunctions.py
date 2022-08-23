# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:29:33 2020

change log:
    4/13/2021 - added functions to import data in smaller chunks. 
        changed loadAlazarData to return data in uint16 format rather than convert to mV.
        added uint16_to_mV to convert uint16 data to mV depending on whether 12 bit 
            sample is stored in least or most significant digits


@author: lfl
"""
import numpy as np
from scipy.constants import hbar, e, pi
from scipy.signal import windows, oaconvolve, savgol_filter
from scipy.optimize import curve_fit,leastsq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import os
from time import perf_counter
from scipy.ndimage.filters import gaussian_filter
from scipy.special import lambertw


def expectedShift(q0, f0, ls, phi, delta=170e-6):
    '''Returns the maximum expected shift (tau = 1) due to single quasiparticle
        trapping

    returns df in linear frequency
    -------------------------------
    q0:     participation ratio
    f0:     zero qp resonance at the given flux [in Hz]
    ls:     the Josephson current of nanosquid (Lj/2)
    phi:    reduced flux -- where you're monitoring
    delta:  superconducting gap in eV
    '''
    # TODO: Is this accurate?
    phi0 = hbar/2/e
    return (-q0 * f0 * ls * delta * e *
            (np.cos(pi * phi) + (np.sin(pi * phi / 2)**4)) / 8 /
            (phi0**2) / (1 - (np.sin(pi * phi / 2)**2)))


def f_n_phi(phi, n, L=1.82897e-9, C=0.739929e-12, Ls=20.8475e-12,
            Delta=2.72370016e-23):
    '''Returns the expected frequency with n trapped quasiparticles

    returns f(phi,n) in linear frequency
    -------------------------------
    phi:    reduced flux -- where you're monitoring
    n:      number of trapped quasiparticles
    L:      linear inductance
    C:      linear capacitance
    Ls:     squid inductance at 0 flux
    Delta:  superconducting gap in J
    '''
    # TODO: is this accurate?
    de = np.pi*phi
    Lsphi = Ls/(1-np.sin(de/2)*np.arctanh(np.sin(de/2)))
    q = Lsphi/(L+Lsphi)
    rphi0 = (2.06783383*1e-15)/(2*np.pi)
    f0 = 1/(2*np.pi*np.sqrt((L+Lsphi)*C))
    alpha = Delta/(2*(rphi0**2))
    # L1 = alpha*(np.cos(de)+np.sin(de/2)**4)/((1-np.sin(de/2)**2)**1.5)
    L1 = alpha*np.cos(de/2)
    return f0 - (q*f0*Lsphi*n*L1/2)
#     return L1


def get_Ls(q0, L):
    '''
    From participation ratio and linear inductance, get the nano squid inductance

    Parameters
    ----------
    q0 : float
        participation ratio obtained by fitting flux curve.
    L : float
        linear structure inductance, likely simulated.

    Returns
    -------
    Ls : float
        The nanosquid inductance.

    '''
    # TODO: apply iterative method to converge on best estimate
    return L*q0/(1-q0)


def uint16_to_mV(data, bitshifted=False):
    '''
    convert unsigned 16 bit integer data from Alazar to mV.

    Parameters
    ----------
    data : array, dtype=np.uint16
        data as read from alazar.
    bitshifted : boolean, optional
        should be True if the 12 bit alazar data is stored in the least 
        significant bits of each 16 bit sample. The default is False.

    Returns
    -------
    mV : array with dtype=float64
        The data in mV units.

    '''
    if bitshifted:
        return (data - 2047.5) * (400/2047.5)
    else:
        return (data - 32767.5) * (400/32767.5)


def getNumberSegmentsForBigData(fpath, segmentSizeGB=10):
    '''Get the chunking characteristics for importing large datasets.

    returns nSegments, elementsPerSegment
    nSegments = number of segments of the provided size required to load a
                    large dataset for chunked processing.
    elementsPerSegment = number of 16 bit integers to load in a single chunk.
    ----------------------------------------
    fpath:          file path to .bin file
    segmentsSizeGB: how many GB of raw data you want to load in a single chunk.
                        Must be smaller than system memory.
    '''
    if fpath[-4:] != '.bin':
        print('fpath should be full path to .bin filetype of raw digitizer data.')
        return None
    bytesPerGB = 1024**3
    bytesPerElement = 2
    elementsPerGB = bytesPerGB/bytesPerElement
    elementsPerSegment = segmentSizeGB * elementsPerGB
    elementsPerFile = os.stat(fpath).st_size // bytesPerElement
    nSegments = np.ceil(elementsPerFile/elementsPerSegment)
    return int(nSegments), int(elementsPerSegment)


def loadChunk(fpath, elementsPerSegment, segment, nChannels=2):
    '''Load a single chunk of data from a large .bin save file.

    returns DATA array with dtype = np.uint16
    ------------------------------------------
    fpath:      file path to .bin file
    elementsPerSegment: How many uint16 are in a single chunk
    segment:    which chunk you want, starts with first chunk as segment=0
    nChannels:  number of channels which fpath file includes
    '''
    if fpath[-4:] != '.bin':
        print('fpath should be full path to .bin filetype of raw digitizer data.')
        return None
    bytesPerElement = 2
    skipBytes = bytesPerElement * elementsPerSegment * segment
    DATA = np.fromfile(fpath,dtype=np.uint16,count=elementsPerSegment,offset=skipBytes)
    if nChannels == 2:
        DATA = DATA.reshape((2,len(DATA)//2),order='F')
    return DATA

def loadAlazarData(fpath, nChannels=2):
    '''pulls ATS9371 digitizer data from .bin save file (fpath) and returns
        array in mV units.

    returns DATA array in mV.
    if nChannels = 2, returns (#Samples,2) array. Otherwise 1 dimension array
    ------------------------
    fpath:      file path to .bin file
    nChannels:  number of channels which fpath file includes
    '''
    if fpath[-4:] != '.bin':
        print('fpath should be full path to .bin filetype of raw digitizer data.')
        return None
#     DATA = (np.fromfile(fpath,dtype=np.uint16)- 2047.5) * 400/2047.5
    DATA = np.fromfile(fpath,dtype=np.uint16)
    if nChannels == 2:
        DATA = DATA.reshape((2,len(DATA)//2),order='F')
    return DATA

# def loadAlazarDataCppAvg(fpath,nChannels=2):
#     '''pulls ATS9371 digitizer data from .bin save file (fpath) and returns array in mV units.
    
#     returns DATA array in mV.
#     if nChannels = 2, returns (#Samples,2) array. Otherwise 1 dimension array
#     ------------------------
#     fpath:      file path to .bin file
#     nChannels:  number of channels which fpath file includes
#     '''
#     if fpath[-4:] != '.bin':
#         print('fpath should be full path to .bin filetype of raw digitizer data.')
#         return None
#     DATA = (np.fromfile(fpath,dtype=np.uint16)- 32767.5) * (800/65536)
#     if nChannels == 2:
#         DATA = DATA.reshape((2,len(DATA)//2))
#     return DATA

def HannConvolution(data,avgTime,sampleRate):
    '''Smooths data by convolving with a Hann window of duration avgTime.
    
    returns smoothed data with original shape.
    -------------------------------------
    data:       single or dual channel data. dual channel should have shape (nSamples,2)
    avgTime:    duration of Hann window in seconds
    sampleRate: sample rate of data in Hz.
    '''
    nAvg = int(max(avgTime*sampleRate,1))
    window = windows.hann(nAvg)
    norm = sum(window)
    if len(data.shape) == 2:
        mean = np.mean(data,axis=1)
        window = np.vstack((window,window))
        return (oaconvolve((data.T-mean).T,window,mode='same',axes=1).T/norm + mean).T
    else:
        mean = np.mean(data)
        return oaconvolve(data-mean,window,mode='same')/norm + mean

def FlattopConvolution(data,avgTime,sampleRate):
    '''Smooths data by convolving with a flattop window of duration avgTime.
    
    returns smoothed data with original shape.
    -------------------------------------
    data:       single or dual channel data. dual channel should have shape (nSamples,2)
    avgTime:    duration of Hann window in seconds
    sampleRate: sample rate of data in Hz. 
    '''
    nAvg = int(max(avgTime*sampleRate,1))
    window = windows.flattop(nAvg)
    norm = sum(window)
    if len(data.shape) == 2:
        mean = np.mean(data,axis=1)
        window = np.vstack((window,window))
        return (oaconvolve((data.T-mean).T,window,mode='same',axes=1).T/norm + mean).T
    else:
        mean = np.mean(data)
        return oaconvolve(data-mean,window,mode='same')/norm + mean
    
def GaussianConvolution(data,avgTime,sampleRate):
    '''Smooths data by convolving with a gaussian window of duration avgTime
    and standard deviation avgTime/7.
    
    returns smoothed data with original shape.
    -------------------------------------
    data:       single or dual channel data. dual channel should have shape (nSamples,2)
    avgTime:    duration of Hann window in seconds
    sampleRate: sample rate of data in Hz.
    '''
    nAvg = int(max(avgTime*sampleRate,1))
    window = windows.gaussian(nAvg,nAvg/7)
    norm = sum(window)
    if len(data.shape) == 2:
        mean = np.mean(data,axis=1)
        window = np.vstack((window,window))
        return (oaconvolve((data.T-mean).T,window,mode='same',axes=1).T/norm + mean).T
    else:
        mean = np.mean(data)
        return oaconvolve(data-mean,window,mode='same')/norm + mean

def BoxcarConvolution(data,avgTime,sampleRate):
    '''Smooths data by convolving with a boxcar window of duration avgTime.
    
    returns smoothed data with original shape.
    -------------------------------------
    data:       single or dual channel data. dual channel should have shape (nSamples,2)
    avgTime:    duration of boxcar window in seconds
    sampleRate: sample rate of data in Hz.
    '''
    nAvg = int(max(avgTime*sampleRate,1))
    window = windows.boxcar(nAvg)
    norm = sum(window)
    if len(data.shape) == 2:
        mean = np.mean(data,axis=1)
        window = np.vstack((window,window))
        return (oaconvolve((data.T-mean).T,window,mode='same',axes=1).T/norm + mean).T
    else:
        mean = np.mean(data)
        return oaconvolve(data-mean,window,mode='same')/norm + mean

def BoxcarDownsample(data,avgTime,sampleRate,returnRate=False):
    '''Integrates data by boxcar averaging blocks of duration avgTime.
    
    returns integrated data with reduced shape and optionally the reduced sample rate.
    -------------------------------------
    data:       single or dual channel data. dual channel should have shape (nSamples,2)
    avgTime:    duration of boxcar window in seconds
    sampleRate: sample rate of data in Hz.
    returnRate: boolean, default False. If True, returns the new rate, which may 
                    be slightly off from 1/avgTime due to rounding.
    '''
    nAvg = int(max(avgTime*sampleRate,1))
    if len(data.shape) == 2:
        nSamples = data.shape[1]
        data2 = data[:,:(nSamples//nAvg)*nAvg].reshape((2,nSamples//nAvg,nAvg))
    else:
        nSamples = len(data)
        data2 = data[:(nSamples//nAvg)*nAvg].reshape((nSamples//nAvg,nAvg))
    if returnRate:
        return np.mean(data2,axis=-1), sampleRate/nAvg
    return np.mean(data2,axis=-1)

# def BoxcarDownsampleCppAvg(data,avgTime,sampleRate,returnRate=False):
#     '''Integrates data by boxcar averaging blocks of duration avgTime.
    
#     returns integrated data with reduced shape and optionally the reduced sample rate.
#     -------------------------------------
#     data:       single or dual channel data. dual channel should have shape (nSamples,2)
#     avgTime:    duration of boxcar window in seconds
#     sampleRate: sample rate of data in Hz.
#     returnRate: boolean, default False. If True, returns the new rate, which may 
#                     be slightly off from 1/avgTime due to rounding.
#     '''
#     nAvg = int(max(avgTime*sampleRate,1))
#     if len(data.shape) == 2:
#         nSamples = data.shape[1]
#         data2 = data[:,:(nSamples//nAvg)*nAvg].reshape((2,nSamples//nAvg,nAvg))
#     else:
#         nSamples = len(data)
#         data2 = data[:(nSamples//nAvg)*nAvg].reshape((nSamples//nAvg,nAvg))
#     if returnRate:
#         return np.mean(data2,axis=-1), sampleRate/nAvg
#     return np.mean(data2,axis=-1)

def plotComplexHist(I,Q,bins=(80,80),figsize=[9,6],returnHistData=False):
    '''From I and Q data, plot a greyscale log histogram in complex plane.
    
    returns a pyplot subplot for further modification.
    ----------------------------------------------------
    I:          1 dim data array.
    Q:          data array with same size aas I
    bins:       (a,b) where a is # bins along I and b is along Q
    figsize:    passed to pyplot.figure to set figure size
    ----------------------------------------------------
    
    Example:
    h = plotComplexHist(DATA[:,0],DATA[:,1])
    h.set_xlabel('I [mV]')
    h.set_ylabel('Q [mV]')
    h.set_title('test')
    plt.savefig(r'path/to/save.png')
    '''
    fig =plt.figure(figsize=figsize,constrained_layout=True)
    h = fig.add_subplot()
    hs = plt.hist2d(I,Q,bins=bins,norm=LogNorm(),cmap=plt.get_cmap('Greys'))
    hb = plt.colorbar(hs[-1], shrink=0.9, extend='both')
    h.grid()
    h.set_aspect('equal')
    if returnHistData:
        return h, hs
    return h
        
def make_ellipses(gmm,ax,colors):
    '''Adds colored ellipses illustrating mean and std dev of mode to the given pyplot subplot.
    
    for each mode in gmm, plots an ellipse at the mean radius equal to std. dev.
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    ax:     pyplot subplot you want to add ellipses to
    colors: list of colors to use for ellipses. length must match number of modes.
    -----------------------------
    
    Example:
    h = plotComplexHist(DATA[:,0],DATA[:,1])
    make_ellipses(guassianMixture,h,['red','#fee090','magenta'])
    h.set_xlabel('I [mV]')
    h.set_ylabel('Q [mV]')
    h.set_title('test with ellipses')
    plt.savefig(r'path/to/save.png')
    '''
    for n, color in enumerate(colors):
        # get the covariance matrix for the mode associated with n trapped QPs
        covariances = gmm.covariances_[n][:2,:2]
        # v are the eigenvalues of covariance matrix, aka the variances along major and minor axis of ellipse. w are the eigenvectors. Order is smallest to v to largest v
        v, w = np.linalg.eigh(covariances)
        # normalize the eigenvector associated with the smallest variance, i.e., the variance along the minor axis.
        u = w[0] / np.linalg.norm(w[0])
        # get the angle from +x axis to minor axis of ellipse
        angle = 180*np.arctan2(u[1],u[0])/np.pi
        # v is now the diameter of the ellipse in minor, major order. It is equal to 2 std deviations.
        v = 2. *np.sqrt(v)
        # make the ellipse for mode n. Centered at mean with major and minor radius of 1 std deviation. and rotated to align to the data.
        ell = Ellipse(gmm.means_[n,:2],v[0],v[1],180+angle,color=color,fill=False)
        # now we just add the ellipses to the plot. Note that these ellipses shade the area in which all data points are within 1 std deviation of the mean.
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.8)
        ax.add_artist(ell)
        ax.set_aspect('equal','datalim')
        
def make_ellipsesHMM(hmm,ax,colors):
    '''Adds colored ellipses illustrating mean and std dev of mode to the given pyplot subplot.
    
    for each mode in gmm, plots an ellipse at the mean radius equal to std. dev.
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    ax:     pyplot subplot you want to add ellipses to
    colors: list of colors to use for ellipses. length must match number of modes.
    -----------------------------
    
    Example:
    h = plotComplexHist(DATA[:,0],DATA[:,1])
    make_ellipses(guassianMixture,h,['red','#fee090','magenta'])
    h.set_xlabel('I [mV]')
    h.set_ylabel('Q [mV]')
    h.set_title('test with ellipses')
    plt.savefig(r'path/to/save.png')
    '''
    for n, color in enumerate(colors):
        # get the covariance matrix for the mode associated with n trapped QPs
        covariances = hmm.covars_[n][:2,:2]
        # v are the eigenvalues of covariance matrix, aka the variances along major and minor axis of ellipse. w are the eigenvectors. Order is smallest to v to largest v
        v, w = np.linalg.eigh(covariances)
        # normalize the eigenvector associated with the smallest variance, i.e., the variance along the minor axis.
        u = w[0] / np.linalg.norm(w[0])
        # get the angle from +x axis to minor axis of ellipse
        angle = 180*np.arctan2(u[1],u[0])/np.pi
        # v is now the diameter of the ellipse in minor, major order. It is equal to 2 std deviations.
        v = 2. *np.sqrt(v)
        # make the ellipse for mode n. Centered at mean with major and minor radius of 1 std deviation. and rotated to align to the data.
        ell = Ellipse(hmm.means_[n,:2],v[0],v[1],180+angle,color=color,fill=False)
        # now we just add the ellipses to the plot. Note that these ellipses shade the area in which all data points are within 1 std deviation of the mean.
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.8)
        ax.add_artist(ell)
        ax.set_aspect('equal','datalim')

def make_ellipses2(means,varis,ax,colors):
    '''Adds colored ellipses illustrating mean and std dev of mode to the given pyplot subplot.
    
    for each mode in gmm, plots an ellipse at the mean radius equal to std. dev.
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    ax:     pyplot subplot you want to add ellipses to
    colors: list of colors to use for ellipses. length must match number of modes.
    -----------------------------
    
    Example:
    h = plotComplexHist(DATA[:,0],DATA[:,1])
    make_ellipses(guassianMixture,h,['red','#fee090','magenta'])
    h.set_xlabel('I [mV]')
    h.set_ylabel('Q [mV]')
    h.set_title('test with ellipses')
    plt.savefig(r'path/to/save.png')
    '''
    for n, color in enumerate(colors):
        # determine if first or second element in variances is minor axis
        if np.argmin(varis[n,:2]):
            minor = 2*varis[n,1]
            major = 2*varis[n,0]
            theta = 180*varis[n,2]/np.pi - 90
        else:
            minor = 2*varis[n,0]
            major = 2*varis[n,1]
            theta = 180*varis[n,2]/np.pi
        ell = Ellipse(means[n],minor,major,180-theta,color=color,fill=False)
        # now we just add the ellipses to the plot. Note that these ellipses shade the area in which all data points are within 1 std deviation of the mean.
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.8)
        ax.add_artist(ell)
        ax.set_aspect('equal','datalim')
        
def getGaussianSNR(gmm,mode1=0,mode2=1):
    '''from the scikit-lear gaussian mixture fit, get teh SNR between 2 given modes
    
    returns SNR in power units (V^2/sigma^2)
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    mode1:  index corresponding to desired mode
    mode2:  index corresponding to desired mode
    -----------------------------
    '''
    v01 = gmm.means_[mode2]-gmm.means_[mode1]
    u01 = v01 / np.linalg.norm(v01)
    var0_01 = np.linalg.multi_dot([u01,gmm.covariances_[mode1],u01])
    var1_01 = np.linalg.multi_dot([u01,gmm.covariances_[mode2],u01])
    snr01 = np.linalg.norm(v01)**2 / np.sqrt(var0_01*var1_01)
    return snr01

def getSNRhmm(hmm,mode1=0,mode2=1):
    '''from the scikit-lear gaussian mixture fit, get teh SNR between 2 given modes
    
    returns SNR in power units (V^2/sigma^2)
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    mode1:  index corresponding to desired mode
    mode2:  index corresponding to desired mode
    -----------------------------
    '''
    v01 = hmm.means_[mode2]-hmm.means_[mode1]
    u01 = v01 / np.linalg.norm(v01)
    var0_01 = np.linalg.multi_dot([u01,hmm.covars_[mode1],u01])
    var1_01 = np.linalg.multi_dot([u01,hmm.covars_[mode2],u01])
    snr01 = np.linalg.norm(v01)**2 / np.sqrt(var0_01*var1_01)
    return snr01

def getDataSNR(mode1data,mode2data):
    '''from selected subsets of data containing only a single mode, get SNR between 2 modes
    
    returns SNR in power units (V^2/sigma^2)
    -----------------------------
    gmm:    the scikit-learn Gaussian Mixture that fits your data
    mode1:  index corresponding to desired mode
    mode2:  index corresponding to desired mode
    -----------------------------
    '''
    v01 = np.mean(mode2data,axis=1)-np.mean(mode1data,axis=1)
    norm = np.linalg.norm(v01)
    u01 = v01 / norm
    var0_01 = np.linalg.multi_dot([u01,np.cov(mode1data),u01])
    var1_01 = np.linalg.multi_dot([u01,np.cov(mode2data),u01])
    snr01 = norm**2 / np.sqrt(var0_01*var1_01)
    return snr01

def predictWithBayes(gmm,data,nMemory=1):
    '''Use Bayesian inference alongside gaussian mixture probabilities to extract occupation
    
    returns estimated occupation as time series
    ---------------------------
    gmm:        the scikit-learn Gaussian Mixture that fits your data
    data:       nSamples x 2 array of data, columns correspond to I,Q
    nMemory:    number of previous samples to include in Bayesian inference
    ----------------------------
    '''
    probs = gmm.predict_proba(data)
    #bayes = probs[1:]*probs[:-1]/np.array([np.sum(probs[1:]*probs[:-1],axis=1),]*3).T
    bayes = probs[1:]*probs[:-1]/np.array([np.sum(probs[1:]*probs[:-1],axis=1),]*probs.shape[-1]).T
    for i in range(1,nMemory):
        #bayes = 1/(1+(1/bayes[:-1]-1)*(1-probs[i+1:])/probs[i+1:])
        #bayes = probs[i+1:]*bayes[:-1]/np.array([np.sum(probs[i+1:]*bayes[:-1],axis=1),]*3).T
        bayes = probs[i+1:]*bayes[:-1]/np.array([np.sum(probs[i+1:]*bayes[:-1],axis=1),]*probs.shape[-1]).T
    bayes = np.vstack((probs[:nMemory],bayes))
    return np.argmax(bayes,axis=1).astype(np.uint8)

def extractTimeBetweenTrapEvents(nEst,time):
    '''extracts the time between subsequent trapping events with no regard for which modes are involved.
    
    returns array of trap times in same units as given time series
    -----------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    T_id = np.diff(nEst,prepend=0) > 0
    return np.diff(time[T_id])

def extractLifetimes(nEst,time):
    '''extracts the times spent in each mode
    
    returns dictionary of arrays with keys '0','1','2',etc. corresponding to occupations
    ------------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    nModes = np.max(nEst) + 1
    dd = {}
    for i in range(nModes):
        mode = nEst == i
        mask = np.diff(mode,prepend=False)
        starts = time[mask][0::2]
        stops = time[mask][1::2]
        dd.update({str(i):stops - starts[:len(stops)]})
    return dd

def extractAntiLifetimes(nEst,time):
    '''extracts the times spent outside each mode
    
    returns dictionary of arrays with keys '0','1','2',etc. corresponding to occupations
    ------------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    nModes = np.max(nEst) + 1
    dd = {}
    for i in range(nModes):
        mode = nEst == i
        mask = np.diff(mode,prepend=False)
        starts = time[mask][0::2]
        stops = time[mask][1::2]
        dd.update({str(i):starts[1:] - stops[:len(starts)-1]})
    return dd

def extractDurations(nEst,time):
    '''extracts the time spent in traps with no regard for which modes are involved.
    
    returns array of trap durations in same units as given time series
    -----------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    diff = np.diff(nEst,prepend=np.int8(0))
    T_id = diff > 0
    R_id = diff < 0
    starts = np.repeat(time[T_id],diff[T_id])
    stops = np.repeat(time[R_id],np.abs(diff[R_id]))
    end = min(len(starts),len(stops))
    return stops[:end] - starts[:end]

def plotTauDist(dist,bins=80,color='grey',alpha=0.3,figsize=[9,6]):
    '''Creates new figure with given distribution as a histogram and returns nonzero bins with centers.
    
    returns pyplot subplot, nonzero bin counts, nonzero bin centers
    ---------------------------
    dist:   data to histogram
    bins:   passed to pyplot.hist
    color:  passed to pyplot.hist
    alpha:  passed to pyplot.hist
    figsize:    passed to pyplot.figure
    '''
    fig = plt.figure(figsize=figsize,constrained_layout=True)
    h = fig.add_subplot()
    hi = h.hist(dist,bins=bins,color=color,alpha=alpha,density=True)
    # h.set_xlim(hi[1][1],hi[1][-1])
    # h.set_ylim(0,1.5*np.max(hi[0][3:]))
    binmask = np.array(hi[0] > 0,dtype=bool)
    BinCenters = (hi[1][1:] + hi[1][:-1])/2
    return h, hi[0][binmask], BinCenters[binmask]


def plotTimeSeries(data,nEst,time,start,stop,zeroTime=False):
    '''
    generates a new figure with 2 panel subplots comparing IQ data and 
    occupation for subset of time between start and stop.

    Parameters
    ----------
    data : array with shape (2, # samples)
        IQ data.
    nEst : array with shape (# samples)
        the occupation.
    time : array with shape (# samples)
        times corresponding to data.
    start : float
        start time for plotting (same units as parameter time)
    stop : flaot
        stop time for plotting (same units as parameter time).

    Returns
    -------
    fig : pyplot figure handle
    ax : pyplot axes for subplots
        shape is (2,) with ax[0] addressing the top panel.

    '''
    fig,ax = plt.subplots(2,1,figsize=[9,6],constrained_layout=True)
    mask = np.logical_and(time >= start,time <=stop)
    plttime = time[mask] - time[mask][0] if zeroTime else time[mask]
    ax[0].plot(plttime,data[0][mask],label='I')
    ax[0].plot(plttime,data[1][mask],label='Q')
    ax[0].legend()
    ax[1].plot(plttime,nEst[mask],label='QP #')
    ax[1].legend()
    return fig,ax


def extractLifetimesWithModes(nEst,time):
    '''extract the times spent in mode i before tranfering to mode j
    
    returns dictionary of arrays with keys 'ij' for 1/rate from mode i to mode j.
    ------------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    nModes = np.max(nEst) + 1
    dd = {}
    for i in range(nModes):
        mode = nEst == i
        mask = np.diff(mode,prepend=False)
        starts = time[mask][0::2]
        stops = time[mask][1::2]
        toMode = nEst[mask][1::2]
        durations = stops - starts[:len(stops)]
        # modemasks = {str(j):toMode == j for j in range(nModes)}
        dd.update({str(i)+str(j):durations[toMode == j] for j in range(nModes)})
    return dd


def extractLifetimesWithTwoModes(nEst,time):
    '''extract the lifetime of mode j conditioned on being in mode i previously and mode k afterwards.
    
    returns dictionary of arrays with keys 'ijk' for 1/rate from j to k conditioned on being in mode i previously.
    ------------------------------------
    nEst:   the predicted occupation
    time:   time data with same size as nEst
    ------------------------------------
    '''
    nModes = np.max(nEst) + 1
    dd = {}
    mask = np.diff(nEst,prepend=False).astype(bool)
    mask[0] = True
    modei = nEst[mask][:-2]
    modej = nEst[mask][1:-1]
    modek = nEst[mask][2:]
    durations = np.diff(time[mask])[1:]
    dd.update({str(i)+str(j)+str(k):durations[np.logical_and(np.logical_and(modei == i,modej == j), modek == k)] for i in range(nModes) for j in range(nModes) for k in range(nModes)})
    return dd

def exp(t,a,tau):
    '''simple exponential function for fitting'''
    return a*np.exp(-t/tau)

def fitExpDecay(dist,t,cut=0,returnTauDetector=True,returnSGDIST=False):
    '''estimates the detection rate and fits the distribution to exponential.
    
    returns pars, cov from scipy.optimize.curve_fit and optionally the detector timescale.
    --------------------------------------
    dist:   data, presumably lifetimes between events
    t:      times corresponding to dist, must have same size
    returnTauDetector:  Boolean, if False, only returns pars, cov.
    '''
    cutmask = t >= cut
    window = max(int(len(dist)*0.04),5)
    window += 0 if window%2 else 1 # ensure window is odd
    sgdist = savgol_filter(dist,window,3)
    tdetInd = np.argmax(sgdist)
    tauDetector = t[tdetInd]
    ampGuess = 1.2*sgdist[tdetInd]
    mask = sgdist[tdetInd:] < ampGuess/np.e
    tauGuess = t[tdetInd:][mask][0]
    # tauGuess = cut
    pars, cov = curve_fit(exp,t[cutmask],dist[cutmask],p0=[ampGuess,tauGuess])
    if returnSGDIST and returnTauDetector:
        return pars, cov, tauDetector, sgdist
    elif returnTauDetector:
        return pars, cov, tauDetector
    elif returnSGDIST:
        return pars, cov, sgdist
    else:
        return pars, cov
    
def fitAndPlotExpDecay(dist,cut=None,returnTaus=False,figsize=[3.325,3.325],bins=100):
    '''plots a histogram of distribution and fits it to an exponential decay with lot's of formatting already done.
    
    returns a pyplot subplot handle for further editing and saving.
    --------------------------------------
    dist:   data, presumably lifetimes between events. Assumed units are microseconds
    cut:    optional, the fit is performed to values greater than this. If None, the cut will be taken as the mean of the distribution.
    returnTaus:  optional, boolean. If True, a list of times is returned as [tau from exp fit, fit error, tau detector]
    figsize: optional, passed to pyplot figure
    bins: optional, number of bins for histogram
    '''
    if cut is None:
        cut = np.mean(dist)
    h,hi,bc = plotTauDist(dist,bins=bins,figsize=figsize);
    pars,cov,taud,sgdist = fitExpDecay(hi,bc,cut=cut,returnSGDIST=True);
    perr = np.sqrt(np.diag(cov)) # 1 sigma error on fit parameters
    fit = exp(bc,*pars);
    lowb = exp(bc,pars[0]-perr[0],pars[1]-perr[1])
    uppb = exp(bc,pars[0]+perr[0],pars[1]+perr[1])
    h.plot(bc,fit,color='darkgreen',label='fit $\\tau = {:6.1f}\pm{:6.1f} \\mu s$'.format(pars[1],perr[1]));
    h.fill_between(bc,uppb,lowb,color='lightgreen')
    # h.plot(bc,sgdist,color='purple',label='sav-gol filtered');
    h.axvline(cut,color='magenta',ls='dashdot',label='cutoff $= {:6.1f} \\mu$s'.format(cut))
    h.axvline(taud,color='red',ls='dashed',label='$\\tau_d = {:6.1f} \\mu$s'.format(taud))
    h.legend();
    h.set_xlabel('Time [$\\mu$s]')
    if returnTaus:
        taus = [pars[1],perr[1],taud]
        return h, taus
    else:
        return h
    

def plotBurstSearch(nEst,avgTime,sampleRate,method='boxcar'):
    '''performs a rolling average of the estimated occupation and generates a 
    log scale histogram of results for visual estimation of the appropriate 
    threshold for burst events.

    Parameters
    ----------
    nEst : numpy array
        The estimated QP occupation as given by predictWithBayes().
    avgTime : float or int
        Duration of the rolling average window.
    sampleRate : float or int
        The sample rate of data.

    Returns
    -------
    h : matplotlib subplot
        matplotlib.axes._subplots.AxesSubplot
    burstSearch : numpy array
        the rolling average of occupation.
    '''
    if method == 'flattop':
        burstSearch = (FlattopConvolution(nEst,avgTime,sampleRate)*(2**8/np.max(nEst))).astype(np.uint8)
    elif method == 'gaussian':
        burstSearch = (GaussianConvolution(nEst,avgTime,sampleRate)*(2**8/np.max(nEst))).astype(np.uint8)
    else:
        burstSearch = (BoxcarConvolution(nEst,avgTime,sampleRate)*(2**8/np.max(nEst))).astype(np.uint8)
    fig = plt.figure(figsize=[9,6],constrained_layout=True)
    h = fig.add_subplot()
    hi = h.hist(burstSearch,bins=100,log=True)
    h.set_xlabel('burstSearch value')
    return h, burstSearch

def getBurstIndices(burstSearch,threshold,nConsecutive=5):
    '''Get the indices of detected burst events, given a threshold from visual 
    inspection of plotBurstSearch().

    Parameters
    ----------
    burstSearch : numpy array
        The rolling average of occupation.
    threshold : float or int
        values above threshold are considered burst events.
    nConsecutive : int, optional
        How many consecutive points in burstSearch need to be above threshold
        to count as an independent burst event. This keeps us from getting 
        multiple 'events' as the rolling average moves through threshold. 
        The default is 5.

    Returns
    -------
    burstIndices : boolean array
        The indices corresponding to burst events. Can be used with your time
        array to extract the times of burst events.

    '''
    mask = burstSearch > threshold
    d = np.diff(mask,prepend=np.int8(0))
    start = np.where(d == 1)[0]
    end = np.where(d == -1)[0]
    mask2 = (end - start) >= nConsecutive
    burstIndices = start[mask2]
    return burstIndices

def getRates(nEst,sampleRate):
    '''
    gets the mean trap and release rates for nEst. 
    Primarily used for quick estimation of the rolling poisson rates for 
    burst searches

    Parameters
    ----------
    nEst : array with shape (# samples)
        occupation.
    sampleRate : float
        the sample rate of nEst.

    Returns
    -------
    meanT : float
        The mean trapping rate in same units as provided sampleRate.
    meanR : float
        The mean release rate in same units as provided sampleRate.

    '''
    d = np.diff(nEst,prepend=np.int8(0))
    Tid = d > 0
    Rid = d < 0
    traps = np.diff(np.hstack((0,Tid.nonzero()[0])))/sampleRate
    starts = np.repeat(Tid.nonzero()[0],d[Tid])
    stops = np.repeat(Rid.nonzero()[0],np.abs(d[Rid]))
    end = min(len(starts),len(stops))
    releases = (stops[:end] - starts[:end])/sampleRate
    meanT = 1/np.mean(traps)
    meanR = 1/np.mean(releases)
    return meanT, meanR

def getRollingRates(nEst,sampleRate,windowDuration):
    '''
    uses function getRates() to estimate the trap and release rates 
    for windowDuration length subsets of nEst

    Parameters
    ----------
    nEst : array with shape (# samples)
        occupation.
    sampleRate : float
        the sample rate of nEst.
    windowDuration : float
        length of time blocks to break nEst into for rate analysis.

    Returns
    -------
    rollingRates : array with shape (2, # samples//(windowDuration*sampleRate))
        first row is trapping rates, second row is release.

    '''
    window = int(max(windowDuration*sampleRate,1))
    return np.apply_along_axis(getRates,0,nEst[:(nEst.size//window)*window].reshape((window,nEst.size//window),order='f'),sampleRate)

# def gaussian(widths,means):
#     """Returns a gaussian function with the given parameters"""
#     width_x = float(widths[0])
#     width_y = float(widths[1])
#     return lambda x,y: np.exp(
#                 -(((means[0]-x)/width_x)**2+((means[1]-y)/width_y)**2)/2
#                 )/(2*np.pi*width_x*width_y)

def getGaussian(args,*p,means=None,varis=None):
    '''
    get scaled gaussian distribution for single or multiple modes with given parameters.

    Parameters
    ----------
    args : 2-tuple
        (xx,yy) where xx and yy are N x N arrays like np.meshgrid(). 
        This is the field over which gaussian is calculated.
    p : 1-d array-like variable length.
        a list of parameters for each mode in gaussian distribution.
        for M modes in distribution, should have length M or M*6 per conditions below.
        IF means and varis are None, follows order: amplitude,x0,y0,sigmax,sigmay,angle.
        IF means and varis ARE PROVIDED, then p contains only amplitudes
    means : array with shape (M,2), optional
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional
        variances and angle of each mode. each row is [sigma x, sigma y, theta]
        
    Returns
    -------
    array with shape N x N.
        the scaled gaussian distribution with M modes.

    '''
    gg = _gaussian(args,*p,means=means,varis=varis)
    ii = int(np.sqrt(gg.size))
    return gg.reshape((ii,ii),order='f')


def fitGaussian(histDATA,guess,means=None,varis=None):
    '''
    from histogram data and initial guess, fit the distribution to M gaussians.

    Parameters
    ----------
    histDATA : tuple with (histogram z data, histogram x edges, histogram y edges)
        histogram data as returned by np.histogram2d() or plt.hist2d().
    guess : list or 1-d array.
        a list of parameters for each mode in gaussian distribution.
        for M modes in distribution, should have length M or M*6 per conditions below.
        IF means and varis are None, follows order: amplitude,x0,y0,sigmax,sigmay,angle.
        IF means and varis ARE PROVIDED, then contains only amplitudes
    means : array with shape (M,2), optional
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional
        variances and angle of each mode. each row is [sigma x, sigma y, theta]

    Returns
    -------
    xx : N x N array.
        results of np.meshgrid() on bin centers.
    yy : N x N array.
        results of np.meshgrid() on bin centers.
    amplitudes : array with shape (M,)
        the amplitude of each gaussian, such that integrating over the mode gives amplitude.
    means : array with shape (M,2), optional. returned only if means and varis are not provided.
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional. returned only if means and varis are not provided.
        variances and angle of each mode. each row is [sigma x, sigma y, theta]        

    '''
    mask = ~(histDATA[0]>0)
    xx,yy = np.meshgrid((histDATA[1][:-1]+histDATA[1][1:])/2,(
        histDATA[2][:-1]+histDATA[2][1:])/2)
    xxm = np.ma.masked_array(xx,mask)
    yym = np.ma.masked_array(yy,mask)
    datam = np.ma.masked_array(histDATA[0],mask)
    callGaussian = lambda args,*p: _gaussian(args,*p,means=means,varis=varis)
    if type(means) is type(None):
        bound = ([0,-np.inf,-np.inf,0,0,-np.pi]*(len(guess)//6),
                 [np.inf,np.inf,np.inf,np.inf,np.inf,np.pi]*(len(guess)//6))
        pars,cov = curve_fit(callGaussian,(xxm,yym),datam.ravel(),p0=guess,bounds=bound)
        amp = pars[::6]
        mean = np.array([pars[i+1:i+3] for i in range(0,len(pars),6)])
        vari = np.array([pars[i+3:i+6] for i in range(0,len(pars),6)])
        return xx,yy,amp,mean,vari
    else:
        bound = (0,np.inf)
        amp,cov = curve_fit(callGaussian,(xxm,yym),datam.ravel(),p0=guess,bounds=bound)
        return xx,yy,amp

def fitGaussiansIndependent(quietSubs):
    '''
    Fit each mode independently to get the means and variances

    Parameters
    ----------
    quietSubs : list
        each element is a subset of DATA in with unchanging occupation.

    Returns
    -------
    amps : array with shape (M,)
        amplitudes for M gaussian modes.
    means : array with shape (M,2)
        centers for M gaussian modes.
    varis : array with shape (M,3)
        sigma X, sigma Y, and angle for M gaussian modes.

    '''
    amps = []
    means = []
    varis = []
    for n in range(len(quietSubs)):
        mean = np.mean(quietSubs[n],axis=1)
        hi = np.histogram2d(quietSubs[n][0],quietSubs[n][1],bins=(80,80))
        xx,yy,A,M,V = fitGaussian(hi,[np.max(hi[0]),mean[0],mean[1],1,1,0])
        amps.append(A)
        means.append(M)
        varis.append(V)    
    amps = np.squeeze(amps)
    means = np.squeeze(means)
    varis = np.squeeze(varis)
    return amps, means, varis

def getGaussianProbabilities(DATA,amps,means,varis):
    '''
    gets the posterior probability of each mode conditioned at each point in DATA.

    Parameters
    ----------
    DATA : array with shape (2, # samples)
        The alazar data.
    amps : array with shape (M, 2)
        the amplitude of each gaussian, such that integrating over the mode gives amplitude.
    means : array with shape (M,2)
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3)
        variances and angle of each mode. each row is [sigma x, sigma y, theta]

    Returns
    -------
    prob : array with shape (# samples, M)
        each row in prob corresponds to the posterior probabilies of 
        modes 0,...,M conditioned on that point in data.

    '''
    weights = np.asarray(amps)/np.sum(amps)
    prob = np.array([(W/A)*_gaussian((DATA[0],DATA[1]),*(A,),means=(M,),varis=(V,)) for W,A,M,V in zip(weights,amps,means,varis)])
    prob = prob/np.sum(prob,axis=0)
    return prob.T

def predictWithBayes2(probs,nMemory=1):
    '''Use Bayesian inference alongside gaussian probabilities to extract occupation
    
    returns estimated occupation as time series
    ---------------------------
    probs:      The gaussian posterior probabilities of data, as returned
                    by getGaussianProbabilities()
    nMemory:    number of previous samples to include in Bayesian inference
    ----------------------------
    '''
    #bayes = 1/(1+(1/probs[:-1]-1)*(1-probs[1:])/probs[1:])
    #bayes = probs[1:]*probs[:-1]/np.array([np.sum(probs[1:]*probs[:-1],axis=1),]*3).T
    bayes = probs[1:]*probs[:-1]/np.array([np.sum(probs[1:]*probs[:-1],axis=1),]*probs.shape[-1]).T
    for i in range(1,nMemory):
        #bayes = 1/(1+(1/bayes[:-1]-1)*(1-probs[i+1:])/probs[i+1:])
        #bayes = probs[i+1:]*bayes[:-1]/np.array([np.sum(probs[i+1:]*bayes[:-1],axis=1),]*3).T
        bayes = probs[i+1:]*bayes[:-1]/np.array([np.sum(probs[i+1:]*bayes[:-1],axis=1),]*probs.shape[-1]).T
    bayes = np.vstack((probs[:nMemory],bayes))
    return np.argmax(bayes,axis=1).astype(np.uint8)

def _gaussian(args,*p,means=None,varis=None):
    '''
    A arbitrary sum of normalized gaussian distributions with scaling amplitudes.

    Parameters
    ----------
    args : tuple (xx,yy)
        xx and yy can either by field points such as np.meshgrid() or IQ data.
    *p : 1-d array-like variable length. MUST BE CALLED WITH PRECEEDING ASTERISK
        asterisk is required so python treats this as multiple arguments rather than a single list.
        a list of parameters for each mode in gaussian distribution.
        for M modes in distribution, should have length M or M*6 per conditions below.
        IF means and varis are None, follows order: amplitude,x0,y0,sigmax,sigmay,angle.
        IF means and varis ARE PROVIDED, then p contains only amplitudes
    means : array with shape (M,2), optional
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional
        variances and angle of each mode. each row is [sigma x, sigma y, theta]

    Returns
    -------
    Z : 1-d array
        the values of gaussian distribution at each point.

    '''
    xx,yy = args
    if type(means) is not type(None) and type(varis) is not type(None):
        varis = np.array(varis)
        means = np.array(means)
        if np.ndim(varis) == 1:
            varis = np.array([varis,])
        if np.ndim(means) == 1:
            varis = np.array([means,])
        a = (np.cos(varis[:,2])**2)/(varis[:,0]**2) + (np.sin(varis[:,2])**2)/(varis[:,1]**2)
        b = (-(np.sin(2*varis[:,2]))/(2*varis[:,0]**2) + (np.sin(2*varis[:,2]))/(2*varis[:,1]**2))
        c = (np.sin(varis[:,2])**2)/(varis[:,0]**2) + (np.cos(varis[:,2])**2)/(varis[:,1]**2)
        detInvSIG = a*c-b**2
        z = np.sum([p[i]*np.exp(
            -(a[i]*(means[i,0]-xx)**2 + 
              2*b[i]*(means[i,0]-xx)*(means[i,1]-yy) + 
              c[i]*(means[i,1]-yy)**2)/2
            )/(2*np.pi*np.sqrt(1/detInvSIG[i])) for i in range(0,len(p))],axis=0)
    else:
        z = np.sum([p[i]*np.exp(
            -(((np.cos(p[i+5])**2)/(p[i+3]**2) + (np.sin(p[i+5])**2)/(p[i+4]**2))*(p[i+1]-xx)**2 +
            ((np.sin(p[i+5])**2)/(p[i+3]**2) + (np.cos(p[i+5])**2)/(p[i+4]**2))*(p[i+2]-yy)**2 +
            2*(-(np.sin(2*p[i+5]))/(2*p[i+3]**2) + (np.sin(2*p[i+5]))/(2*p[i+4]**2))*(p[i+1]-xx)*(p[i+2]-yy))/2
            )/(2*np.pi*np.sqrt(1/(
                (((np.cos(p[i+5])**2)/(p[i+3]**2) + (np.sin(p[i+5])**2)/(p[i+4]**2))*
                ((np.sin(p[i+5])**2)/(p[i+3]**2) + (np.cos(p[i+5])**2)/(p[i+4]**2))) - 
                (-(np.sin(2*p[i+5]))/(2*p[i+3]**2) + (np.sin(2*p[i+5]))/(2*p[i+4]**2))**2
                ))
                ) for i in range(0,len(p),6)],axis=0)
        
    return z.ravel('f')

def getQuietSubsets(DATA,nEst,time,thresholds):
    '''
    stitch together periods of DATA in which the occupation is unchanged for
    longer than the given threshold for that mode.

    Parameters
    ----------
    DATA : array with shape (2, # samples)
        The alazar data.
    nEst : array with shape (# samples)
        the occupation
    time : array with shape (# samples)
        the times corresponding to data.
    thresholds : list or 1-d array
        For each mode, the minimum time the occupation must be unchanged for a 
        segment of DATA to be stiched into the quietSubsets.

    Returns
    -------
    quietSubs : list
        each element of the list is a subset of DATA in which the occupation
        does not change. elements are in order of ascending QP occupation.

    '''
    quietSubs = []
    lStops = []
    lStarts = []
    for i in range(len(thresholds)):
        mask = np.diff(nEst == i,prepend=False)
        starts = time[mask][0::2]
        stops = time[mask][1::2]
        lStarts.append(starts[:len(stops)])
        lStops.append(stops)
        mask = np.argwhere(stops - starts[:len(stops)] > thresholds[i])[:,0]
        quietSubs.append(
            np.hstack([DATA[np.array([np.logical_and(
                time > starts[mask[j]]+0.05*thresholds[i],
                time < stops[mask[j]]-0.05*thresholds[i]),]*2
                )].reshape(2,-1) for j in range(len(mask))]))
    return quietSubs

def getQuietSubsets2(DATA,nEst,nSubs=100):
    '''
    stitch together the <nSubs> longest periods of occupation of each mode.
    
    Parameters
    ----------
    DATA : array with shape (2, # samples)
        The alazar data.
    nEst : array with shape (# samples)
        the occupation
    nSubs: how many subsets to stitch together

    Returns
    -------
    quietSubs : list
        each element of the list is a subset of DATA in which the occupation
        does not change. elements are in order of ascending QP occupation.

    '''
    quietSubs = []
    now = perf_counter()
    time = np.arange(len(nEst))
    for i in range(max(nEst)+1):
        try:
            N = int(nSubs)
        except (AttributeError, TypeError):
            raise AssertionError('nSubs should be an integer')
        mask = np.diff(nEst == i, prepend=False)
        print('mode {} selected at {}'.format(i,perf_counter()-now))
        starts = time[mask][0::2]
        stops = time[mask][1::2]
        durations = stops - starts[:len(stops)]
        print('durations found in {}'.format(perf_counter()-now))
        if N > 0.25*len(durations):
            N = int(0.25*len(durations))
            print('nSubs has been updated to {} for mode {} with {} counts'.format(N, i, len(durations)))
        mask2 = np.argsort(durations)[-N:]
        print('sorted mask found in {}'.format(perf_counter()-now))
        cut = 0.05*durations[mask2][0]
        mask3 = np.any([np.logical_and(time > starts[mask2[j]]+cut,time < stops[mask2[j]]-cut) for j  in range(len(mask2))],axis=0)
        quietSubs.append(DATA[np.array([mask3,]*2)].reshape(2,-1))
        print('mode {} data stitched in {}'.format(i,perf_counter()-now))
    return quietSubs

def plotQuietSubsets(quietSubs):
    fig = plt.figure(figsize=[3.325,3.325],constrained_layout=True)
    h = fig.add_subplot()
    colors = ['Reds','Oranges','Greens','Blues','Purples']
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for n in range(len(quietSubs)):
        h.hist2d(quietSubs[n][0],quietSubs[n][1],bins=(50,50),norm=LogNorm(),cmap=plt.get_cmap(colors[n%len(colors)]))
        xmins.append(np.min(quietSubs[n][0]))
        xmaxs.append(np.max(quietSubs[n][0]))
        ymins.append(np.min(quietSubs[n][1]))
        ymaxs.append(np.max(quietSubs[n][1]))
    xmin = np.min(xmins) - 1
    xmax = np.max(xmaxs) + 1
    ymin = np.min(ymins) - 1
    ymax = np.max(ymaxs) + 1
    h.set_xlim(xmin,xmax)
    h.set_ylim(ymin,ymax)
    return h 
        

def getQuietThresholds(lifetimes):
    '''
    Obtain the thresholds for getQuietSubsets() from the distribution of 
    lifetimes in each mode, as returned by extractLifetimes()

    Parameters
    ----------
    lifetimes : dictionary
        keys are '0','1','2',etc. each element is the distribution of lifetimes
        in that mode. Typically would come from calling extractLifetimes()

    Returns
    -------
    thresholds : list
        Each element is a threshold for obtaining .

    '''
    thresholds = []
    for key in lifetimes:
        # print('\nmode {}'.format(key))
        # ma = np.max(lifetimes[key])
        me = np.mean(lifetimes[key])
        hist = np.histogram(lifetimes[key],bins=100)
        tcent = (hist[1][:-1]+hist[1][1:])/2
        pars,cov = fitExpDecay(hist[0],tcent,cut=me,returnTauDetector=False)
        # fd = (ma/me)*(ma-me)
        # fd = 30*(ma-me)/ma
        # print('max is {}'.format(ma))
        # print('mean is {}'.format(me))
        # thresh = min(fd*me,0.6*ma)
        thresh = 4*pars[1]
        # print('threshold would be {}'.format(thresh))
        thresholds.append(thresh)
    return thresholds

# def gaussianMix(heights,widths,means):
#     return lambda x,y: np.sum([h*gaussian(w,m)(x,y)for h,m,w in zip(heights,means,widths)],axis=0)

# def fitgaussian(histDATA,widths,means):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution found by a fit"""
#     xx,yy = np.meshgrid((histDATA[1][:-1]+histDATA[1][1:])/2,(
#         histDATA[2][:-1]+histDATA[2][1:])/2)
#     params = [widths,means]
#     p0 = np.repeat(np.max(histDATA[0]),len(means))
#     callGaussian = lambda d,*p: np.ravel(gaussianMix(np.array(p),*params)(xx,yy))
#     # errorfunction = lambda p,params: np.ravel(gaussianMix(
#     #     p,*params)(xx,yy) - histDATA[0])
#     # p, success = leastsq(errorfunction, p0, args = params)
#     pars,cov = curve_fit(callGaussian,np.array([xx,yy]),histDATA[0].ravel(),p0=p0)
#     return xx,yy,pars

def subtractionGaussianFit(DATA,nModes):
    '''
    Fit each mode independently to get the means and variances

    Parameters
    ----------
    quietSubs : list
        each element is a subset of DATA in with unchanging occupation.

    Returns
    -------
    amps : array with shape (M,)
        amplitudes for M gaussian modes.
    means : array with shape (M,2)
        centers for M gaussian modes.
    varis : array with shape (M,3)
        sigma X, sigma Y, and angle for M gaussian modes.

    '''
    # get the histogram results
    (hi,x,y) = np.histogram2d(DATA[0],DATA[1],bins=(100,100))
    hismooth = gaussian_filter(hi,[1,1],mode='constant')
    am = hismooth.argmax()
    d = hi.shape[1]
    xc = (x[:-1]+x[1:])/2
    yc = (y[:-1]+y[1:])/2
    cent = [xc[am//d],yc[am%d]]
    print('center for mode {} estimated at {}'.format(0,cent))
    xx,yy = np.meshgrid(xc,yc)
    rads = 2.5*np.sqrt(np.diag(np.cov(DATA)))
    mask = (xx-cent[0])**2 + (yy-cent[1])**2 < np.mean(rads)**2
    masks = []
    h = plotComplexHist(DATA[0],DATA[1],figsize=[3.325,3.325])
    make_ellipses2(np.array([cent,]), np.array([np.append(rads,[0,]),]), h, ['red',])
    plt.show();
    plt.close();
    amps = []
    means = []
    varis = []
    for n in range(nModes):
        # fit the masked array
        masks.append(mask)
        A,M,V = _fitGaussianMasked(~mask,(hi,x,y),[np.max(hismooth),cent[0],cent[1],1,1,0])
        print('mean for mode {} is fit to {}'.format(n,M[0]))
        amps.append(A)
        means.append(M)
        varis.append(V)  
        h = plotComplexHist(DATA[0],DATA[1],figsize=[3.325,3.325])
        make_ellipses2(np.array([np.squeeze(M),]), np.array([np.squeeze(V),]), h, ['blue',])
        plt.show();
        plt.close();
        # subtract off the fit to first mode
        hi = hi - getGaussian((xx,yy), [A,M[0][0],M[0][1],V[0][0],V[0][1],V[0][2]]).T
        # find the next mode and make a new mask around it
        hismooth = gaussian_filter(hi,[1,1],mode='constant')
        hismooth[np.sum(masks,axis=0,dtype=bool).T] = 0
        fig,ax = plt.subplots()
        plt.pcolormesh(xc,yc,hismooth.T,norm=LogNorm())
        plt.colorbar()
        plt.xlim(min(xc),max(xc))
        plt.ylim(min(yc),max(yc))
        am = hismooth.argmax()
        print(hismooth.max(),hismooth[am//d,am%d])
        cent = [xc[am//d],yc[am%d]]
        print(cent)
        make_ellipses2(np.array([cent,]), np.array([np.append(rads,[0,]),]), ax, ['red',])
        plt.show()
        plt.close()
        print('center for mode {} estimated at {}'.format(n+1,cent))
        mask = (xx-cent[0])**2 + (yy-cent[1])**2 < np.mean(rads)**2
        plt.contourf(np.ma.masked_array(xx,~mask),np.ma.masked_array(yy,~mask),np.ma.masked_array(hi,~mask))
        #plt.set_aspect('equal')
        plt.colorbar()
        plt.show()
        plt.close()
    amps = np.squeeze(amps)
    means = np.squeeze(means)
    varis = np.squeeze(varis)
    return amps, means, varis

def _fitGaussianMasked(mask,histDATA,guess,means=None,varis=None):
    '''
    from histogram data and initial guess, fit the distribution to M gaussians.

    Parameters
    ----------
    mask:   boolean array which is True where data should be excluded. i.e., we fit the parts that are False.
    histDATA : tuple with (histogram z data, histogram x edges, histogram y edges)
        histogram data as returned by np.histogram2d() or plt.hist2d().
    guess : list or 1-d array.
        a list of parameters for each mode in gaussian distribution.
        for M modes in distribution, should have length M or M*6 per conditions below.
        IF means and varis are None, follows order: amplitude,x0,y0,sigmax,sigmay,angle.
        IF means and varis ARE PROVIDED, then contains only amplitudes
    means : array with shape (M,2), optional
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional
        variances and angle of each mode. each row is [sigma x, sigma y, theta]

    Returns
    -------
    xx : N x N array.
        results of np.meshgrid() on bin centers.
    yy : N x N array.
        results of np.meshgrid() on bin centers.
    amplitudes : array with shape (M,)
        the amplitude of each gaussian, such that integrating over the mode gives amplitude.
    means : array with shape (M,2), optional. returned only if means and varis are not provided.
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional. returned only if means and varis are not provided.
        variances and angle of each mode. each row is [sigma x, sigma y, theta]        

    '''
    mask2 = ~np.array(histDATA[0] > 0,dtype=bool)
    xx,yy = np.meshgrid((histDATA[1][:-1]+histDATA[1][1:])/2,(
        histDATA[2][:-1]+histDATA[2][1:])/2)
    xxm = np.ma.masked_array(xx,mask)
    yym = np.ma.masked_array(yy,mask)
    datam = np.ma.masked_array(histDATA[0],mask.T+mask2)
    callGaussian = lambda args,*p: _gaussian(args,*p,means=means,varis=varis)
    if type(means) is type(None):
        bound = ([0,-np.inf,-np.inf,0,0,-np.pi]*(len(guess)//6),
                 [np.inf,np.inf,np.inf,np.inf,np.inf,np.pi]*(len(guess)//6))
        pars,cov = curve_fit(callGaussian,(xxm,yym),datam.ravel(),p0=guess,bounds=bound)
        amp = pars[::6]
        mean = np.array([pars[i+1:i+3] for i in range(0,len(pars),6)])
        vari = np.array([pars[i+3:i+6] for i in range(0,len(pars),6)])
        return amp,mean,vari
    else:
        bound = (0,np.inf)
        amp,cov = curve_fit(callGaussian,(xxm,yym),datam.ravel(),p0=guess,bounds=bound)
        return amp

def PoissonCorrection(ti,td,tinot,tdnot):
    '''
    from exponential fits to the tails of lifetime distributions, apply the 
    correction for finite measurement bandwidth.
    
    parameters
    ----------
    ti:     lifetime in mode
    td:     detector time in mode
    tinot:  lifetime out of mode
    tdnot:  detector time out of mode
    
    returns
    -------
    tau:    corrected lifetime in mode
    '''
    vs = tdnot/tinot
    us = td/ti
    u = us*((1-us**2-vs**2)/(1-us-vs))-us**2
    return td/u

def getTransRatesFromProb(sr,trmat):
    '''
    From the transition matrix of a hidden markov model, get the transition
    rates between different modes.
    
    parameters
    ----------
    sr:     sample rate of HMM data
    trmat:  HMM transition matrix
    
    returns
    -------
    taus:   Array whose elements are the transition rates between modes
    '''
    rates = np.empty(np.shape(trmat))
    n1, n2 = np.shape(trmat)
    for i in range(n1):
        for j in range(n2):
            if i == j:
                rates[i,j] = -sr*np.log(trmat[i,j])
            else:
                rates[i,j] = -sr*lambertw(-trmat[i,j]).real
    return rates


if __name__ == '__main__':
    data = np.vstack((np.random.normal(loc=5,size=1000000),np.random.normal(loc=-5,size=1000000)))
    data = np.hstack((data,np.vstack((np.random.normal(loc=-3,size=1000000),np.random.normal(loc=8,size=1000000)))))
    data = np.hstack((data,np.vstack((np.random.normal(loc=0.1,size=1000000),np.random.normal(loc=4,size=1000000)))))
    means = np.array([[5,-5],[-3,8],[0.1,4]])
    widths = np.ones(means.shape)
    
    fig =plt.figure(figsize=[9,6],constrained_layout=True)
    h = fig.add_subplot()
    h.set_aspect('equal')
    hi = plt.hist2d(data[0],data[1],bins=(80,80),norm=LogNorm(),cmap=plt.get_cmap('Greys'))
    # hi = np.histogram2d(data[0],data[1],bins=(200,200))
    xc = (hi[1][:-1]+hi[1][1:])/2
    yc = (hi[2][:-1]+hi[2][1:])/2
    guess = [60000,5,-5,1,1,0,
             60000,-3,8,1,1,0,
             60000,0.1,4,1,1,0]
    xx,yy,amps,means,varis = fitGaussian(hi,guess)
    # f = gaussianMix(heights,widths,means)
    fit = getGaussian((xx,yy),*amps,means = means,varis = varis)
    h.contour(xc,yc,fit)
    
    fig =plt.figure(figsize=[9,6],constrained_layout=True)
    h = fig.add_subplot()
    h.set_aspect('equal')
    hi = plt.hist2d(data[0],data[1],bins=(80,80),norm=LogNorm(),cmap=plt.get_cmap('Greys'))
    # hi = np.histogram2d(data[0],data[1],bins=(200,200))
    xc = (hi[1][:-1]+hi[1][1:])/2
    yc = (hi[2][:-1]+hi[2][1:])/2
    # amps = pars[::6]
    # means = np.array([pars[i+1:i+3] for i in range(0,len(pars),6)])
    # varis = np.array([pars[i+3:i+6] for i in range(0,len(pars),6)])
    guess = [60000,
             60000,
             60000]
    xx,yy,pars = fitGaussian(hi,guess,means=means,varis=varis)
    # f = gaussianMix(heights,widths,means)
    fit = getGaussian((xx,yy),*pars,means=means,varis=varis)
    h.contour(xc,yc,fit)
    
    
    fig = plt.figure()
    h = fig.add_subplot(projection='3d')
    h.plot_surface(xx,yy,hi[0].T)
    h.set_title('DATA')
    plt.show();
    plt.close();
    
    plt.contour(xc,yc,hi[0].T);
    plt.title('DATA')
    plt.show();
    plt.close();
    
    
    fig = plt.figure()
    h = fig.add_subplot(projection='3d')
    h.plot_surface(xx,yy,fit)
    h.set_title('FIT')
    plt.show();
    plt.close();
    
    plt.contour(xc,yc,fit);
    plt.title('FIT')
    plt.show();
    plt.close();
    
        