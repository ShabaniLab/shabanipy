import numpy as np
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

def lorentzian(f, fr, Ql, Qc, phi, A, A_phi):
    """Lorentzian function
    
    Source
    ----------
    "Efficient and robust analysis of complex scattering data under noise in
     microwave resonators"
    S. Probst, F. B. Song, P. A. Bushev, A. V. Ustinov, and M. Weides
    Rev. Sci. Instrum. 86, 024706 (2015)

    Parameters
    ----------
    f : np.ndarray
        Range of frequencies over which to evaluate the function

    fr : float
        Frequency that the cavity resonates at

    Ql : float
        Loaded quality factor

    Qc : float
        Absolute value of coupling (external) quality factor

    phi : float
        Impedance mismatch

    A : float
        Generalized Amplitude paramter

    Returns
    ----------
    lorentzian : np.ndarray
        1D array of generated lorentzian data for the given parameters 
    """
    cA = A*np.exp(1j*A_phi)
    return cA*(1 - np.exp(1j*phi)*(Ql/Qc)/(1+2j*Ql*(f/fr - 1)))

def loren_amp(*args):
    return convert_to_db(lorentzian(*args)) 
    
def quality_factor(f,P):
    """Extract the total quality factor from a transmission trace
    
    Parameters    
    ----------
    f : np.ndarray
        1D array of frequency points

    P : np.ndarray
        1D of Power in dB

    Returns
    ----------
    Q : float
        Total quality factor
    
    left : float
        3dB point used to the left of resonant frequency 

    right : float
        3dB point used to the right of resonant frequency 

    """

    # Take the minimum as the resonant frequency
    minP = min(P)
    idxc = np.where(P==minP)[0][0]
    P3db = minP+3
    
    # iteratively find right point
    i    = idxc
    while i < len(P)-1 and P[i] < P3db:
        i += 1
    right = i
    
    # iteratively find left point
    i = idxc
    while i > 0 and P[i] < P3db:
        i -= 1
    left = i
    return f[idxc]/np.abs(f[right]-f[left]),left,right

def convert_to_polar(complex_data):
    """Convert complex data to polar coordinates
    
    Parameters
    ----------
    complex_data : np.ndarray, complex
        1D array of complex data points
    
    Returns
    ----------
    r : np.ndarray
        1D array of real valued magnitudes
    
    phi : np.ndarray
        1D array of real valued phases in radians
    """
    r   = np.abs(complex_data)
    phi = np.angle(complex_data)
    return r,phi

def residual(params, freq, data, eps_data):
    res_freq = params['res_freq']
    q_loaded = params['q_loaded']
    q_coup   = params['q_coup']
    phi      = params['phi']
    amp      = params['amp']
    amp_phi  = params['amp']

    model = lorentzian(freq,res_freq,q_loaded,q_coup,phi,amp,amp_phi)
    delta = np.abs(data-model)
    resid = delta/eps_data
    return np.abs(resid)

def fit_resonator(freq,resonator_data,eps_data=1,method='leastsq',
                  initial=[None,10000,15000,0,1,0]):
    res_freq0,q_loaded0,q_coup0,phi0,amp0,amp_phi0 = initial
    if res_freq0 is None:
        res_freq0 = freq[len(freq)//2]
    params = Parameters()
    params.add('res_freq',value=res_freq0,min=min(freq),max=max(freq))
    params.add('q_loaded',value=q_loaded0,min=0)
    params.add('q_coup',value=q_coup0,min=0)
    params.add('phi',value=phi0)
    params.add('amp',value=amp0,min=0)
    params.add('amp_phi',value=amp_phi0)
        
    return minimize(residual,params,method=method,
                    args=(freq,resonator_data,eps_data)) 

def residual_amp(params, freq, data, eps_data):
    res_freq = params['res_freq']
    q_loaded = params['q_loaded']
    q_coup   = params['q_coup']
    phi      = params['phi']
    amp      = params['amp']
    amp_phi  = params['amp']

    model = loren_amp(freq,res_freq,q_loaded,q_coup,phi,amp,amp_phi)
    delta = np.abs(data-model)
    resid = delta/eps_data
    return np.abs(resid)

def fit_resonator_amp(freq,resonator_data,eps_data=1,method='leastsq',
                  initial=[None,10000,15000,0,1,0]):
    res_freq0,q_loaded0,q_coup0,phi0,amp0,amp_phi0 = initial
    if res_freq0 is None:
        res_freq0 = freq[len(freq)//2]
    params = Parameters()
    params.add('res_freq',value=res_freq0,min=min(freq),max=max(freq))
    params.add('q_loaded',value=q_loaded0,min=0)
    params.add('q_coup',value=q_coup0,min=0)
    params.add('phi',value=phi0)
    params.add('amp',value=amp0,min=0)
    params.add('amp_phi',value=amp_phi0)
        
    return minimize(residual_amp,params,method=method,
                    args=(freq,resonator_data,eps_data)) 

def radians_to_degrees(phase):
    return 180*phase/np.pi

def degrees_to_radians(phase):
    return np.pi*phase/180

def phase_in_degrees(data):
    return radians_to_degrees(np.angle(data))

def convert_to_cartesian(x):
    pass

def convert_to_db(x):
    return 20*np.log10(np.abs(x))

def linear_function(x,m,b):
    return m*x + b

def linear_fit(y):
    return curve_fit(linear_function,np.arange(len(y)),y)

def best_linear_fit(y):
    p,c = linear_fit(y)
    m,b = p
    return m,b

def fix_phase_delay(phase,plot_fit=False):
    """Fix overload error in the phase
    
    Parameters
    ----------
    phase : np.ndarry, complex
        1D array of phase data points bounded on -pi to pi

    Returns
    ----------
    corrected_phase : np.ndarray
        1D array of corrected phase data
    """
    cutoff = .9*np.pi   # Jump considered to be an overflow
    N      = len(phase) # Number of points
    L      = 1          # Number of points to measure jump over
    shift  = 2*np.pi    # Amount by which the shift should take place
    unwrapped_phase = np.array([i for i in phase])
    for n in range(N-L):
        diff = unwrapped_phase[n-L]-unwrapped_phase[n]
        sign = (1 if diff > 0 else -1) 
        if np.abs(diff) > cutoff:
            for m in range(n,N):
                unwrapped_phase[m] += sign*shift
    m,b = best_linear_fit(unwrapped_phase)
    corrected_phase = unwrapped_phase - (m*np.arange(N) + b)
    if plot_fit:
        plt.figure(200)
        plt.plot(phase,label='original')
        plt.plot(unwrapped_phase,label='unwrapped')
        plt.plot(corrected_phase,label='corrected')
        plt.plot(m*np.arange(N) + b,label='linear_fit')
        plt.plot([0,N],[-shift/2,-shift/2],'k--',label='bounds')
        plt.plot([0,N],[shift/2,shift/2],'k--')
        plt.legend()
    return corrected_phase

def fix_delay(data):
    Npow,N    = np.shape(data)
    mag,phase = convert_to_polar(data)
    for n in range(Npow):
        phase[n] = fix_phase_delay(phase[n])
    return mag*np.exp(1j*phase)

def detect_samples(freq):
    span    = freq[2]-freq[0]
    samples = []
    last    = 0
    for n in range(len(freq)-1):
        if freq[n+1]-freq[n] > span:
            samples.append((last,n+1))
            last = n+1
    samples.append((last,len(freq)))
    return samples

def load_VNA_data(fname):
    with h5py.File(fname) as f:
        power = f['Data']['Data'][:,0,0]
        freq  = f['Traces']['VNA - S21'][:, 2]
        real  = f['Traces']['VNA - S21'][:, 0]
        imag  = f['Traces']['VNA - S21'][:, 1]
    samples = detect_samples(freq[:,0])
    data    = real + 1j*imag
    freq    = freq.T
    data    = fix_delay(data.T)
    return power,freq,data,samples

def load_VNA_data2(fname):
    with h5py.File(fname) as f:
        center_freq = f['Data']['Data'][:,0]
        power       = f['Data']['Data'][:,1]
        span        = f['Data']['Data'][:,2]
        real  = f['Traces']['VNA - S21'][:, 0]
        imag  = f['Traces']['VNA - S21'][:, 1]
        freq  = f['Traces']['VNA - S21'][:, 2]
    center_freq = center_freq.T.flatten()
    span        = span.T.flatten()
    power       = power.T.flatten()
    samples = detect_samples(freq[:,0])
    data    = real + 1j*imag
    freq    = freq.T
    data    = fix_delay(data.T)
    return power,span,center_freq,freq,data,samples

def expand_data(complex_data):
    real   = np.real(complex_data)
    imag   = np.imag(complex_data)
    mag    = np.abs(complex_data)
    phase  = np.angle(complex_data)
    dB     = convert_to_db(complex_data)
    return real,imag,mag,phase,dB

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shabanipy.utils.labber_io import LabberData
    import h5py
    import sys

    fname = '/Users/javadshabani/Desktop/Projects/MicrowaveCavity/JS179H_JY001_010.hdf5'
    
    #sample_idx = 0
    #power_idx  = 0
    
    #power,freq,data,limits = load_VNA_data2(fname)
    
    scan_idx = 3
    power,span,cfreq,freq,data,limits = load_VNA_data2(fname)
    
    db = convert_to_db(data[scan_idx])

    f = freq[scan_idx]
    d = data[scan_idx]

    lib_fit = False
    if lib_fit:
        from resonator_tools.circuit import notch_port
        
        p1 = notch_port(f,d)
        lam = 1000e6 # smoothness
        p   = 0.98   # asymmetry
        
        fitted_baseline = p1.fit_baseline_amp(p1.z_data_raw,lam,p,niter=10)
        plt.plot(np.absolute(p1.z_data_raw))
        plt.plot(fitted_baseline)
        plt.show()
        
        p1 = notch_port(f,d)
        p1 = notch_port(p1.f_data,
                        p1.z_data_raw/fitted_baseline/0.99)
        p1.autofit()
        p1.plotall()
        print("single photon limit:",
              p1.get_single_photon_limit(diacorr=True), "dBm")
        print("photons in reso for input -140dBm:",
              p1.get_photons_in_resonator(-140,unit='dBm',diacorr=True),
              "photons")
        print("done")
        plt.show()
        sys.exit()

    init = (f[len(f)//2]+1.30,1603+2149,2149,-1.2, 1348/10000,0)
    init = (f[len(f)//2],10000,15000,0,1,0)

    method     = 'least_squares'
    iterations = 5
    for method in ['leastsq','nelder']:
        out = fit_resonator_amp(f,d,10,method,initial=init)
        
        out.params.pretty_print()
        res_freq = out.params['res_freq']
        q_loaded = out.params['q_loaded']
        q_coup   = out.params['q_coup']
        phi      = out.params['phi']
        amp      = out.params['amp']
        amp_phi  = out.params['amp_phi']
    
        res_freq0,q_loaded0,q_coup0,phi0,amp0,amp_phi0 = init
        fit_data = loren_amp(f,
                              res_freq,
                              q_loaded,
                              q_coup,
                              phi,
                              amp,
                              amp_phi)
        
        plt.plot(convert_to_db(d))
        plt.plot(fit_data)
        plt.show()
        sys.exit()

        real   = np.real(d)
        freal  = np.real(fit_data)
        
        imag   = np.imag(d)
        fimag  = np.imag(fit_data)

        mag    = np.abs(d)
        fmag   = np.abs(fit_data)
        
        phase  = np.angle(d)
        fphase = np.angle(fit_data)

        dB     = convert_to_db(d)
        fdB    = convert_to_db(fit_data)
        
        fig = plt.figure(figsize=(15,9))
        fig.suptitle(method)

        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        
        ax1.set_title('real')
        ax1.plot(f,real,label='data')
        ax1.plot(f,freal,label='fit')
        ax1.legend()

        ax2.set_title('imag')
        ax2.plot(f,imag,label='data')
        ax2.plot(f,fimag,label='fit')
        
        ax3.set_title('complex data')
        ax3.plot(real,imag,label='data')
        ax3.plot(freal,fimag,label='fit')
        
        ax4.set_title('mag')
        ax4.plot(f,mag,label='data')
        ax4.plot(f,fmag,label='fit')

        ax5.set_title('phase')
        ax5.plot(f,phase,label='data')
        ax5.plot(f,fphase,label='fit')

        ax6.set_title('dB')
        ax6.plot(f,dB,label='data')
        ax6.plot(f,fdB,label='fit')

        plt.show()

