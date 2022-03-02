import numpy as np
from lmfit.model import Model
from copy import deepcopy
import time 

from shabanipy.quantum_hall.wal.utils import recenter_wal_data, symmetrize_wal_data
from shabanipy.quantum_hall.wal.wal import *

from shabanipy.quantum_hall.wal.universal.magnetoconductivity import wal_magneto_conductance, wal_magneto_conductance_derivative, wal_magneto_conductance_partial_lphi
from shabanipy.quantum_hall.wal.universal.trace_computation import compute_trajectory_traces
from shabanipy.quantum_hall.wal.universal.trajectories.generation import get_detailed_trajectory_data, get_summary_trajectory_data
from shabanipy.quantum_hall.wal.utils import weight_wal_data

# load trajectories
index, l, c_phi, c_3phi, s_phi, s_3phi = get_detailed_trajectory_data()
lengths, surfaces, cosj = get_summary_trajectory_data()

def bound_data(field_raw, resistance_raw, bound=[-0.1, 0.1]):
    field = deepcopy(field_raw)
    resistance = deepcopy(resistance_raw)
    # mask out low field values 
    lower_bound = bound[0]
    mask = np.where(np.greater(field, lower_bound))
    field = field[mask]
    resistance = resistance[mask]
    # mask out high field values 
    upper_bound = bound[1]
    mask = np.where(np.less(field, upper_bound))
    field = field[mask]
    resistance = resistance[mask]
    return field, resistance


def preprocess(field_raw, 
    resistance_raw, bound=[-0.1, 0.1], 
    increasing_sequence=False, 
    only_even_part=False, 
    idx=None):
    field = deepcopy(field_raw)
    resistance = deepcopy(resistance_raw)
    
    if idx is None: 
        field, resistance = recenter_wal_data(field, resistance) 
    else: 
        field = field - field[idx]

    # symmetric bound
    field, resistance = bound_data(field, resistance)

    if only_even_part:
        resistance = (resistance + resistance[::-1])/2
    
    field, resistance = bound_data(field, resistance, bound=bound)

    if field[0] > field[-1]:
        field = field[::-1]
        resistance = resistance[::-1]

    if increasing_sequence:
        field, idx = np.unique(field, return_index=True)
        resistance = resistance[idx]
    
    return field, resistance

def magnetoconductivity_semiclassical(field,
    l_phi, theta_alpha, theta_beta1, theta_beta3, B_magnitude, B_angle):
    traces = compute_trajectory_traces(index=index, 
        l=l, 
        c_phi=c_phi, 
        s_phi=s_phi,  
        theta_alpha=theta_alpha, 
        theta_beta1=theta_beta1, 
        theta_beta3=theta_beta3, 
        B_magnitude=B_magnitude, 
        B_angle=B_angle)
    sigma = wal_magneto_conductance(fields=field, l_phi=l_phi, traces=traces, surfaces=surfaces, lengths=lengths, cosjs=cosj)
    return sigma 

def fit_semiclassical(field,
    conductance_data, 
    phase_relaxation_length_guess=100, 
    cubic_dresselhaus_guess=0.0, 
    linear_dresselhaus_guess=0.0, 
    rashba_guess=np.pi/4):

    print(r"Calculating best fit at zero in-plane field...")
    start = time.time()
    model = Model(magnetoconductivity_semiclassical)
    
    # set parameter hints 
    model.set_param_hint('l_phi', min=10, max=1000, value=phase_relaxation_length_guess, vary=True)
    model.set_param_hint('theta_alpha', min=-2*np.pi, max=2*np.pi, value=rashba_guess, vary=True)
    model.set_param_hint('theta_beta1', min=0, max=2*np.pi, value=linear_dresselhaus_guess, vary=False)
    model.set_param_hint('theta_beta3', value=cubic_dresselhaus_guess, vary=False)
    model.set_param_hint('B_magnitude', value=0.0, vary=False)
    model.set_param_hint('B_angle', value=0.0, vary=False)

    params = model.make_params()
    
    # perform the fit
    res = model.fit(conductance_data, params, field=field, method='nelder')
    print(res.fit_report())
    res = model.fit(conductance_data, res.params, field=field)
    print(res.fit_report())
    
    phi = res.best_values['l_phi']
    phi_std = deepcopy(res.params['l_phi'].stderr)
    alpha = res.best_values['theta_alpha']
    alpha_std = deepcopy(res.params['theta_alpha'].stderr)
    beta = res.best_values['theta_beta1']
    beta_std = res.params['theta_beta1'].stderr
    gamma = res.best_values['theta_beta3']
    gamma_std = deepcopy(res.params['theta_beta3'].stderr)

    print(r"done in {0:.1f} seconds.".format(time.time() - start))

    return phi, phi_std, alpha, alpha_std, beta, beta_std, gamma, gamma_std, res.best_fit

def fit_semiclassical_inplane(field,
    conductance_data, 
    phase_relaxation_length_fixed=680, 
    cubic_dresselhaus_fixed=0.0, 
    linear_dresselhaus_fixed=0.03, 
    rashba_fixed=1.17,
    B_magnitude = 0.0,
    B_angle = 0.0):

    print(r"Calculating best fit in the presence of in-plane field...")
    start = time.time()
    model = Model(magnetoconductivity_semiclassical)
    
    # set parameter hints 
    model.set_param_hint('l_phi', value=phase_relaxation_length_fixed, vary=False)
    model.set_param_hint('theta_alpha', value=rashba_fixed, vary=False)
    model.set_param_hint('theta_beta1', value=linear_dresselhaus_fixed, vary=False)
    model.set_param_hint('theta_beta3', value=cubic_dresselhaus_fixed, vary=False)
    model.set_param_hint('B_magnitude', min=0.0, max=0.2, value=B_magnitude, vary=True)
    model.set_param_hint('B_angle', value=B_angle, vary=False)

    params = model.make_params()
    
    # perform the fit
    res = model.fit(conductance_data, params, field=field, method='nelder')
    print(res.fit_report())
    res = model.fit(conductance_data, res.params, field=field)
    print(res.fit_report())

    r = res.best_values['B_magnitude']
    theta = res.best_values['B_angle']
    r_stderr = res.params['B_magnitude'].stderr
    theta_stderr = res.params['B_angle'].stderr

    print(r"done in {0:.1f} seconds.".format(time.time() - start))

    return r, theta, r_stderr, theta_stderr, res.best_fit
    