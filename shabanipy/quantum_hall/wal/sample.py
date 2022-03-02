from scipy import constants
from scipy.interpolate import CubicSpline

from shabanipy.quantum_hall.conversion import \
    (htr_from_mobility_density,
     mean_free_time_from_mobility,
     kf_from_density)
from shabanipy.quantum_hall.wal.universal.utils import \
    (linear_theta_to_linear_soi, 
    cubic_theta_to_cubic_soi)


class Sample(object):

    def __init__(self,
                 name,
                 probe_current,
                 geometric_factor,
                 effective_mass,
                 gate,
                 density,
                 mobility,
                 ):
        self.name = name
        self.probe_current = probe_current
        self.geometric_factor = geometric_factor
        self.effective_mass = effective_mass
        self.mass = effective_mass * constants.m_e
        self.gate = gate
        self.density = density
        self.mobility = mobility
        self.transport_field, self.transport_time, self.fermi_momentum, self.mean_free_path = \
            self._calculate_transport(density, mobility)
        self.phase_relaxation_theta = None
        self.rashba_theta = None
        self.linear_dresselhaus_theta = None
        self.cubic_dresselhaus_theta = None

    def _calculate_transport(self, density, mobility):
        transport_field = htr_from_mobility_density(mobility, density, self.mass)
        transport_time = mean_free_time_from_mobility(mobility, self.mass)
        fermi_momentum = kf_from_density(density)
        mean_free_path = constants.hbar * fermi_momentum / self.mass * transport_time
        return transport_field, transport_time, fermi_momentum, mean_free_path

    def get_transport(self):
        htr, tau, kf, l = self._calculate_transport(self.density, self.mobility)
        return self.density, self.mobility, htr, tau, kf, l 
    
    def get_transport_interpolated(self, gate):
        density_func = CubicSpline(self.gate, self.density)
        mobility_func = CubicSpline(self.gate, self.mobility)
        transport_field, transport_time, fermi_momentum, mean_free_path = \
            self._calculate_transport(density_func(gate), mobility_func(gate))
        return density_func(gate), mobility_func(gate), transport_field, transport_time, fermi_momentum, mean_free_path

    def set_spin_orbit(self, phase_relaxation_theta, rashba_theta, linear_dresselhaus_theta, cubic_dresselhaus_theta):
        self.phase_relaxation_theta = phase_relaxation_theta
        self.rashba_theta = rashba_theta
        self.linear_dresselhaus_theta = linear_dresselhaus_theta
        self.cubic_dresselhaus_theta = cubic_dresselhaus_theta

    def get_spin_orbit(self, units='theta', interpolate=False, gate=None):
        phi, alpha, beta, gamma = self.phase_relaxation_theta, self.rashba_theta, self.linear_dresselhaus_theta, self.cubic_dresselhaus_theta
        if gate is not None and interpolate:
            phi_func = CubicSpline(self.gate, phi)
            alpha_func = CubicSpline(self.gate, alpha)
            beta_func = CubicSpline(self.gate, beta)
            gamma_func = CubicSpline(self.gate, gamma)
            phi, alpha, beta, gamma = phi_func(gate), alpha_func(gate), beta_func(gate), gamma_func(gate)
        if units != 'theta':
            if interpolate and gate is not None:
                _, _, _, transport_time, fermi_momentum, mean_free_path = self.get_transport_interpolated(gate)
            else:
                transport_time, fermi_momentum, mean_free_path = self.transport_time, self.fermi_momentum, self.mean_free_path
            phi, alpha, beta, gamma = self.convert_spin_orbit_to_ev(phi, alpha, beta, gamma, transport_time, fermi_momentum, mean_free_path)
        return phi, alpha, beta, gamma  

    def convert_spin_orbit_to_ev(self, phi, alpha, beta, gamma, transport_time, fermi_momentum, mean_free_path):
        phi = phi * mean_free_path / 1.0e-6 # um
        alpha = linear_theta_to_linear_soi(alpha, fermi_momentum, transport_time) / constants.e * 1e10 * 1000 # meV.A
        beta = linear_theta_to_linear_soi(beta, fermi_momentum, transport_time) / constants.e * 1e10 * 1000 # meV.A
        gamma = cubic_theta_to_cubic_soi(gamma, fermi_momentum, transport_time) / constants.e * 1e30 # eV.A^3
        return phi, alpha, beta, gamma
