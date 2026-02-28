import numpy as np
from scipy import integrate, constants
from dataclasses import dataclass, field
from typing import List, Dict

HBAR = constants.hbar
E_CHARGE = constants.e
EPS0 = constants.epsilon_0
AMU = constants.atomic_mass
C_LIGHT = constants.c
K_B = constants.k

@dataclass
class WISEParameters:
    ion_height: float = 40e-6
    num_zones: int = 1026
    qubits_per_junction: int = 6
    
    dynamic_electrodes: int = 11970
    shim_electrodes: int = 10260
    
    ion_mass: float = 40 * AMU
    ion_charge: float = E_CHARGE
    
    axial_freq_hz: float = 1.0e7
    radial_freq_hz: float = 1.5e6
    
    noise_beta: float = 1.5
    noise_low_cutoff_hz: float = 300.0
    noise_amplitude_S0: float = None
    
    heating_rate_ref: float = 1000.0
    heating_rate_ref_freq: float = 880e3
    heating_rate_ref_height: float = 50e-6
    
    shim_capacitance: float = 30e-12
    shim_hold_time: float = 180.0
    shim_voltage: float = 10.0
    shim_geometric_factor: float = 0.1
    
    dac_1f_noise_Av: float = 1e-12
    dac_max_freq: float = 1e6
    
    switch_charge_injection: float = 10e-15
    electrode_capacitance: float = 100e-15
    parasitic_capacitance: float = 10e-15
    switch_geometric_factor: float = 0.3
    
    cooling_rate: float = 10e3
    
    rf_voltage: float = 100.0
    rf_frequency: float = 50e6
    shim_to_rf_capacitance: float = 1e-15
    
    electrode_density: float = 1e10
    typical_electrode_area: float = 100e-12
    
    def __post_init__(self):
        if self.noise_amplitude_S0 is None:
            self.noise_amplitude_S0 = self._calculate_S0_from_heating()
    
    def _calculate_S0_from_heating(self):
        omega_ref = 2 * np.pi * self.heating_rate_ref_freq
        omega_ir = 2 * np.pi * self.noise_low_cutoff_hz
        
        height_scaling = (self.heating_rate_ref_height / self.ion_height) ** 4
        heating_rate_scaled = self.heating_rate_ref * height_scaling
        
        S_E_at_ref = (4 * self.ion_mass * HBAR * omega_ref / self.ion_charge**2) * heating_rate_scaled
        
        S0 = S_E_at_ref / ((omega_ir / omega_ref) ** self.noise_beta)
        
        return S0

@dataclass
class IonChain:
    num_ions: int
    trap_frequency_hz: float
    ion_mass: float
    ion_charge: float
    
    positions: np.ndarray = field(init=False)
    mode_frequencies: np.ndarray = field(init=False)
    mode_shapes: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self._calculate_equilibrium_positions()
        self._calculate_normal_modes()
    
    def _calculate_equilibrium_positions(self):
        from scipy.optimize import fsolve
        
        N = self.num_ions
        omegaz = 2 * np.pi * self.trap_frequency_hz
        
        l = (self.ion_charge**2 / (4 * np.pi * EPS0 * self.ion_mass * omegaz**2))**(1/3)
        
        def force_balance(positions):
            forces = np.zeros(N)
            forces += -self.ion_mass * omegaz**2 * positions
            
            for i in range(N):
                for j in range(N):
                    if i != j:
                        distance = abs(positions[i] - positions[j])
                        sign = 1 if positions[j] > positions[i] else -1
                        forces[i] += sign * self.ion_charge**2 / (4 * np.pi * EPS0 * distance**2)
            return forces

        z_guess = np.linspace(-(N-1)*l/2, (N-1)*l/2, N)
        z_eq = fsolve(force_balance, z_guess)
        self.positions = z_eq
    
    def _calculate_normal_modes(self):
        from scipy.linalg import eigh
        
        N = self.num_ions
        omegaz = 2 * np.pi * self.trap_frequency_hz
        
        H = np.zeros((N, N))
        for i in range(N):
            H[i, i] = omegaz**2
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    distance = abs(self.positions[i] - self.positions[j])
                    coupling = -2 * self.ion_charge**2 / (4 * np.pi * EPS0 * self.ion_mass * distance**3)
                    H[i, j] += coupling
                    H[i, i] -= coupling
        
        eigenvalues, eigenvectors = eigh(H)
        idx = np.argsort(eigenvalues)
        self.mode_frequencies = np.sqrt(np.abs(eigenvalues[idx])) / (2 * np.pi)
        self.mode_shapes = eigenvectors[:, idx].T
        
        for m in range(N):
            norm = np.sqrt(np.sum(self.mode_shapes[m]**2))
            self.mode_shapes[m] /= norm

@dataclass
class ElectrodeConfig:
    coupling_matrix: np.ndarray
    dac_groups: List[List[int]]
    switch_settings: np.ndarray
    
    def get_coupling_for_mode(self, mode_shape: np.ndarray) -> np.ndarray:
        return np.sum(self.coupling_matrix * mode_shape[np.newaxis, :], axis=1)

class WISENoiseModel:
    
    def __init__(self, wise_params: WISEParameters):
        self.p = wise_params
    
    def electric_field_noise_spectrum(self, frequency_hz: float) -> float:
        omega = 2 * np.pi * frequency_hz
        omega_ir = 2 * np.pi * self.p.noise_low_cutoff_hz
        
        if omega < omega_ir:
            return self.p.noise_amplitude_S0
        else:
            return self.p.noise_amplitude_S0 * (omega_ir / omega) ** self.p.noise_beta
    
    def electrode_voltage_noise(self, frequency_hz: float, electrode_area: float = None) -> float:
        if electrode_area is None:
            electrode_area = self.p.typical_electrode_area
        
        S_E = self.electric_field_noise_spectrum(frequency_hz)
        A_char = np.pi * self.p.ion_height**2
        S_V = S_E * self.p.ion_height**2 * (electrode_area / A_char)
        
        return S_V
    
    def heating_rate(self, chain: IonChain, mode_idx: int, 
                     electrode_config: ElectrodeConfig) -> float:
        omega_m = 2 * np.pi * chain.mode_frequencies[mode_idx]
        mode_shape = chain.mode_shapes[mode_idx]
        
        E_coupling = electrode_config.get_coupling_for_mode(mode_shape)

        heating_sum = 0
        
        for group in electrode_config.dac_groups:
            if len(group) > 0:
                total_coupling = np.sum(E_coupling[group])
                S_V_avg = np.mean([self.electrode_voltage_noise(omega_m/(2*np.pi)) 
                                   for _ in group])
                heating_sum += total_coupling**2 * S_V_avg
        
        all_electrodes = set(range(len(electrode_config.coupling_matrix)))
        grouped_electrodes = set([e for group in electrode_config.dac_groups for e in group])
        independent_electrodes = all_electrodes - grouped_electrodes
        
        for e in independent_electrodes:
            S_V = self.electrode_voltage_noise(omega_m/(2*np.pi))
            heating_sum += E_coupling[e]**2 * S_V
        
        prefactor = self.p.ion_charge**2 / (4 * self.p.ion_mass * HBAR * omega_m)
        
        return prefactor * heating_sum
    
    def frequency_noise_spectrum(self, chain: IonChain, mode_idx: int,
                                 electrode_config: ElectrodeConfig,
                                 frequency_hz: float) -> float:
        omega_m = 2 * np.pi * chain.mode_frequencies[mode_idx]
        mode_shape = chain.mode_shapes[mode_idx]
        
        E_coupling = electrode_config.get_coupling_for_mode(mode_shape)
        
        freq_coupling = self.p.ion_charge / (self.p.ion_mass * omega_m)
        
        noise_sum = 0
        
        for group in electrode_config.dac_groups:
            if len(group) > 0:
                total_coupling = np.sum(E_coupling[group])
                S_V_avg = np.mean([self.electrode_voltage_noise(frequency_hz) 
                                   for _ in group])
                noise_sum += total_coupling**2 * S_V_avg
        
        all_electrodes = set(range(len(electrode_config.coupling_matrix)))
        grouped_electrodes = set([e for group in electrode_config.dac_groups for e in group])
        independent_electrodes = all_electrodes - grouped_electrodes
        
        for e in independent_electrodes:
            S_V = self.electrode_voltage_noise(frequency_hz)
            noise_sum += E_coupling[e]**2 * S_V
        
        return freq_coupling**2 * noise_sum
    
    def dephasing_variance(self, chain: IonChain, mode_idx: int,
                           electrode_config: ElectrodeConfig,
                           idle_time: float) -> float:
        def integrand(f):
            omega = 2 * np.pi * f
            
            if omega == 0:
                F = idle_time**2 / 2
            else:
                F = (np.sin(omega * idle_time / 2) / (omega / 2))**2
            
            S_dw = self.frequency_noise_spectrum(chain, mode_idx, electrode_config, f)
            
            return S_dw * F
        
        f_min = 1e-3
        f_max = 1e6
        
        f_points = np.logspace(np.log10(f_min), np.log10(f_max), 2000)
        integrand_values = np.array([integrand(f) for f in f_points])
        
        phase_variance = integrate.trapezoid(integrand_values, f_points)
        
        return phase_variance
    
    def dephasing_error_probability(self, chain: IonChain, mode_idx: int,
                                    electrode_config: ElectrodeConfig,
                                    idle_time: float) -> float:
        phase_var = self.dephasing_variance(chain, mode_idx, electrode_config, idle_time)
        
        return 0.5 * (1 - np.exp(-phase_var / 2))
    
    def sample_and_hold_drift_error(self, gate_time: float = 100e-6) -> float:
        deltaV = self.p.shim_voltage * gate_time / self.p.shim_hold_time

        dE_dV = 1 / self.p.ion_height
        domega_dE = 1e3 * 2 * np.pi
        domega_dV = domega_dE * dE_dV
        
        deltaomega = domega_dV * deltaV
        
        error = (deltaomega * gate_time)**2 / 4
        
        return error
    
    def transport_heating(self, shuttle_time: float = 100e-6,
                         with_cooling: bool = True) -> float:
        if with_cooling:
            delta_n = 0.01
        else:
            omega = 2 * np.pi * self.p.axial_freq_hz
            dz0_dV = 10e-9
            S_V = 1e-12
            
            bandwidth = 1 / shuttle_time
            integral = (bandwidth**3) / 3
            
            prefactor = self.p.ion_mass / (2 * HBAR * omega) * dz0_dV**2
            heating_rate = prefactor * (2 * np.pi)**2 * S_V * integral
            
            delta_n = heating_rate * shuttle_time
        
        return delta_n
    
    def rf_pickup_error(self, gate_time: float = 100e-6) -> float:
        C_total = self.p.shim_capacitance + self.p.shim_to_rf_capacitance
        V_s_RF = self.p.rf_voltage * self.p.shim_to_rf_capacitance / C_total
        
        omega_z = 2 * np.pi * self.p.axial_freq_hz
        Omega_RF = 2 * np.pi * self.p.rf_frequency
        
        numerator = self.p.ion_charge * self.p.shim_geometric_factor * V_s_RF / self.p.ion_height
        denominator = self.p.ion_mass * (Omega_RF**2 - omega_z**2)
        z_mm = numerator / denominator
        
        v_mm = z_mm * Omega_RF
    
        deltaomega_omega = v_mm / C_LIGHT
        
        error = (deltaomega_omega * omega_z * gate_time)**2 * self.p.shim_electrodes
        
        return error
    
    def dac_1f_noise_error(self, chain: IonChain, mode_idx: int,
                           electrode_config: ElectrodeConfig,
                           gate_time: float = 100e-6) -> float:
        omega_m = 2 * np.pi * chain.mode_frequencies[mode_idx]
        mode_shape = chain.mode_shapes[mode_idx]
        
        E_coupling = electrode_config.get_coupling_for_mode(mode_shape)
        
        S_V = self.p.dac_1f_noise_Av / gate_time
        
        noise_sum = 0
        for group in electrode_config.dac_groups:
            if len(group) > 0:
                total_coupling = np.sum(E_coupling[group])
                noise_sum += total_coupling**2 * S_V
        
        all_electrodes = set(range(len(electrode_config.coupling_matrix)))
        grouped_electrodes = set([e for group in electrode_config.dac_groups for e in group])
        independent_electrodes = all_electrodes - grouped_electrodes
        
        for e in independent_electrodes:
            noise_sum += E_coupling[e]**2 * S_V
        
        freq_coupling = self.p.ion_charge / (self.p.ion_mass * omega_m)
        phase_var = freq_coupling**2 * noise_sum * gate_time**2
        
        error = 0.5 * (1 - np.exp(-phase_var / 2)) * self.p.dynamic_electrodes
        
        return error
    
    def ms_gate_infidelity(self, chain: IonChain, mode_idx: int,
                          electrode_config: ElectrodeConfig,
                          gate_time: float = 100e-6,
                          initial_phonons: float = 0.01,
                          include_all_noise: bool = True) -> Dict:
        
        Gamma = self.heating_rate(chain, mode_idx, electrode_config)
        n_avg = initial_phonons + Gamma * gate_time / 2
        eta = 0.1
        alpha_sq = eta**2 / 4
        heating_error = 4 * (2 * n_avg + 1) * alpha_sq
        
        phase_var = self.dephasing_variance(chain, mode_idx, electrode_config, gate_time)
        dephasing_error = 0.5 * phase_var
        
        total_error = heating_error + dephasing_error
        
        results = {
            'heating_rate_Hz': Gamma,
            'n_avg': n_avg,
            'heating_error': heating_error,
            'dephasing_error': dephasing_error,
        }
        
        if include_all_noise:
            drift_error = self.sample_and_hold_drift_error(gate_time)
            
            transport_error_cooling = 0.1**2 * self.transport_heating(gate_time, with_cooling=True)
            
            rf_error = self.rf_pickup_error(gate_time)
            
            dac_error = self.dac_1f_noise_error(chain, mode_idx, electrode_config, gate_time)
        
            results.update({
                'drift_error': drift_error,
                'transport_error': transport_error_cooling,
                'rf_pickup_error': rf_error,
                'dac_1f_noise_error': dac_error,
            })
            total_error += drift_error + transport_error_cooling + rf_error + dac_error
        
        results['total_error'] = total_error
        
        return results

def generate_realistic_electrode_config(chain: IonChain, p: WISEParameters) -> ElectrodeConfig:
    n_ions = chain.num_ions
    n_electrodes = 20
    
    coupling_matrix = np.zeros((n_electrodes, n_ions))
    
    base_coupling = 1 / p.ion_height
    
    for e in range(n_electrodes):
        for i in range(n_ions):
            electrode_pos = (e - n_electrodes/2) * 50e-6
            ion_pos = chain.positions[i]
            distance = abs(electrode_pos - ion_pos)
            
            coupling_matrix[e, i] = base_coupling * np.exp(-distance / p.ion_height)
    
    dac_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
    
    switch_settings = np.ones(n_electrodes)
    
    return ElectrodeConfig(coupling_matrix, dac_groups, switch_settings)

def calculate_gate_infidelity(num_ions: int = 2, mode_idx: int = 0, 
                            background_heating: float = 1000.0,
                            trap_frequency_hz: float = 1e6,
                            ion_height: float = 40e-6,
                            gate_time: float = 100e-6) -> float:
    
    wise_params = WISEParameters(
        ion_height=ion_height,
        heating_rate_ref=background_heating,
        noise_beta=1.5,
        noise_low_cutoff_hz=300.0
    )
    
    model = WISENoiseModel(wise_params)
    
    chain = IonChain(
        num_ions=num_ions,
        trap_frequency_hz=trap_frequency_hz,
        ion_mass=wise_params.ion_mass,
        ion_charge=wise_params.ion_charge
    )
    
    electrode_config = generate_realistic_electrode_config(chain, wise_params)
    
    gate_errors = model.ms_gate_infidelity(
        chain=chain,
        mode_idx=mode_idx,
        electrode_config=electrode_config,
        gate_time=gate_time,
        initial_phonons=0.01,
        include_all_noise=True
    )
    
    return gate_errors['total_error']

if __name__ == "__main__":
    
    infidelity = calculate_gate_infidelity(
        num_ions=200,
        background_heating=10000,
        trap_frequency_hz=1e7,
    )
    
    print(f"{infidelity:.4e}")