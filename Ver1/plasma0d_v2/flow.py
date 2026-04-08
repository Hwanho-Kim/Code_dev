"""CSTR flow model for concentration-based species.

dc_i/dt|_flow = (c_i,in - c_i) / tau

where tau = V / Q_actual is the residence time.
"""

import numpy as np
from .constants import R_GAS, T_STP, P_STP, NA


class FlowModel:
    """Flow model supporting CSTR and PFR.

    CSTR: dc_i/dt = R_i + (c_in_i - c_i) / τ      (perfect mixing)
    PFR:  dc_i/dt = R_i                             (no mixing, follow gas parcel for t=τ)
    """
    
    def __init__(self, V_reactor: float, Q_slm: float, P_gas: float,
                 T_gas_init: float, flow_model: str = 'CSTR'):
        self.V_reactor = V_reactor
        self.Q_slm = Q_slm
        self.P_gas = P_gas
        self.flow_model = flow_model.upper()
        
        self.x_inlet = np.array([])
        self.c_inlet = np.array([])
    
    def configure(self, species_manager, inlet_composition: dict):
        """Set inlet composition.
        
        Args:
            species_manager: SpeciesManager instance
            inlet_composition: dict of {species_name: mole_fraction}
        """
        n_sp = species_manager.n_species
        self.x_inlet = np.zeros(n_sp)
        self.c_inlet = np.zeros(n_sp)
        
        # Store mole fractions (T-independent) for dynamic c_inlet computation
        c_total_stp = P_STP / (R_GAS * T_STP)  # reference for display
        
        for name, x in inlet_composition.items():
            if species_manager.has(name):
                idx = species_manager.index(name)
                self.x_inlet[idx] = x
                self.c_inlet[idx] = x * c_total_stp  # reference only
        
        print(f"  Flow configured: Q={self.Q_slm} slm, V={self.V_reactor*1e6:.1f} cm³, "
              f"model={self.flow_model}")
        for name, x in inlet_composition.items():
            if species_manager.has(name):
                idx = species_manager.index(name)
                print(f"    {name}: x={x:.4f}, c_in(STP)={self.c_inlet[idx]:.2f} mol/m³")
    
    def get_residence_time(self, T_gas: float) -> float:
        """Residence time [s] used in ODE terms (energy flow loss, etc.).

        PFR returns 1e10 so that all τ-based loss terms vanish
        (we follow the gas parcel; there is no outlet loss).
        """
        if self.flow_model == 'PFR':
            return 1e10
        Q_actual = self.Q_slm * (T_gas / T_STP) * (P_STP / self.P_gas) / 60000.0
        if Q_actual > 0:
            return self.V_reactor / Q_actual
        return 1e10

    def get_physical_residence_time(self, T_gas: float) -> float:
        """Actual τ = V/Q regardless of flow model.  Use for PFR t_end."""
        Q_actual = self.Q_slm * (T_gas / T_STP) * (P_STP / self.P_gas) / 60000.0
        if Q_actual > 0:
            return self.V_reactor / Q_actual
        return 1e10

    def compute_flow_source(self, concentrations: np.ndarray, T_gas: float) -> np.ndarray:
        """Flow source terms [mol/(m³·s)].

        CSTR: (c_in(T) - c) / τ
        PFR:  0  (no mixing — following gas parcel)
        """
        if self.flow_model == 'PFR':
            return np.zeros_like(concentrations)
        tau = self.get_residence_time(T_gas)
        c_total_at_T = self.P_gas / (R_GAS * T_gas)
        c_inlet_at_T = self.x_inlet * c_total_at_T
        return (c_inlet_at_T - concentrations) / tau
