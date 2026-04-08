"""Physical constants and unit conversions (SI + COMSOL chemical engineering)."""

# Fundamental constants
KB = 1.380649e-23       # Boltzmann constant [J/K]
QE = 1.602176634e-19    # Elementary charge [C]
ME = 9.1093837015e-31   # Electron mass [kg]
NA = 6.02214076e23      # Avogadro number [1/mol]
R_GAS = 8.314462618     # Ideal gas constant [J/(mol·K)]
EPS0 = 8.854187817e-12  # Vacuum permittivity [F/m]
PI = 3.141592653589793

# Unit conversions
TD_TO_VM2 = 1.0e-21          # Townsend to V·m²
EV_TO_J = QE                 # 1 eV = 1.602e-19 J
EV_TO_JMOL = 96485.0         # 1 eV = 96485 J/mol
JMOL_TO_EV = 1.0 / EV_TO_JMOL

# STP conditions (for flow conversion)
T_STP = 273.15    # [K]
P_STP = 101325.0  # [Pa]

def slm_to_m3s(Q_slm, T_gas, P_gas):
    """Convert standard liters per minute to actual m³/s at (T_gas, P_gas).
    
    Q_actual = Q_slm * (T/T_STP) * (P_STP/P) / 60000
    """
    return Q_slm * (T_gas / T_STP) * (P_STP / P_gas) / 60000.0

def number_density_to_concentration(n):
    """Convert number density [1/m³] to concentration [mol/m³]."""
    return n / NA

def concentration_to_number_density(c):
    """Convert concentration [mol/m³] to number density [1/m³]."""
    return c * NA

def total_concentration(P, T):
    """Total gas concentration [mol/m³] from ideal gas law: c = P / (R*T)."""
    return P / (R_GAS * T)

def total_number_density(P, T):
    """Total gas number density [1/m³] from ideal gas law: N = P / (kB*T)."""
    return P / (KB * T)
