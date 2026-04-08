"""
Configuration constants and parameters for Plasma-Liquid Interaction simulation.

This module contains all physical constants, default parameters, and configuration
values used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# Physical Constants
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants"""
    R: float = 8.314462618              # Gas constant [J/(mol·K)]
    KB_T_OVER_P: float = 1.3625946e-22  # kB/P in cm³ units at 1 atm
    AVOGADRO: float = 6.022e23          # Avogadro's number [1/mol]
    MOLECULES_TO_MOL_L: float = 1.66e-21  # Conversion factor: molecules/cm³ → mol/L


@dataclass(frozen=True)
class N2O4EquilibriumConstants:
    """N2O4 ↔ 2NO2 equilibrium constants"""
    KP_298: float = 6.75                # Equilibrium constant at 298.15K
    DELTA_H: float = -57120.0           # Enthalpy change [J/mol]
    REF_TEMP: float = 298.15            # Reference temperature [K]


@dataclass(frozen=True)
class WaterConstants:
    """Water equilibrium constants"""
    KW: float = 1e-14                   # Water ion product at 25°C


# =============================================================================
# Henry's Law Constants
# =============================================================================
# Two sets available:
#   1. Liu 2015 (dimensionless): H = C_aq / C_gas (both in mol/L)
#   2. Standard literature (M/atm): H = C_aq(M) / P(atm)
#
# Conversion: H_dimensionless = H_(M/atm) × RT = H_(M/atm) × 24.45 at 298K
#
# Ref: Sander 2023 ACP, NIST WebBook, Liu 2015 J. Phys. D
# =============================================================================

# Liu 2015 dimensionless constants (original)
HENRY_CONSTANTS_LIU2015: Dict[str, float] = {
    'O3': 0.2298,
    'NO': 0.046,
    'NO2': 0.978,
    'NO3': 44.0,
    'N2O4': 36.67,
    'N2O5': 51.34,
    'HONO': 1198.0,
    'HONO2': 5.1e6,
    'H2O2': 2.1e6,
}

# Standard literature values in M/atm (mol/(L·atm)) at 298K
# Sources: Sander 2023, NIST WebBook
# These appear ~24× smaller numerically but are physically equivalent
HENRY_CONSTANTS_M_ATM: Dict[str, float] = {
    'O3': 0.011,        # NIST: 0.011 mol/(kg·bar)
    'NO': 0.002,        # Low solubility
    'NO2': 0.02,        # NIST: 0.012-0.041 mol/(kg·bar)
    'NO3': 1.8,         # Estimated from Liu
    'N2O4': 1.5,        # Estimated from Liu
    'N2O5': 2.1,        # Estimated from Liu
    'HONO': 50.0,       # NIST: 50 mol/(kg·bar)
    'HONO2': 2.1e5,     # NIST: 2.1×10³ mol/(m³·Pa) = 2.1×10⁵ M/atm
    'H2O2': 8.0e4,      # NIST: ~8×10⁴ mol/(kg·bar)
}

# Active Henry constants - change this to switch between sets
# Options: 'liu2015' (dimensionless) or 'm_atm' (M/atm, appears smaller)
HENRY_CONSTANT_TYPE: str = 'liu2015'  # Options: 'liu2015', 'm_atm', 'disabled'

# Select active constants based on type
if HENRY_CONSTANT_TYPE == 'm_atm':
    HENRY_CONSTANTS = HENRY_CONSTANTS_M_ATM
else:
    HENRY_CONSTANTS = HENRY_CONSTANTS_LIU2015


# =============================================================================
# Mass Transfer Configuration (Two-Film Theory)
# Ref: Lee 2023 Chem. Eng. J. (Eq. 9, 10) & Liu 2015 J. Phys. D (Table 4)
# =============================================================================

@dataclass
class MassTransferConfig:
    """
    Two-film theory mass transfer parameters for 0D model.

    Based on Lee et al. (2023) Chemical Engineering Journal 458:141425
    and Liu et al. (2015) J. Phys. D: Appl. Phys. 48:495201

    For 0D model, we use volumetric mass transfer coefficient:
        k_L × a = (A/V) × Dadj / δxliq

    where A/V is the surface area to volume ratio of the liquid.

    Mass transfer rate: dC_aq/dt = k_L × a × (C_eq - C_aq)
    """
    # Boundary layer thicknesses [m]
    # Ref: Lee 2023 - δxgas: 1-10 mm, δxliq: 10-100 μm
    # Ref: Silsby 2021 - stagnant liquid δxliq ~ 100 μm, unstirred gas ~ 1 mm
    delta_x_gas: float = 0.001     # Gas film thickness [m] = 1 mm
    delta_x_liq: float = 0.0001    # Liquid film thickness [m] = 100 μm

    # Default diffusion coefficients [m²/s]
    # Ref: Liu 2015 Table 4, general literature
    D_gas_default: float = 1.5e-5   # Gas diffusivity in air (~10⁻⁵)
    D_liq_default: float = 1.5e-9   # Liquid diffusivity in water (~10⁻⁹)

    # Liquid geometry for 0D model
    # For petri dish: diameter ~35mm, depth ~10mm
    # A/V = πr² / (πr²h) = 1/h
    liquid_depth: float = 0.01      # Liquid depth [m] = 10 mm
    area_to_volume_ratio: float = 100.0  # A/V [m⁻¹] = 1/depth = 100 m⁻¹

    # Enable/disable mass transfer limitation
    enable_mass_transfer: bool = True

    # Note: Mass transfer now uses Lee 2023 flux-based approach (Eq. 9, 10)
    # ΔC = k_mt × (C_eq - C_liq) × Δt
    # No characteristic_time needed - flux calculated at each timestep


# Gas-phase diffusion coefficients [m²/s] at 300K
# Ref: General literature values
GAS_DIFFUSIVITY: Dict[str, float] = {
    'O3': 1.5e-5,
    'NO': 2.0e-5,
    'NO2': 1.5e-5,
    'NO3': 1.2e-5,
    'N2O4': 1.0e-5,
    'N2O5': 0.9e-5,
    'HONO': 1.3e-5,
    'HONO2': 1.2e-5,
    'H2O2': 1.3e-5,
    'OH': 2.0e-5,
}

# Aqueous-phase diffusion coefficients [m²/s] at 300K
# Ref: Liu 2015 J. Phys. D Table 4
LIQUID_DIFFUSIVITY: Dict[str, float] = {
    'O3': 1.75e-9,      # [55]
    'NO': 2.21e-9,      # [63]
    'NO2': 1.85e-9,     # [61]
    'NO3': 1.0e-9,      # [64]
    'N2O4': 1.0e-9,     # assumed
    'N2O5': 1.0e-9,     # assumed
    'HONO': 1.85e-9,    # same as HNO2 [61]
    'HONO2': 2.6e-9,    # [62]
    'H2O2': 1.0e-9,     # [57]
    'OH': 2.0e-9,       # [56]
    'HO2': 1.0e-9,      # assumed
}


# =============================================================================
# pKa Values for Acid-Base Equilibria
# =============================================================================

PKA_VALUES: Dict[str, float] = {
    'H2O': 13.999,
    'H2O2': 11.65,
    'OH': 11.9,
    'HO2': 4.8,
    'ONOOH': 6.6,
    'O2NOOH': 5.9,
    'HONO2': -1.34,
    'HONO': 3.4,
    # Chlorine species (for saline)
    # Ref: Wikipedia, PubChem
    'HClO': 7.5,    # Hypochlorous acid (7.40-7.54)
    'HClO2': 1.95,  # Chlorous acid (1.94-1.96)
}

# Acid-base pair mapping: total_name -> (acid_name, base_name, pKa)
ACID_BASE_PAIRS: Dict[str, tuple] = {
    'HONO_total': ('HONO', 'NO2-', 3.4),
    'HONO2_total': ('HONO2', 'NO3-', -1.34),
    'H2O2_total': ('H2O2', 'HO2-', 11.65),
    'HO2_total': ('HO2', 'O2-', 4.8),
    'ONOOH_total': ('ONOOH', 'ONOO-', 6.6),
    'O2NOOH_total': ('O2NOOH', 'O2NOO-', 5.9),
    # Chlorine species (for saline)
    'HClO_total': ('HClO', 'ClO-', 7.5),
    'HClO2_total': ('HClO2', 'ClO2-', 1.95),
}


# =============================================================================
# Species Lists
# =============================================================================

GAS_PHASE_SPECIES: List[str] = [
    'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2', 'H2O2'
]

AQUEOUS_SPECIES: List[str] = [
    # Neutral molecules (not acid-base)
    'H', 'H2', 'O', 'O2', 'O3', 'N2',
    'NO', 'NO2', 'NO3', 'N2O', 'N2O3', 'N2O4', 'N2O5',
    'OH', 'HO3',
    # Acid-base pairs as TOTAL variables (HA + A-)
    'HONO_total',      # HONO + NO2-
    'HONO2_total',     # HONO2 + NO3-
    'H2O2_total',      # H2O2 + HO2-
    'HO2_total',       # HO2 + O2-
    'ONOOH_total',     # ONOOH + ONOO-
    'O2NOOH_total',    # O2NOOH + O2NOO-
    # Ions (non-equilibrium)
    'H+', 'OH-', 'O-', 'O3-'
]

# Additional species for saline solution (Cl chemistry)
SALINE_SPECIES: List[str] = [
    # Chlorine neutral molecules
    'Cl', 'Cl2', 'HCl', 'Cl2O', 'Cl2O2', 'Cl2O3', 'Cl2O4', 'Cl2O5', 'Cl2O6',
    'ClO', 'ClO2', 'ClO3', 'ClNO2',
    # Chlorine acid-base pairs
    'HClO_total',      # HClO + ClO-
    'HClO2_total',     # HClO2 + ClO2-
    # Chlorine ions
    'Cl-', 'Cl2-', 'Cl3-', 'ClO3-', 'ClO4-',
    'HOCl-', 'HOClH',
]

TRACKED_SPECIES: List[str] = ['NO', 'NO2', 'NO2-', 'NO3-', 'OH', 'ONOO-', 'H2O2']

# =============================================================================
# Diagnostic Species for Reaction Contribution Analysis
# Based on Liu 2015 (DI Water) and Liu 2016 (Saline) papers
# =============================================================================

# DI Water key species (14 species)
# ROS: H2O2, HO2-, O3, OH, HO2, O2-
# RNS: NO2-, NO3-, HONO, HONO2, ONOO-, ONOOH, NO
# pH: H+
DIAGNOSTIC_SPECIES_BASE: List[str] = [
    # ROS (Reactive Oxygen Species)
    'H2O2', 'HO2-', 'O3', 'OH', 'HO2', 'O2-',
    # RNS (Reactive Nitrogen Species)
    'NO2-', 'NO3-', 'HONO', 'HONO2', 'ONOO-', 'ONOOH', 'NO',
    # pH indicator
    'H+',
]

# Additional Saline species (8 species)
# RCS: HClO, ClO-, Cl2, Cl-, Cl, ClO, Cl2-, ClNO2
DIAGNOSTIC_SPECIES_SALINE: List[str] = [
    # RCS (Reactive Chlorine Species)
    'HClO', 'ClO-', 'Cl2', 'Cl-', 'Cl', 'ClO', 'Cl2-', 'ClNO2',
]


# =============================================================================
# Default Parameters
# =============================================================================

@dataclass
class DefaultParameters:
    """Default simulation parameters"""
    temperature_C: float = 25.0
    temperature_K: float = 298.15
    humidity: float = 0.5               # Relative humidity (0-1)
    initial_pH: float = 7.0
    o3_h2o2_ratio: float = 5000.0
    smooth_window: int = 11
    smooth_polyorder: int = 3
    outlier_iqr_factor: float = 3.0
    trace_concentration: float = 1e-30  # Minimum concentration [mol/L] (lowered to avoid N2O5 artifacts)


@dataclass
class ODESolverConfig:
    """ODE solver configuration"""
    methods: List[tuple] = field(default_factory=lambda: [
        # Stiff solvers first (for saline/complex chemistry)
        ('BDF', 1e-4, 1e-10),
        ('Radau', 1e-4, 1e-10),
        # Non-stiff solvers as fallback
        ('RK45', 1e-3, 1e-9),
        ('RK23', 1e-3, 1e-9),
    ])
    max_time_step: float = 0.001        # Maximum time step [s]
    euler_dt: float = 0.0001            # Euler integration step [s]
    euler_max_steps: int = 100
    max_rate: float = 1e8               # Maximum allowed reaction rate
    max_concentration: float = 1.0      # Maximum concentration [mol/L]


@dataclass
class PreprocessConfig:
    """Preprocessing configuration"""
    remove_outliers: bool = True
    smooth_data: bool = True
    estimate_n2o4: bool = True
    estimate_hono: bool = True
    estimate_h2o2: bool = True


# =============================================================================
# GUI Configuration
# =============================================================================

@dataclass(frozen=True)
class GUIConfig:
    """GUI window configuration"""
    title: str = "NOx Analyzer with Reaction Contribution Analysis v10.0 (Refactored)"
    width: int = 1400
    height: int = 900
    version: str = "10.0"


# =============================================================================
# Singleton instances for easy access
# =============================================================================

PHYSICAL = PhysicalConstants()
N2O4_EQ = N2O4EquilibriumConstants()
WATER = WaterConstants()
DEFAULTS = DefaultParameters()
ODE_CONFIG = ODESolverConfig()
PREPROCESS_CONFIG = PreprocessConfig()
GUI_CONFIG = GUIConfig()
MASS_TRANSFER = MassTransferConfig()
