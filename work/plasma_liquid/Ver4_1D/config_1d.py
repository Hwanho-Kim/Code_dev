"""
Configuration for 1D Plasma-Liquid Diffusion-Reaction Model.

Based on Ver3 config.py with modifications for 1D spatial model:
- Removed area_to_volume_ratio (not needed in 1D)
- Added 1D grid parameters (liquid_depth, N_z)
- Mass transfer at interface via flux BC (not volumetric k_L·a)

References:
  Liu et al. (2015) J. Phys. D: Appl. Phys. 48, 495201
  Lee et al. (2023) Chem. Eng. J. 458, 141425
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
    FARADAY: float = 96485.33212        # Faraday constant [C/mol]
    EPSILON_0: float = 8.854187817e-12  # Vacuum permittivity [F/m]
    EPSILON_R_WATER: float = 78.4       # Relative permittivity of water at 298K
    KB: float = 1.380649e-23            # Boltzmann constant [J/K]
    E_CHARGE: float = 1.602176634e-19   # Elementary charge [C]


@dataclass(frozen=True)
class N2O4EquilibriumConstants:
    """N2O4 ↔ 2NO2 equilibrium constants"""
    KP_298: float = 6.75
    DELTA_H: float = -57120.0
    REF_TEMP: float = 298.15


@dataclass(frozen=True)
class WaterConstants:
    """Water equilibrium constants"""
    KW: float = 1e-14


# =============================================================================
# Henry's Law Constants (Liu 2015 dimensionless)
# =============================================================================

HENRY_CONSTANTS: Dict[str, float] = {
    'O3': 0.2298,
    'O': 0.03,            # O(3P) atom; very low solubility, reacts instantly
    'NO': 0.046,
    'NO2': 0.978,
    'NO3': 44.0,
    'N2O4': 36.67,
    'N2O5': 51.34,
    'HONO': 1198.0,
    'HONO2': 5.1e6,
    'H2O2': 2.1e6,
}


# =============================================================================
# Diffusion Coefficients
# =============================================================================

# Gas-phase [m²/s] at 300K
GAS_DIFFUSIVITY: Dict[str, float] = {
    'O': 2.5e-5,
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

# Aqueous-phase [m²/s] at 300K (Liu 2015 Table 4)
LIQUID_DIFFUSIVITY: Dict[str, float] = {
    'O3': 1.75e-9,
    'NO': 2.21e-9,
    'NO2': 1.85e-9,
    'NO3': 1.0e-9,
    'N2O4': 1.0e-9,
    'N2O5': 1.0e-9,
    'HONO': 1.85e-9,
    'HONO2': 2.6e-9,
    'H2O2': 1.0e-9,
    'OH': 2.0e-9,
    'HO2': 1.0e-9,
    'Cl-': 2.03e-9,       # textbook
    'TPA': 7.5e-10,       # Stokes-Einstein, disodium terephthalate dianion
    'hTPA': 7.0e-10,      # 2-hydroxyterephthalate
}

# Default diffusivities for species not in the tables above
D_GAS_DEFAULT: float = 1.5e-5   # [m²/s]
D_LIQ_DEFAULT: float = 1.5e-9   # [m²/s]


# =============================================================================
# 1D Mass Transfer Configuration
# =============================================================================

@dataclass
class MassTransfer1DConfig:
    """
    1D mass transfer parameters.

    In the 1D model, mass transfer at the gas-liquid interface is
    handled as a flux boundary condition:
        -D_i × ∂C_i/∂z|_{z=0} = k_L,i × (C_eq,i - C_i(0))

    where k_L,i = D_adj,i / δ_liq (Lee 2023 Eq. 9-10).

    No area_to_volume_ratio needed — the flux BC naturally
    couples the interface to the liquid interior.
    """
    # Boundary layer thicknesses [m]
    delta_x_gas: float = 0.01      # Gas film thickness [m]
    delta_x_liq: float = 0.0001    # Liquid film thickness [m]

    liquid_depth: float = 0.01     # [m] petri dish 10mm

    # Enable/disable mass transfer
    enable_mass_transfer: bool = True

    # Boundary condition type for gas-liquid interface
    #   'two_film'   : D_adj two-film model (Lee 2023)
    #   'dirichlet'  : C(0) ≈ C_eq via stiff relaxation (k_mt = 1.0 m/s)
    #   'film'       : k_mt = D_l / δ_liq (Heirman 2025 Eq.6, α_b=1)
    #   'film_alpha' : k_mt = α_b × D_l / δ_liq (Heirman 2025 Eq.7)
    #   'gas_alpha'  : gas + interface (Schwartz 1986, no liquid film)
    #   'three_film' : full Schwartz 1986 (gas + interface + liquid film)  ← project default 2026-04-23
    bc_type: str = 'three_film'
    alpha_b: float = 1.0           # mass accommodation coefficient (film_alpha only)

    # Species-specific α_b (overrides alpha_b when bc_type='film_alpha').
    # Keys must match GAS_TO_AQUEOUS_MAP gas-side names.
    # Sources: Kolb 2010, Davidovits 2006, IUPAC (Ammann 2013), Graves 2012,
    #          Heirman 2012, Bruggeman 2016.  See notes/alpha_b_literature.md.
    alpha_b_species: Dict[str, float] = field(default_factory=lambda: {
        'N2O5':  0.03,   # Kolb 2010
        'O3':    0.05,   # Davidovits 2006
        'H2O2':  0.1,    # IUPAC / Kolb 2010
        'NO':    0.001,  # Graves 2012 (poorly constrained)
        'NO2':   0.03,   # Bruggeman 2016
        'NO3':   0.03,   # assumed same as NO2
        'N2O4':  0.03,   # assumed same as NO2
        'HONO':  0.05,   # IUPAC / Davidovits 2006
        'HONO2': 0.07,   # Davidovits 2006 / JPL
    })


# =============================================================================
# 1D Grid Configuration
# =============================================================================

@dataclass
class Grid1DConfig:
    """Spatial grid configuration for 1D model.

    Two modes:
      Uniform:   N_z cells of equal width dz = L / N_z.
      Geometric: cells grow from dz_min at interface (z=0) by factor
                 stretch_ratio per cell. N_z is auto-computed from L.
                 Resolves Debye layer (~7 nm) near the interface while
                 keeping bulk cells coarse.
    """
    N_z: int = 100                    # Number of cells (uniform mode)
    dz_min: float = 2e-9             # Min cell width at interface [m]
    stretch_ratio: float = 1.15      # Geometric stretching ratio (>1)


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
}

# Acid-base pair mapping: total_name -> (acid_name, base_name, pKa)
ACID_BASE_PAIRS: Dict[str, tuple] = {
    'HONO_total': ('HONO', 'NO2-', 3.4),
    'HONO2_total': ('HONO2', 'NO3-', -1.34),
    'H2O2_total': ('H2O2', 'HO2-', 11.65),
    'HO2_total': ('HO2', 'O2-', 4.8),
    'ONOOH_total': ('ONOOH', 'ONOO-', 6.6),
    'O2NOOH_total': ('O2NOOH', 'O2NOO-', 5.9),
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

# Species that undergo gas-liquid mass transfer
# These species have Henry's law constants and will have flux BCs at z=0
TRANSFERABLE_SPECIES: List[str] = [
    'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5',
    'HONO_total',   # HONO dissolves → enters HONO_total pool
    'HONO2_total',  # HONO2 dissolves → enters HONO2_total pool
    'H2O2_total',   # H2O2 dissolves → enters H2O2_total pool
]

GAS_TO_AQUEOUS_MAP: Dict[str, str] = {
    'O3': 'O3',
    'NO': 'NO',
    'NO2': 'NO2',
    'NO3': 'NO3',
    'N2O4': 'N2O4',
    'N2O5': 'N2O5',
    'HONO': 'HONO_total',
    'HONO2': 'HONO2_total',
    'H2O2': 'H2O2_total',
}

# Charge number Z for directly tracked species (non-total).
# Total variables use effective Z computed from speciation at runtime.
SPECIES_CHARGE: Dict[str, int] = {
    'H+': +1,
    'OH-': -1,
    'O-': -1,
    'O3-': -1,
}

# TPA alkaline scenario — TPA²⁻ / hTPA²⁻ dianions. Na⁺ handled via fixed_cation_conc.
TPA_SPECIES: List[str] = ['TPA', 'hTPA']
TPA_SPECIES_CHARGE: Dict[str, int] = {
    'TPA': -2,
    'hTPA': -2,
}

SALINE_SPECIES: List[str] = [
    'Cl', 'Cl2', 'HCl', 'Cl2O', 'Cl2O2', 'Cl2O3', 'Cl2O4', 'Cl2O5', 'Cl2O6',
    'ClO', 'ClO2', 'ClO3', 'ClNO2',
    'HClO_total', 'HClO2_total',
    'Cl-', 'Cl2-', 'Cl3-', 'ClO3-', 'ClO4-',
    'HOCl-', 'HOClH',
]

SALINE_ACID_BASE_PAIRS: Dict[str, tuple] = {
    'HClO_total': ('HClO', 'ClO-', 7.5),
    'HClO2_total': ('HClO2', 'ClO2-', 1.95),
}

SALINE_SPECIES_CHARGE: Dict[str, int] = {
    'Cl-': -1,
    'Cl2-': -1,
    'Cl3-': -1,
    'ClO3-': -1,
    'ClO4-': -1,
    'HOCl-': -1,
}


# =============================================================================
# Poisson Configuration
# =============================================================================

@dataclass(frozen=True)
class PoissonConfig:
    enabled: bool = False
    epsilon_r: float = 78.4       # water at 298K


# =============================================================================
# Default Parameters
# =============================================================================

@dataclass
class DefaultParameters:
    """Default simulation parameters"""
    temperature_C: float = 25.0
    temperature_K: float = 298.15
    humidity: float = 0.5
    initial_pH: float = 7.0
    o3_h2o2_ratio: float = 5000.0
    smooth_window: int = 11
    smooth_polyorder: int = 3
    outlier_iqr_factor: float = 3.0
    trace_concentration: float = 1e-30


@dataclass
class ODESolverConfig:
    """ODE solver configuration for 1D PDE system."""
    method: str = 'BDF'          # Primary solver (stiff system)
    rtol: float = 1e-6           # Tight enough for trace species (O₃ ~nM, H₂O₂ ~0.1nM)
    atol: float = 1e-15          # Must be << min(trace radical conc); OH/HO2 ~1e-12 M
    max_step: float = 1.0        # Max time step [s] (for time-varying BCs)
    max_rate: float = 1e8        # Maximum reaction rate clamp
    max_concentration: float = 1.0  # Maximum concentration [mol/L]

    # Custom implicit solver (Gummel + backward Euler + Newton)
    dt_init: float = 1.0         # Initial timestep [s]
    dt_min: float = 1e-3         # Minimum timestep [s]
    dt_max: float = 10.0         # Maximum timestep [s]
    newton_maxiter: int = 8      # Max Newton iterations per timestep
    gummel_maxiter: int = 5      # Max Gummel (Poisson) iterations per timestep
    gummel_tol: float = 1e-4     # Gummel relative E-field change tolerance


@dataclass
class PreprocessConfig:
    """Preprocessing configuration"""
    remove_outliers: bool = True
    smooth_data: bool = True
    estimate_n2o4: bool = True
    estimate_hono: bool = True
    estimate_h2o2: bool = True


# =============================================================================
# Singleton instances
# =============================================================================

PHYSICAL = PhysicalConstants()
N2O4_EQ = N2O4EquilibriumConstants()
WATER = WaterConstants()
DEFAULTS = DefaultParameters()
ODE_CONFIG = ODESolverConfig()
PREPROCESS_CONFIG = PreprocessConfig()
MASS_TRANSFER = MassTransfer1DConfig()
GRID = Grid1DConfig()
POISSON = PoissonConfig()
