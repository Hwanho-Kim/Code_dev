"""
Plasma-Liquid Interaction Simulation Package

NOx analyzer with reaction contribution analysis for plasma-liquid chemistry.

Modules:
- config: Configuration constants and parameters
- utils: Logging, error handling, and utility functions
- chemistry_utils: Shared chemistry calculations
- preprocessor: Gas phase data preprocessing
- chemistry: Aqueous phase ODE chemistry system
- gui: Graphical user interface

Usage:
    # GUI mode
    python main.py

    # CLI mode
    python main.py --input data.csv --output results/

    # As a library
    from preprocessor import GasPhasePreprocessor
    from chemistry import CompleteAqueousChemistry
"""

__version__ = "10.0.0"
__author__ = "HKim"

from .config import (
    PHYSICAL,
    N2O4_EQ,
    WATER,
    HENRY_CONSTANTS,
    DEFAULTS,
    GAS_PHASE_SPECIES,
    AQUEOUS_SPECIES,
)

from .preprocessor import GasPhasePreprocessor, PreprocessParams
from .chemistry import CompleteAqueousChemistry

__all__ = [
    # Config
    'PHYSICAL',
    'N2O4_EQ',
    'WATER',
    'HENRY_CONSTANTS',
    'DEFAULTS',
    'GAS_PHASE_SPECIES',
    'AQUEOUS_SPECIES',
    # Classes
    'GasPhasePreprocessor',
    'PreprocessParams',
    'CompleteAqueousChemistry',
]
