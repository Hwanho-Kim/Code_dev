"""Species management for concentration-based (mol/m³) plasma chemistry."""

import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class SpeciesType(Enum):
    ELECTRON = "electron"
    NEUTRAL = "neutral"
    RADICAL = "radical"
    EXCITED = "excited"
    ION_POSITIVE = "ion_positive"
    ION_NEGATIVE = "ion_negative"


@dataclass
class Species:
    name: str
    type: SpeciesType
    mass: float          # molar mass [kg/mol]
    delta_hf_kj: float = 0.0  # ΔHf° [kJ/mol] at 298K
    index: int = -1      # index in state vector (assigned at finalization)
    
    @property
    def is_electron(self):
        return self.type == SpeciesType.ELECTRON


class SpeciesManager:
    """Manages species list and maps names to indices.
    
    State vector order: [c_e, c_1, c_2, ..., c_Ns, n_e*eps_mean, T_gas]
    where c_i are concentrations in mol/m³.
    Electron is always index 0.
    """
    
    def __init__(self):
        self._species: Dict[str, Species] = {}
        self._finalized = False
        self._index_map: Dict[str, int] = {}
        self._species_list: List[Species] = []
    
    def add_species(self, name: str, stype: str, mass: float,
                    delta_hf_kj: float = 0.0):
        if name in self._species:
            return
        st = SpeciesType(stype)
        sp = Species(name=name, type=st, mass=mass, delta_hf_kj=delta_hf_kj)
        self._species[name] = sp
    
    def load_from_yaml(self, filepath: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        for sp_data in data['species']:
            self.add_species(sp_data['name'], sp_data['type'], sp_data['mass'],
                             delta_hf_kj=sp_data.get('delta_hf_kj', 0.0))
    
    def finalize(self):
        """Assign indices: electron=0, then rest alphabetically."""
        idx = 0
        self._species_list = []
        
        # Electron first
        if 'e' in self._species:
            self._species['e'].index = 0
            self._species_list.append(self._species['e'])
            self._index_map['e'] = 0
            idx = 1
        
        # Rest sorted by name
        for name in sorted(self._species.keys()):
            if name == 'e':
                continue
            self._species[name].index = idx
            self._species_list.append(self._species[name])
            self._index_map[name] = idx
            idx += 1
        
        self._finalized = True
        print(f"  Species finalized: {self.n_species} species (electron at index 0)")
    
    @property
    def n_species(self) -> int:
        return len(self._species)
    
    @property
    def n_state(self) -> int:
        """Total state vector length: n_species + 2 (electron energy + Tgas)."""
        return self.n_species + 2
    
    @property
    def idx_energy(self) -> int:
        """Index of electron energy density in state vector."""
        return self.n_species
    
    @property
    def idx_Tgas(self) -> int:
        """Index of gas temperature in state vector."""
        return self.n_species + 1
    
    def index(self, name: str) -> int:
        return self._index_map[name]
    
    def get(self, name: str) -> Species:
        return self._species[name]
    
    def has(self, name: str) -> bool:
        return name in self._species
    
    def __iter__(self):
        return iter(self._species_list)
    
    def __len__(self):
        return len(self._species)
    
    @property
    def names(self) -> List[str]:
        return [sp.name for sp in self._species_list]
    
    @property
    def electron_index(self) -> int:
        return 0
    
    def get_mass_array(self) -> np.ndarray:
        """Return array of molar masses [kg/mol] in species order."""
        return np.array([sp.mass for sp in self._species_list])

    def get_delta_hf_array(self) -> np.ndarray:
        """Return array of ΔHf° [kJ/mol] at 298K in species order."""
        return np.array([sp.delta_hf_kj for sp in self._species_list])
