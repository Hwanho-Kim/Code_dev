"""
Data loader for Humidity Experiment OAS measurements.

Loads 9 CSV files (3 RH × 3 V), normalizes inputs, creates PyTorch tensors.
File naming: {RH}_{kV}.csv  (e.g., 25_3.2.csv → RH=25%, V=3.2kV)

Columns: Time, O3, NO2, NO3, N2O5, HONO
Units: Time in seconds, concentrations in cm⁻³
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── Data directory ──
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / 'empty chamber' / 'empty chamber' / 'Humidity exp'
)

# ── Conditions ──
RH_VALUES = [25, 55, 65]       # %
V_VALUES  = [2.6, 3.2, 3.6]   # kV

# ── Measured species (column order in CSV) ──
MEASURED_SPECIES = ['O3', 'NO2', 'NO3', 'N2O5', 'HONO']

# ── Normalization constants ──
T_MAX  = 600.0    # seconds
V_MIN  = 2.6      # kV
V_MAX  = 3.6
RH_MIN = 25.0     # %
RH_MAX = 65.0

# Log-scale floors to avoid log(0)
CONC_FLOOR = 1e6   # cm⁻³ (below any physical signal)


def _normalize_inputs(t: np.ndarray, v: float, rh: float) -> tuple:
    """Normalize (t, V, RH) to approximately [0, 1]."""
    t_norm  = t / T_MAX
    v_norm  = (v - V_MIN) / (V_MAX - V_MIN)
    rh_norm = (rh - RH_MIN) / (RH_MAX - RH_MIN)
    return t_norm, v_norm, rh_norm


class PlasmaOASDataset:
    """
    Loads all 9 conditions into a single dataset.

    Attributes:
        inputs:     (N_total, 3) tensor — (t̃, Ṽ, R̃H) normalized
        targets:    (N_total, 5) tensor — log₁₀ measured concentrations
        raw_conc:   (N_total, 5) tensor — raw concentrations (cm⁻³)
        conditions: list of (rh, v) tuples matching data order
        cond_idx:   (N_total,) tensor — condition index [0..8]
        n_per_cond: int — number of time points per condition
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        exclude_condition: Optional[tuple[int, float]] = None,
    ):
        """
        Args:
            data_dir: Override data directory.
            exclude_condition: (RH, V) tuple to exclude for leave-one-out CV.
        """
        if data_dir is None:
            data_dir = DATA_DIR

        all_inputs = []
        all_targets = []
        all_raw = []
        all_cond_idx = []
        self.conditions = []
        cond_i = 0

        for rh in RH_VALUES:
            for v in V_VALUES:
                if exclude_condition and (rh, v) == exclude_condition:
                    continue

                fname = f'{rh}_{v}.csv'
                fpath = data_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f'Missing: {fpath}')

                df = pd.read_csv(fpath).dropna(subset=['Time'])
                t = df['Time'].values.astype(np.float64)
                n_pts = len(t)

                # Normalize inputs
                t_norm, v_norm, rh_norm = _normalize_inputs(t, v, rh)
                inp = np.column_stack([
                    t_norm,
                    np.full(n_pts, v_norm),
                    np.full(n_pts, rh_norm),
                ])
                all_inputs.append(inp)

                # Concentrations (floor zeros)
                conc = np.zeros((n_pts, len(MEASURED_SPECIES)), dtype=np.float64)
                for j, sp in enumerate(MEASURED_SPECIES):
                    vals = df[sp].values.astype(np.float64)
                    conc[:, j] = np.maximum(vals, CONC_FLOOR)
                all_raw.append(conc)

                # Log₁₀ targets
                all_targets.append(np.log10(conc))

                # Condition index
                all_cond_idx.append(np.full(n_pts, cond_i, dtype=np.int64))
                self.conditions.append((rh, v))
                cond_i += 1

        # Stack
        self.inputs   = torch.tensor(np.vstack(all_inputs),   dtype=torch.float32)
        self.targets  = torch.tensor(np.vstack(all_targets),  dtype=torch.float32)
        self.raw_conc = torch.tensor(np.vstack(all_raw),      dtype=torch.float32)
        self.cond_idx = torch.tensor(np.concatenate(all_cond_idx), dtype=torch.long)
        self.n_per_cond = len(pd.read_csv(data_dir / f'{RH_VALUES[0]}_{V_VALUES[0]}.csv'))
        self.n_conditions = cond_i

    def __len__(self) -> int:
        return len(self.inputs)

    def get_condition_mask(self, cond_i: int) -> torch.Tensor:
        """Boolean mask for a specific condition index."""
        return self.cond_idx == cond_i

    def get_condition_data(self, cond_i: int):
        """Return (inputs, targets, raw_conc) for one condition."""
        mask = self.get_condition_mask(cond_i)
        return self.inputs[mask], self.targets[mask], self.raw_conc[mask]

    def summary(self) -> str:
        lines = [
            f'PlasmaOASDataset: {len(self)} points, {self.n_conditions} conditions',
            f'  Points per condition: {self.n_per_cond}',
            f'  Input shape:  {tuple(self.inputs.shape)}',
            f'  Target shape: {tuple(self.targets.shape)}',
        ]
        for i, (rh, v) in enumerate(self.conditions):
            mask = self.get_condition_mask(i)
            conc = self.raw_conc[mask]
            lines.append(
                f'  [{i}] RH={rh}% V={v}kV: '
                f'O₃={conc[:,0].max():.2e} NO₂={conc[:,1].max():.2e} '
                f'HONO={conc[:,4].max():.2e}'
            )
        return '\n'.join(lines)


if __name__ == '__main__':
    ds = PlasmaOASDataset()
    print(ds.summary())
