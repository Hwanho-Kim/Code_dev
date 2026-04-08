#!/usr/bin/env python3
"""
Inference: load trained PINN → predict gas concentrations for 1D liquid sim.

Generates time series of ALL gas species (measured + predicted) for a given
(V, RH) condition. Output format matches what Ver4_1D/pde_solver.py expects.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from .data_loader import PlasmaOASDataset, T_MAX, V_MIN, V_MAX, RH_MIN, RH_MAX
from .model import PlasmaChemPINN, IDX_NO, IDX_HONO2, IDX_H2O2, IDX_Q_OH, IDX_Q_HONO, IDX_Q_NO
from .qssa import solve_oh_ho2_coupled
from .reactions import compute_n2o4


CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'


def load_model(checkpoint: str = 'best.pt', device: str = 'cpu') -> PlasmaChemPINN:
    model = PlasmaChemPINN()
    state = torch.load(CHECKPOINT_DIR / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_condition(
    model: PlasmaChemPINN,
    rh: float,
    v_kv: float,
    t_max: float = 600.0,
    dt: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Predict all gas species for one (RH, V) condition.

    Returns dict with keys: 'Time', 'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5',
                             'HONO', 'HONO2', 'H2O2', 'Q_OH', 'Q_HONO', 'Q_NO',
                             'OH', 'HO2'
    All concentrations in cm⁻³.
    """
    times = np.arange(0, t_max + dt, dt)
    n = len(times)

    t_norm = torch.tensor(times / T_MAX, dtype=torch.float32)
    v_norm = torch.full((n,), (v_kv - V_MIN) / (V_MAX - V_MIN), dtype=torch.float32)
    rh_norm = torch.full((n,), (rh - RH_MIN) / (RH_MAX - RH_MIN), dtype=torch.float32)
    inputs = torch.stack([t_norm, v_norm, rh_norm], dim=1)

    out = model.predict_concentrations(inputs)

    oh, ho2 = solve_oh_ho2_coupled(
        out['O3'], out['NO2'], out['HONO'],
        out['HONO2'], out['H2O2'], out['NO'], out['Q_OH'],
    )

    return {
        'Time':  times,
        'O3':    out['O3'].numpy(),
        'NO':    out['NO'].numpy(),
        'NO2':   out['NO2'].numpy(),
        'NO3':   out['NO3'].numpy(),
        'N2O4':  compute_n2o4(out['NO2']).numpy(),
        'N2O5':  out['N2O5'].numpy(),
        'HONO':  out['HONO'].numpy(),
        'HONO2': out['HONO2'].numpy(),
        'H2O2':  out['H2O2'].numpy(),
        'Q_OH':  out['Q_OH'].numpy(),
        'Q_HONO': out['Q_HONO'].numpy(),
        'Q_NO':  out['Q_NO'].numpy(),
        'OH':    oh.numpy(),
        'HO2':   ho2.numpy(),
    }


def predict_all_conditions(
    model: PlasmaChemPINN | None = None,
    checkpoint: str = 'best.pt',
) -> dict[tuple[int, float], dict]:
    """Predict for all 9 conditions. Returns {(RH, V): prediction_dict}."""
    if model is None:
        model = load_model(checkpoint)

    from .data_loader import RH_VALUES, V_VALUES
    results = {}
    for rh in RH_VALUES:
        for v in V_VALUES:
            results[(rh, v)] = predict_condition(model, rh, v)
    return results


if __name__ == '__main__':
    model = load_model()
    results = predict_all_conditions(model)

    for (rh, v), pred in results.items():
        no_max = pred['NO'].max()
        hono2_max = pred['HONO2'].max()
        h2o2_max = pred['H2O2'].max()
        oh_max = pred['OH'].max()
        print(
            f'RH={rh:2d}% V={v}kV: '
            f'NO={no_max:.2e} HONO₂={hono2_max:.2e} '
            f'H₂O₂={h2o2_max:.2e} OH={oh_max:.2e}'
        )
