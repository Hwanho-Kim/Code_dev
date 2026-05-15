#!/usr/bin/env python3
"""Verify mass balance for fig2b speciation-weighted budget.

For each panel species:
  budget_sum(t) = Σ_rxn (stoich × ∫ rate(z,t) × f(z,t) dz / L)
  dC/dt_truth(t) = finite-diff of vol-avg [species](t) (includes diffusion + ∂f/∂t)

Expected discrepancy:
- Non-acid-base (OH, HO3, O3-): diffusion across cells only
- Acid-base (HO2, O2-): diffusion + (∂f/∂t × pool) term not captured
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
sys.path.insert(0, str(_proj_root / 'Figures'))

from chemistry_1d import AqueousChemistry1D
from config_1d import ACID_BASE_PAIRS


SPEC_TO_TOTAL_PAIRS = {
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'NO2-': 'HONO_total', 'HONO': 'HONO_total',
    'NO3-': 'HONO2_total', 'HONO2': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
}


def main():
    VOLT = '3.6kV'
    cache = (_proj_root / 'Figures' / 'DIW results'
             / f'{VOLT}_Humid_fitting_three_film_HONOvar_v3'
             / 'cache' / 'three_film_abspecies_dg0.0100.npz')
    d = dict(np.load(cache, allow_pickle=True))
    snap_y = d['snap_y']; snap_t = d['snap_t']
    dz = d['dz_cells']; L = float(d['L'])
    nt, N_z, N_s = snap_y.shape

    chem = AqueousChemistry1D(saline_mode=False)
    idx = chem.species_idx
    h_idx = idx['H+']
    n_rxn = len(chem.reactions)

    # snapshots to check
    test_t = [60, 120, 180, 300, 600]
    test_i = [int(np.argmin(np.abs(snap_t - t))) for t in test_t]
    species_panels = ['HO2', 'HO3', 'O2-', 'O3-', 'OH']

    # Precompute [sp](t) for each panel species
    print('Computing vol-avg [species](t) for finite-diff truth...')
    conc_t = {sp: np.zeros(nt) for sp in species_panels}
    for sp in species_panels:
        total = SPEC_TO_TOTAL_PAIRS.get(sp)
        if total is not None and total in idx:
            t_idx = idx[total]
            pair = ACID_BASE_PAIRS[total]
            acid_form, base_form, pKa = pair
            Ka = 10 ** -pKa
            for i in range(nt):
                Hp_z = np.maximum(snap_y[i][:, h_idx], 1e-14)
                f = Hp_z/(Hp_z+Ka) if sp == acid_form else Ka/(Hp_z+Ka)
                conc_t[sp][i] = np.dot(snap_y[i][:, t_idx] * f, dz) / L
        else:
            sp_idx = idx[sp]
            for i in range(nt):
                conc_t[sp][i] = np.dot(snap_y[i][:, sp_idx], dz) / L

    # finite-diff dC/dt
    dCdt = {sp: np.gradient(conc_t[sp], snap_t) for sp in species_panels}

    # Compute budget Σ for each snapshot we want
    print(f'\nMass balance at selected t values ({VOLT}):')
    print(f'{"species":<6} {"t [s]":<6} {"budget Σ":<13} {"dC/dt FD":<13} '
          f'{"Σ/FD":<10} {"residual":<13}')
    for sp in species_panels:
        total = SPEC_TO_TOTAL_PAIRS.get(sp)
        if total is not None and total in ACID_BASE_PAIRS:
            acid_form, base_form, pKa = ACID_BASE_PAIRS[total]
            Ka = 10 ** -pKa
            match_set = {acid_form, base_form, total}
        else:
            match_set = {sp}

        for i in test_i:
            # per-cell rates at this snapshot
            rates_2d = np.zeros((n_rxn, N_z))
            for j in range(N_z):
                yc = np.clip(snap_y[i][j, :].copy(), chem.trace, 1.0)
                yc[h_idx] = max(yc[h_idx], 1e-14)
                spec_c = chem.speciate(yc)
                for ri, rxn_d in enumerate(chem._rxn_data):
                    rates_2d[ri, j] = chem._compute_single_rate(rxn_d, yc, spec_c)
            Hp_z = np.maximum(snap_y[i][:, h_idx], 1e-14)
            if total is not None and total in ACID_BASE_PAIRS:
                f_z = Hp_z/(Hp_z+Ka) if sp == acid_form else Ka/(Hp_z+Ka)
            else:
                f_z = np.ones(N_z)
            weights = f_z * dz
            speciated = (rates_2d * weights[None, :]).sum(axis=1) / L  # (n_rxn,)

            budget = 0.0
            for ri, rxn in enumerate(chem.reactions):
                reac = rxn['reactants']; prod = rxn.get('products', {})
                in_r = set(reac.keys()) & match_set
                in_p = set(prod.keys()) & match_set
                if not in_r and not in_p:
                    continue
                stoich = (sum(int(prod[s]) for s in in_p)
                          - sum(int(reac[s]) for s in in_r))
                budget += stoich * speciated[ri]

            fd = dCdt[sp][i]
            ratio = budget / fd if abs(fd) > 1e-25 else float('nan')
            resid = budget - fd
            print(f'{sp:<6} {snap_t[i]:<6.0f} {budget:+.3e}   {fd:+.3e}   '
                  f'{ratio:<10.3f} {resid:+.3e}')


if __name__ == '__main__':
    main()
