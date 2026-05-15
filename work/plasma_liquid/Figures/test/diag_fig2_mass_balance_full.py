#!/usr/bin/env python3
"""Final mass-balance check for fig2 + fig2b in user's framework.

Per species panel:
  Σ panel = (strict-direct rxns, full stoich) + (AB equilibration line)
            + (MT line, for transferable species)

For each species, compute Σ panel at selected snapshots and compare to FD
dC/dt (truth). Residual sources:
  - diffusion (HO2_total / HONO_total / etc. pool transport across cells)
  - ∂f/∂t × pool (pH drift; only for acid-base members)
For non-pair / non-transferable (OH, HO3, O3-): residual = diffusion only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
sys.path.insert(0, str(_proj_root / 'Figures'))

from chemistry_1d import AqueousChemistry1D
from config_1d import ACID_BASE_PAIRS, GAS_TO_AQUEOUS_MAP
import gen_all_figures as gaf
from gen_all_figures import _get_solver

SPEC_TO_TOTAL = {
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'NO2-': 'HONO_total', 'HONO': 'HONO_total',
    'NO3-': 'HONO2_total', 'HONO2': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
}


def vol_avg(arr, dz, L):
    return float((arr * dz).sum() / L)


def compute_budget(solver, snap_t, snap_y, snap_i, sp, mt_flux_dict_t):
    """Return Σ panel in user's framework + components."""
    chem = solver.chem
    dz, L = solver.dz_cells, solver.L
    h_idx = chem.species_idx['H+']
    N_z = solver.N_z
    n_rxn = len(chem.reactions)

    y_2d = snap_y[snap_i]
    Hp_z = np.maximum(y_2d[:, h_idx], 1e-14)

    # Per-cell rates
    rates_2d = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y_2d[j, :].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates_2d[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)

    # (1) Strict direct rxns
    direct = 0.0
    for ri, rxn in enumerate(chem.reactions):
        reac = rxn['reactants']
        prod = rxn.get('products', {})
        stoich = int(prod.get(sp, 0)) - int(reac.get(sp, 0))
        if stoich == 0:
            continue
        direct += stoich * vol_avg(rates_2d[ri], dz, L)

    # (2) AB line
    ab = 0.0
    total = SPEC_TO_TOTAL.get(sp)
    if total is not None and total in ACID_BASE_PAIRS:
        acid, base, pKa = ACID_BASE_PAIRS[total]
        Ka = 10 ** -pKa
        f_mol_z = Hp_z / (Hp_z + Ka)
        f_ion_z = Ka / (Hp_z + Ka)
        acid_z = np.zeros(N_z)
        base_z = np.zeros(N_z)
        for ri, rxn in enumerate(chem.reactions):
            reac = rxn['reactants']
            prod = rxn.get('products', {})
            s_a = int(prod.get(acid, 0)) - int(reac.get(acid, 0))
            s_b = int(prod.get(base, 0)) - int(reac.get(base, 0))
            if s_a != 0:
                acid_z += s_a * rates_2d[ri]
            if s_b != 0:
                base_z += s_b * rates_2d[ri]
        if sp == acid:
            ab_z = -f_ion_z * acid_z + f_mol_z * base_z
        else:
            ab_z = f_ion_z * acid_z - f_mol_z * base_z
        ab = vol_avg(ab_z, dz, L)

    # (3) MT line
    mt = 0.0
    if total is not None and total in ACID_BASE_PAIRS:
        acid, base, pKa = ACID_BASE_PAIRS[total]
        Ka = 10 ** -pKa
        f_mol_z = Hp_z / (Hp_z + Ka)
        f_ion_z = Ka / (Hp_z + Ka)
        f_self_surface = f_mol_z[0] if sp == acid else f_ion_z[0]
        match_set = {acid, base, total}
        mt_pool = sum(mt_flux_dict_t.get(n, 0.0) for n in match_set)
        mt = f_self_surface * mt_pool
    else:
        # non-pair: MT enters as self
        mt = mt_flux_dict_t.get(sp, 0.0)

    return direct, ab, mt, direct + ab + mt


def main():
    VOLT = '3.6kV'
    cache = (_proj_root / 'Figures' / 'DIW results'
             / f'{VOLT}_Humid_fitting_three_film_HONOvar_v3'
             / 'cache' / 'three_film_abspecies_dg0.0100.npz')
    d = dict(np.load(cache, allow_pickle=True))
    snap_y = d['snap_y']; snap_t = d['snap_t']
    dz = d['dz_cells']; L = float(d['L'])
    nt = len(snap_t)

    # Solver for compute_rates_snapshot (MT)
    gaf.DEFAULT_GAS_SHEET = VOLT
    times, gas_conc = gaf.load_gas_data()
    rh80 = gaf.RH80_RATIOS.get(VOLT, {})
    no2_arr = gas_conc.get('NO2', np.zeros_like(times))
    n2o5_arr = gas_conc.get('N2O5', np.zeros_like(times))
    o3_arr = gas_conc.get('O3', np.zeros_like(times))
    hono_gas = no2_arr * rh80.get('HONO_NO2', 0.097)
    hono2_gas = n2o5_arr * rh80.get('HONO2_N2O5', 0.83)
    h2o2_gas = o3_arr * rh80.get('H2O2_O3', 0.003)
    solver = _get_solver(times, gas_conc)
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)

    chem = solver.chem
    idx = chem.species_idx
    h_idx = idx['H+']

    species_list = ['HO2', 'HO3', 'O2-', 'O3-', 'OH',
                    'NO3-', 'O3', 'NO2-', 'H2O2']

    # Vol-avg [sp](t) — speciated for acid-base, direct for non-pair
    def conc_t(sp):
        total = SPEC_TO_TOTAL.get(sp)
        if total is not None and total in ACID_BASE_PAIRS and total in idx:
            acid, base, pKa = ACID_BASE_PAIRS[total]
            Ka = 10 ** -pKa
            tot_idx = idx[total]
            c = np.zeros(nt)
            for i in range(nt):
                Hp = np.maximum(snap_y[i][:, h_idx], 1e-14)
                f = Hp / (Hp + Ka) if sp == acid else Ka / (Hp + Ka)
                c[i] = vol_avg(snap_y[i][:, tot_idx] * f, dz, L)
            return c
        elif sp in idx:
            sp_i = idx[sp]
            return np.array([vol_avg(snap_y[i][:, sp_i], dz, L)
                             for i in range(nt)])
        return np.zeros(nt)

    # FD dC/dt
    dCdt = {sp: np.gradient(conc_t(sp), snap_t) for sp in species_list}

    test_t = [60, 120, 180, 300, 600]
    test_i = [int(np.argmin(np.abs(snap_t - t))) for t in test_t]

    print(f'\n=== Mass balance @ {VOLT}, user framework (strict + AB + MT) ===')
    print(f'{"sp":<6} {"t":<5} {"direct":<12} {"AB line":<12} '
          f'{"MT line":<12} {"Σ panel":<12} {"FD truth":<12} '
          f'{"Σ/FD":<7} {"resid frac":<10}')
    for sp in species_list:
        for i_snap in test_i:
            t_i = snap_t[i_snap]
            # MT
            _, mt_flux_t = gaf.compute_rates_snapshot(solver, snap_y[i_snap], t_i)
            d_, a, m, sum_panel = compute_budget(
                solver, snap_t, snap_y, i_snap, sp, mt_flux_t)
            fd = dCdt[sp][i_snap]
            ratio = sum_panel / fd if abs(fd) > 1e-25 else float('nan')
            resid_frac = (fd - sum_panel) / fd if abs(fd) > 1e-25 else float('nan')
            print(f'{sp:<6} {t_i:<5.0f} {d_:+.3e}  {a:+.3e}  {m:+.3e}  '
                  f'{sum_panel:+.3e}  {fd:+.3e}  {ratio:<7.3f} {resid_frac:<10.3f}')


if __name__ == '__main__':
    main()
