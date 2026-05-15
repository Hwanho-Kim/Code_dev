#!/usr/bin/env python3
"""Final mass-balance check after AB-residual closure fix.

Σ panel = strict_direct + MT_line + AB_flux  (by construction = ΔC/Δt FD)
The AB_flux = ΔC/Δt - strict_direct - MT_line captures HO2 ⇌ H+ + O2- net
equilibrium flux (= ⟨HO2_total × ∂f/∂t⟩ + spatial covariance + diff residual).
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
import gen_all_figures as gaf
from gen_all_figures import _get_solver, compute_rates_snapshot

SPEC_TO_TOTAL = {
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'NO2-': 'HONO_total', 'HONO': 'HONO_total',
    'NO3-': 'HONO2_total', 'HONO2': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
}


def run_for_voltage(VOLT):
    cache = (_proj_root / 'Figures' / 'DIW results'
             / f'{VOLT}_Humid_fitting_three_film_HONOvar_v3'
             / 'cache' / 'three_film_abspecies_dg0.0100.npz')
    d = dict(np.load(cache, allow_pickle=True))
    snap_y = d['snap_y']; snap_t = d['snap_t']
    dz = d['dz_cells']; L = float(d['L'])
    nt, N_z, _ = snap_y.shape

    gaf.DEFAULT_GAS_SHEET = VOLT
    times, gas_conc = gaf.load_gas_data()
    rh80 = gaf.RH80_RATIOS.get(VOLT, {})
    hono_gas = gas_conc.get('NO2', np.zeros_like(times)) * rh80.get('HONO_NO2', 0.097)
    hono2_gas = gas_conc.get('N2O5', np.zeros_like(times)) * rh80.get('HONO2_N2O5', 0.83)
    h2o2_gas = gas_conc.get('O3', np.zeros_like(times)) * rh80.get('H2O2_O3', 0.003)
    solver = _get_solver(times, gas_conc)
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)

    chem = solver.chem
    idx = chem.species_idx
    h_idx = idx['H+']
    n_rxn = len(chem.reactions)

    # Per-snapshot per-cell rates + MT
    print('Computing per-cell rates + MT...')
    all_rates = np.zeros((nt, n_rxn, N_z))
    all_mt = []
    for i in range(nt):
        for j in range(N_z):
            yc = np.clip(snap_y[i][j].copy(), chem.trace, 1.0)
            yc[h_idx] = max(yc[h_idx], 1e-14)
            spec = chem.speciate(yc)
            for ri, rxn_d in enumerate(chem._rxn_data):
                all_rates[i, ri, j] = chem._compute_single_rate(rxn_d, yc, spec)
        _, mt_i = compute_rates_snapshot(solver, snap_y[i], snap_t[i])
        all_mt.append(mt_i)

    species_list = ['HO2', 'HO3', 'O2-', 'O3-', 'OH',
                    'NO3-', 'O3', 'NO2-', 'H2O2']

    def speciated_conc_t(sp):
        total = SPEC_TO_TOTAL.get(sp)
        if total is not None and total in ACID_BASE_PAIRS and total in idx:
            acid, base, pKa = ACID_BASE_PAIRS[total]
            Ka = 10 ** -pKa
            tot_idx = idx[total]
            c = np.zeros(nt)
            for i in range(nt):
                Hp = np.maximum(snap_y[i][:, h_idx], 1e-14)
                f = Hp/(Hp+Ka) if sp == acid else Ka/(Hp+Ka)
                c[i] = np.dot(snap_y[i][:, tot_idx] * f, dz) / L
            return c
        elif sp in idx:
            sp_i = idx[sp]
            return np.array([np.dot(snap_y[i][:, sp_i], dz)/L for i in range(nt)])
        return np.zeros(nt)

    test_t = [60, 120, 180, 300, 600]
    test_i = [int(np.argmin(np.abs(snap_t - t))) for t in test_t]

    print(f'\n=== Mass balance @ {VOLT} (residual AB closure) ===')
    print(f'{"sp":<6} {"t":<5} {"Σ panel":<13} {"FD truth":<13} '
          f'{"Σ/FD":<10} {"|Σ−FD|/|FD|":<12}')

    worst_ratio_err = 0.0
    worst_sp_t = ''
    for sp in species_list:
        total = SPEC_TO_TOTAL.get(sp)
        c_t = speciated_conc_t(sp)
        dc_dt = np.gradient(c_t, snap_t)
        # strict-direct per time
        strict_t = np.zeros(nt)
        for ri, rxn in enumerate(chem.reactions):
            reac = rxn['reactants']; prod = rxn.get('products', {})
            stoich = int(prod.get(sp, 0)) - int(reac.get(sp, 0))
            if stoich == 0:
                continue
            strict_t += stoich * (all_rates[:, ri, :] * dz[None, :]).sum(axis=1) / L
        # MT
        mt_t = np.zeros(nt)
        if total is not None and total in ACID_BASE_PAIRS:
            acid, base, pKa = ACID_BASE_PAIRS[total]
            Ka = 10 ** -pKa
            match = {acid, base, total}
            for i in range(nt):
                Hp_s = max(snap_y[i][0, h_idx], 1e-14)
                f_self = Hp_s/(Hp_s+Ka) if sp == acid else Ka/(Hp_s+Ka)
                mt_t[i] = f_self * sum(all_mt[i].get(n, 0.0) for n in match)
        else:
            for i in range(nt):
                mt_t[i] = all_mt[i].get(sp, 0.0)
        # AB residual
        ab_t = dc_dt - strict_t - mt_t if total in ACID_BASE_PAIRS else np.zeros(nt)
        sum_t = strict_t + mt_t + ab_t

        for snap_idx in test_i:
            t_v = snap_t[snap_idx]
            s, fd = sum_t[snap_idx], dc_dt[snap_idx]
            if abs(fd) > 1e-30:
                ratio = s / fd
                rel_err = abs(s - fd) / abs(fd)
                if rel_err > worst_ratio_err:
                    worst_ratio_err = rel_err
                    worst_sp_t = f'{sp}@{t_v:.0f}s'
            else:
                ratio = float('nan')
                rel_err = float('nan')
            print(f'{sp:<6} {t_v:<5.0f} {s:+.3e}  {fd:+.3e}  '
                  f'{ratio:<10.6f} {rel_err:<12.2e}')

    print(f'\n  WORST relative error: {worst_ratio_err:.2e} at {worst_sp_t}')


def main():
    for V in ['2.6kV', '3.2kV', '3.6kV']:
        run_for_voltage(V)


if __name__ == '__main__':
    main()
