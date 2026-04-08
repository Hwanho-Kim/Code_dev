"""Parametric sweep: V_eff / V_reactor separation validation.

Continuous power mode, P=5W, Q=0.4 slm, 500ms simulation.
Sweeps V_reactor with fixed V_eff to find physically realistic combination
matching ~5% CH4 conversion.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasma0d_v2.config import load_config, setup_simulation


def run_single(V_eff_m3, V_reactor_m3, P_W, Q_slm, t_end_s=0.5):
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm

    cfg['solver'] = {
        't_end': t_end_s,
        'n_points': 2000,
        'method': 'BDF',
        'rtol': 1e-6,
        'atol': 1e-10,
        'max_step': 1e-4,
        'constrained': False,
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))

    scfg = cfg['solver']
    result = solver.solve(
        y0, t_span,
        n_points=scfg['n_points'],
        method=scfg['method'],
        rtol=scfg['rtol'],
        atol=scfg['atol'],
        max_step=scfg['max_step'],
        constrained=scfg.get('constrained', False),
    )

    ch4_idx = result.species_names.index('CH4')
    co2_idx = result.species_names.index('CO2')
    c_ch4_0 = result.concentrations[ch4_idx, 0]
    c_ch4_f = result.concentrations[ch4_idx, -1]
    c_co2_0 = result.concentrations[co2_idx, 0]
    c_co2_f = result.concentrations[co2_idx, -1]

    ch4_conv = (c_ch4_0 - c_ch4_f) / c_ch4_0 * 100 if c_ch4_0 > 0 else 0
    co2_conv = (c_co2_0 - c_co2_f) / c_co2_0 * 100 if c_co2_0 > 0 else 0

    tau_ms = solver.flow.get_residence_time(300.0) * 1e3

    return {
        'V_eff_cm3': V_eff_m3 * 1e6,
        'V_reactor_cm3': V_reactor_m3 * 1e6,
        'f': V_eff_m3 / V_reactor_m3,
        'tau_ms': tau_ms,
        'ne_final': result.ne_m3[-1],
        'Te_final': result.Te_eV[-1],
        'Tgas_final': result.T_gas[-1],
        'CH4_conv': ch4_conv,
        'CO2_conv': co2_conv,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    P_W = 5.0
    Q_slm = 0.4
    V_eff_cm3 = 0.4
    V_eff_m3 = V_eff_cm3 * 1e-6

    V_reactor_list_cm3 = [5, 10, 20, 50, 100, 200]

    print("=" * 100)
    print(f"  V_eff/V_reactor Separation Sweep")
    print(f"  P={P_W}W, Q={Q_slm} slm, V_eff={V_eff_cm3} cm³, t_end=500ms")
    print("=" * 100)

    results = []
    for V_r_cm3 in V_reactor_list_cm3:
        V_r_m3 = V_r_cm3 * 1e-6
        print(f"\n{'='*80}")
        print(f"  V_reactor = {V_r_cm3} cm³, f = {V_eff_m3/V_r_m3:.4f}")
        print(f"{'='*80}")
        try:
            r = run_single(V_eff_m3, V_r_m3, P_W, Q_slm, t_end_s=0.5)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 100)
    print(f"  SWEEP RESULTS: V_eff={V_eff_cm3} cm³, P={P_W}W, Q={Q_slm} slm")
    print("=" * 100)
    header = f"{'V_r(cm³)':>10} {'f':>8} {'τ(ms)':>8} {'n_e(m⁻³)':>12} {'Te(eV)':>8} {'Tg(K)':>8} {'CH4%':>8} {'CO2%':>8} {'time(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['V_reactor_cm3']:10.1f} {r['f']:8.4f} {r['tau_ms']:8.1f} "
              f"{r['ne_final']:12.2e} {r['Te_final']:8.2f} {r['Tgas_final']:8.1f} "
              f"{r['CH4_conv']:+8.2f} {r['CO2_conv']:+8.2f} {r['wall_time']:8.1f}")
