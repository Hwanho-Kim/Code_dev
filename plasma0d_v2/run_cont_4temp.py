#!/usr/bin/env python3
"""Try multiple power settings to find which matches CLAUDE.md baseline."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA

TEMPS = [303.0, 373.0, 453.0, 523.0]


def run_one(T_gas, overrides=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['flow'] = cfg.get('flow', {})
    cfg['flow']['model'] = 'PFR'
    cfg['initial']['T_gas'] = T_gas
    if overrides:
        for k, v in overrides.items():
            keys = k.split('.')
            d = cfg
            for kk in keys[:-1]:
                d = d[kk]
            d[keys[-1]] = v

    solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
    tau = solver.flow.get_physical_residence_time(T_gas)

    scfg = cfg_out['solver']
    result = solver.solve(
        y0, (0.0, tau), n_points=2000,
        method=scfg.get('method', 'BDF'),
        rtol=scfg.get('rtol', 1e-6),
        atol=scfg.get('atol', 1e-12),
        max_step=scfg.get('max_step', None),
        constrained=scfg.get('constrained', False),
    )

    ch4_i = solver.sm.index('CH4')
    ch4_0 = result.concentrations[ch4_i, 0]
    ch4_f = result.concentrations[ch4_i, -1]
    conv = (ch4_0 - ch4_f) / ch4_0 * 100 if ch4_0 > 0 else 0.0
    return conv, tau


def run_case(label, overrides=None):
    print(f"\n--- {label} ---")
    convs = []
    for T in TEMPS:
        conv, tau = run_one(T, overrides)
        convs.append(conv)
    print(f"  {label}: {'/'.join(f'{c:.2f}' for c in convs)}%")
    return convs


def main():
    print("=" * 60)
    print("  Finding CLAUDE.md baseline match")
    print("  Expected: 4.35 / 5.84 / 10.33 / 16.45%")
    print("=" * 60)

    # Default config (vi_envelope, P=1.62W, V_eff=4.9cm³, PFR)
    r1 = run_case("A: default vi_envelope P=1.62W")

    # Constant P=6.5W
    r2 = run_case("B: constant P=6.5W", {'power_mode': 'constant', 'P_input_W': 6.5})

    # vi_envelope P=6.5W
    r3 = run_case("C: vi_envelope P=6.5W", {'P_input_W': 6.5})

    print(f"\n{'='*60}")
    print(f"  CLAUDE.md  : 4.35 / 5.84 / 10.33 / 16.45%")
    print(f"  A(vi 1.62) : {'/'.join(f'{c:.2f}' for c in r1)}%")
    print(f"  B(c  6.5)  : {'/'.join(f'{c:.2f}' for c in r2)}%")
    print(f"  C(vi 6.5)  : {'/'.join(f'{c:.2f}' for c in r3)}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
