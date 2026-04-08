#!/usr/bin/env python3
"""Trial runner for plasma0d_v2 solver troubleshooting.

Runs multiple solver configurations and reports key metrics.
Usage: python -m plasma0d_v2.trial_runner
"""

import sys
import os
import time
import json
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plasma0d_v2.config import load_config
from plasma0d_v2.main import setup_simulation
from plasma0d_v2.constants import NA


def run_trial(cfg, base_dir, trial_name, solver_overrides):
    """Run one trial with specific solver settings. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  TRIAL: {trial_name}")
    print(f"  Overrides: {solver_overrides}")
    print(f"{'='*60}")

    # Apply overrides
    for k, v in solver_overrides.items():
        cfg['solver'][k] = v

    try:
        solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)

        scfg = cfg_out['solver']
        t_start = time.time()
        result = solver.solve(
            y0, t_span,
            n_points=scfg.get('n_points', 2000),
            method=scfg.get('method', 'BDF'),
            rtol=scfg.get('rtol', 1e-6),
            atol=scfg.get('atol', 1e-12),
            max_step=scfg.get('max_step', None),
            seg_dt=scfg.get('seg_dt', 0.0),
        )
        wall_time = time.time() - t_start

        # Analyze results
        n_e = result.ne_m3
        Te = result.Te_eV
        Tg = result.T_gas
        t = result.t

        # Key metrics
        ne_min = np.min(n_e)
        ne_max = np.max(n_e)
        ne_final = n_e[-1]
        Te_max = np.max(Te)
        Te_final = Te[-1]
        Tg_final = Tg[-1] if Tg is not None else 0
        n_negative = int(np.sum(n_e < 0))

        # Check species negativity
        n_sp = result.n_species
        c_all = result.y[:n_sp, :]
        n_neg_species = 0
        neg_species_names = []
        for i in range(n_sp):
            if np.any(c_all[i, :] < 0):
                n_neg_species += 1
                neg_species_names.append(result.species_names[i])

        # Check pulse cycling (does Te rise during pulses?)
        # Find points where power > 0
        power = result.power_Wm3 if result.power_Wm3 is not None else np.zeros_like(t)
        pulse_mask = power > 0
        n_pulse_points = int(np.sum(pulse_mask))
        Te_during_pulse = Te[pulse_mask] if n_pulse_points > 0 else np.array([0])
        Te_during_after = Te[~pulse_mask] if np.sum(~pulse_mask) > 0 else np.array([0])

        metrics = {
            'trial': trial_name,
            'wall_time_s': round(wall_time, 1),
            'rhs_evals': result.n_rhs_evals,
            'n_points': len(t),
            'ne_min': f'{ne_min:.3e}',
            'ne_max': f'{ne_max:.3e}',
            'ne_final': f'{ne_final:.3e}',
            'Te_max_eV': round(Te_max, 3),
            'Te_final_eV': round(Te_final, 4),
            'Tg_final_K': round(Tg_final, 2),
            'n_negative_ne': n_negative,
            'n_neg_species': n_neg_species,
            'neg_species': neg_species_names[:5],  # first 5
            'Te_pulse_max': round(float(np.max(Te_during_pulse)), 3) if n_pulse_points > 0 else 0,
            'Te_after_min': round(float(np.min(Te_during_after)), 4) if np.sum(~pulse_mask) > 0 else 0,
            'solver_msg': result.solver_message[:60],
            'overrides': solver_overrides,
        }

        # Physical reasonableness check
        reasonable = (
            ne_min >= 0 and          # no negative n_e
            Te_max > 0.5 and         # Te rises above thermal
            n_neg_species == 0 and    # no negative species
            ne_final > 1e6            # n_e not collapsed to zero
        )
        metrics['REASONABLE'] = reasonable

        print(f"\n  RESULT: {'PASS' if reasonable else 'FAIL'}")
        for k, v in metrics.items():
            if k not in ('overrides', 'neg_species', 'solver_msg'):
                print(f"    {k}: {v}")
        if neg_species_names:
            print(f"    neg_species: {', '.join(neg_species_names[:5])}")

        return metrics

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return {
            'trial': trial_name,
            'REASONABLE': False,
            'error': str(e),
            'overrides': solver_overrides,
        }


def main():
    config_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
    base_dir = os.path.dirname(os.path.abspath(config_file))

    # Shorter test duration for quick iteration
    test_t_end = 1500.0e-6  # 1.5 ms = 2 full cycles

    trials = [
        # Trial 2: Single BDF, no segmentation, post-clamp only
        ("T2_BDF_single_postclamp", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'BDF', 'max_step': 1.0e-6, 'seg_dt': 0.0,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 3: Radau method
        ("T3_Radau_single", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'Radau', 'max_step': 1.0e-6, 'seg_dt': 0.0,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 4: BDF with tighter tolerances
        ("T4_BDF_tight_tol", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'BDF', 'max_step': 0.5e-6, 'seg_dt': 0.0,
            'rtol': 1e-8, 'atol': 1e-14,
        }),
        # Trial 5: BDF with segmented (250 µs = 6 segments per 1.5ms)
        ("T5_BDF_seg250us", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'BDF', 'max_step': 1.0e-6, 'seg_dt': 250.0e-6,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 6: Radau with segmented (250 µs)
        ("T6_Radau_seg250us", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'Radau', 'max_step': 1.0e-6, 'seg_dt': 250.0e-6,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 7: BDF smaller max_step
        ("T7_BDF_maxstep_100ns", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'BDF', 'max_step': 0.1e-6, 'seg_dt': 0.0,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 8: LSODA (auto-switches stiff/non-stiff)
        ("T8_LSODA_single", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'LSODA', 'max_step': 1.0e-6, 'seg_dt': 0.0,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
        # Trial 9: Segmented per pulse period (750 µs)
        ("T9_BDF_seg750us", {
            't_end': test_t_end, 'n_points': 3000,
            'method': 'BDF', 'max_step': 1.0e-6, 'seg_dt': 750.0e-6,
            'rtol': 1e-6, 'atol': 1e-10,
        }),
    ]

    all_results = []
    for name, overrides in trials:
        cfg = load_config(config_file)
        metrics = run_trial(cfg, base_dir, name, overrides)
        all_results.append(metrics)

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Trial':<25} {'Time':>6} {'RHS':>8} {'ne_min':>12} {'ne_final':>12} "
          f"{'Te_max':>7} {'#neg':>5} {'OK?':>4}")
    print("-" * 80)
    for m in all_results:
        if 'error' in m:
            print(f"{m['trial']:<25} ERROR: {m['error'][:50]}")
        else:
            print(f"{m['trial']:<25} {m['wall_time_s']:>5.0f}s {m['rhs_evals']:>8} "
                  f"{m['ne_min']:>12} {m['ne_final']:>12} "
                  f"{m['Te_max_eV']:>6.2f} {m['n_neg_species']:>5} "
                  f"{'YES' if m['REASONABLE'] else 'NO':>4}")

    # Save results
    out_file = os.path.join(base_dir, 'trial_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    return all_results


if __name__ == '__main__':
    main()
