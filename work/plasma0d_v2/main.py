"""Main entry point for plasma0d_v2 simulation."""

import argparse
import os
import sys
import numpy as np

from .config import load_config, setup_simulation
from .output import save_results


def generate_demo_vi_curve(filepath: str):
    """Generate synthetic DBD V-I curve for demo."""
    freq = 10e3
    T = 1.0 / freq
    n_pts = 5000
    t = np.linspace(0, T, n_pts)
    V_peak = 5000.0
    V = V_peak * np.sin(2 * np.pi * freq * t)
    C_dbd = 20e-12
    I_disp = C_dbd * V_peak * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    phase = 2 * np.pi * freq * t
    I_spike = 0.05 * np.exp(-((np.sin(phase))**2 - 1)**2 / 0.01) * np.sign(np.cos(phase))
    I = I_disp + I_spike
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("time_s,voltage_V,current_A\n")
        for i in range(n_pts):
            f.write(f"{t[i]:.10e},{V[i]:.6e},{I[i]:.6e}\n")
    print(f"  Generated synthetic V-I curve: {filepath}")


def run_demo(base_dir: str = None):
    """Run demo simulation."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dir = os.path.join(base_dir, 'input')
    vi_file = os.path.join(input_dir, 'vi_curve.csv')
    
    if not os.path.exists(vi_file):
        generate_demo_vi_curve(vi_file)
    
    cfg = {
        'input_dir': 'input',
        'species_file': 'species.yaml',
        'reactions_file': 'reactions.yaml',
        'cross_section_dir': 'cross_sections',
        'vi_curve_file': 'vi_curve.csv',
        # vi_curve_options: {} -- auto-detect
        # EN_min / EN_max: omitted -> auto from V-I curve
        'EN_points': 200,
        'reactor': {
            'd_gap': 1.0e-3,
            'volume': 1.0e-6,
            'pressure': 101325.0,
        },
        'inlet_composition': {
            'CH4': 0.05,
            'CO2': 0.05,
            'N2': 0.78,
            'O2': 0.12,
        },
        'flow': {'Q_slm': 1.0},
        'initial': {
            'T_gas': 300.0,
            'ne': 1e12,
            'Te_eV': 2.0,
        },
        # solver params: omitted -> auto from V-I pulse analysis
        'T_wall': 300.0,
        'M_avg': 0.028,
        'cp_avg': 1040.0,
        'wall_loss_freq': 100.0,
    }
    
    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    
    scfg = cfg['solver']
    result = solver.solve(
        y0, t_span,
        n_points=scfg.get('n_points', 2000),
        method=scfg.get('method', 'BDF'),
        rtol=scfg.get('rtol', 1e-6),
        atol=scfg.get('atol', 1e-12),
        max_step=scfg.get('max_step', None),
    )
    
    output_dir = os.path.join(base_dir, 'output')
    save_results(result, output_dir)
    _print_final_summary(result)
    return result


def run_config(config_file: str):
    """Run simulation from YAML config file."""
    cfg = load_config(config_file)
    base_dir = os.path.dirname(os.path.abspath(config_file))
    
    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    
    scfg = cfg['solver']
    result = solver.solve(
        y0, t_span,
        n_points=scfg.get('n_points', 2000),
        method=scfg.get('method', 'BDF'),
        rtol=scfg.get('rtol', 1e-6),
        atol=scfg.get('atol', 1e-12),
        max_step=scfg.get('max_step', None),
        seg_dt=scfg.get('seg_dt', 0.0),
        constrained=scfg.get('constrained', False),
    )
    
    output_dir = os.path.join(base_dir, cfg.get('output_dir', 'output'))
    save_results(result, output_dir)
    _print_final_summary(result)
    return result


def _print_final_summary(result):
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Final n_e     = {result.ne_m3[-1]:.3e} m-3")
    print(f"  Final Te      = {result.Te_eV[-1]:.2f} eV")
    print(f"  Final T_gas   = {result.T_gas[-1]:.1f} K")
    
    if 'CH4' in result.species_names:
        idx = result.species_names.index('CH4')
        c0 = result.concentrations[idx, 0]
        cf = result.concentrations[idx, -1]
        if c0 > 0:
            print(f"  CH4 conversion = {(c0-cf)/c0*100:.4f}%")
    
    print(f"  Wall time     = {result.wall_time:.1f}s")
    print(f"  RHS evals     = {result.n_rhs_evals}")


def main():
    parser = argparse.ArgumentParser(description='0D Plasma Chemistry Simulation')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic V-I')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.config:
        run_config(args.config)
    else:
        parser.print_help()
        print("\nUsage:")
        print("  python -m plasma0d_v2 --demo              # synthetic V-I demo")
        print("  python -m plasma0d_v2 --config config.yaml  # real V-I curve")


if __name__ == '__main__':
    main()
