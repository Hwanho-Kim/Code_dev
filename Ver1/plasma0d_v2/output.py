"""Output: save simulation results as CSV, JSON, and plots."""

import numpy as np
import json
import os


def save_results(result, output_dir: str = 'output'):
    """Save all simulation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    save_time_evolution(result, output_dir)
    save_electron_params(result, output_dir)
    save_summary(result, output_dir)
    try:
        generate_plots(result, output_dir)
    except Exception as e:
        print(f"  Warning: plot generation failed: {e}")


def save_time_evolution(result, output_dir):
    """Save species concentrations vs time."""
    n_t = len(result.t)
    filepath = os.path.join(output_dir, 'time_evolution.csv')
    
    header = 'time_s'
    for name in result.species_names:
        header += f',c_{name}_mol_m3'
    
    data = np.zeros((n_t, 1 + result.n_species))
    data[:, 0] = result.t
    data[:, 1:] = result.concentrations.T
    
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')
    print(f"  Saved {filepath}")


def save_electron_params(result, output_dir):
    """Save electron parameters vs time."""
    filepath = os.path.join(output_dir, 'electron_params.csv')
    
    header = 'time_s,ne_m3,Te_eV,EN_Td,power_Wm3,Tgas_K'
    data = np.column_stack([
        result.t,
        result.ne_m3,
        result.Te_eV,
        result.EN_Td,
        result.power_Wm3,
        result.T_gas,
    ])
    
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')
    print(f"  Saved {filepath}")


def save_summary(result, output_dir):
    """Save simulation summary as JSON."""
    filepath = os.path.join(output_dir, 'summary.json')
    
    summary = {
        'n_species': result.n_species,
        'species_names': result.species_names,
        'n_time_points': len(result.t),
        'time_range_us': [float(result.t[0]*1e6), float(result.t[-1]*1e6)],
        'wall_time_s': result.wall_time,
        'n_rhs_evals': result.n_rhs_evals,
        'solver_message': result.solver_message,
        'final_values': {
            'ne_m3': float(result.ne_m3[-1]),
            'Te_eV': float(result.Te_eV[-1]),
            'Tgas_K': float(result.T_gas[-1]),
        },
    }
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {filepath}")


def generate_plots(result, output_dir):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    t_us = result.t * 1e6
    
    # 1. Electron density
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_us, result.ne_m3)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('n_e [m⁻³]')
    ax.set_title('Electron Density')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'electron_density.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Electron temperature
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_us, result.Te_eV)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Te [eV]')
    ax.set_title('Electron Temperature')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'electron_temperature.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Gas temperature
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_us, result.T_gas)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('T_gas [K]')
    ax.set_title('Gas Temperature')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'gas_temperature.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Major species concentrations
    fig, ax = plt.subplots(figsize=(12, 6))
    major = ['CH4', 'CO2', 'N2', 'O2', 'H2', 'H2O', 'CO', 'CH3', 'OH', 'O', 'H']
    for name in major:
        if name in result.species_names:
            idx = result.species_names.index(name)
            c = result.concentrations[idx, :]
            if np.max(c) > 1e-30:
                ax.semilogy(t_us, np.maximum(c, 1e-30), label=name)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Concentration [mol/m³]')
    ax.set_title('Major Species Concentrations')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'species_major.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Hydrocarbon products
    fig, ax = plt.subplots(figsize=(12, 6))
    products = ['C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'CH3OH', 'CH2O', 'C2H5OH']
    for name in products:
        if name in result.species_names:
            idx = result.species_names.index(name)
            c = result.concentrations[idx, :]
            if np.max(c) > 1e-30:
                ax.semilogy(t_us, np.maximum(c, 1e-30), label=name)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Concentration [mol/m³]')
    ax.set_title('Hydrocarbon Product Concentrations')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'species_products.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. E/N and power
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t_us, result.EN_Td)
    ax1.set_ylabel('E/N [Td]')
    ax1.set_title('Reduced Electric Field')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_us, result.power_Wm3 / 1e6)
    ax2.set_xlabel('Time [µs]')
    ax2.set_ylabel('Power [MW/m³]')
    ax2.set_title('Deposited Power Density')
    ax2.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'EN_and_power.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")
