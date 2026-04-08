"""Test A/V ratio effect on mass transfer"""
import numpy as np
import importlib
import config
importlib.reload(config)

from chemistry_utils import (
    calculate_adjusted_diffusivity,
    calculate_mass_transfer_coefficient,
    molecules_to_molar,
)
from config import HENRY_CONSTANTS, MASS_TRANSFER

print('='*80)
print('A/V Ratio Effect on Mass Transfer')
print('='*80)
print(f'Fixed: dx_liq = {MASS_TRANSFER.delta_x_liq*1e6:.0f} um, dx_gas = {MASS_TRANSFER.delta_x_gas*1e3:.0f} mm')
print()

# A/V values to test (m^-1)
# A/V = 1/depth for flat surface
av_ratios = [100, 500, 1000, 5000, 10000]
depths_mm = [1000/av for av in av_ratios]

species_test = ['NO2', 'N2O5', 'H2O2']
gas_concs = {'NO2': 1e14, 'N2O5': 1e12, 'H2O2': 1e11}

# Part 1: k_mt vs A/V
print('Part 1: k_mt [s^-1] vs A/V [m^-1]')
print('-'*80)
print(f'{"Species":8} {"H":>10}' + ''.join([f'{av:>12}' for av in av_ratios]))
print(' '*20 + ''.join([f'({d:.1f}mm)'.rjust(12) for d in depths_mm]))
print('-'*80)

for species in species_test:
    H = HENRY_CONSTANTS.get(species, 1.0)
    row = f'{species:8} {H:>10.1e}'

    for av in av_ratios:
        D_adj = calculate_adjusted_diffusivity(species, H)
        k_mt = (D_adj / MASS_TRANSFER.delta_x_liq) * av
        row += f'{k_mt:>12.2e}'

    print(row)

# Part 2: Time constant
print()
print('Part 2: Time constant tau = 1/k_mt [seconds]')
print('-'*80)
print(f'{"Species":8} {"H":>10}' + ''.join([f'{av:>12}' for av in av_ratios]))
print('-'*80)

for species in species_test:
    H = HENRY_CONSTANTS.get(species, 1.0)
    row = f'{species:8} {H:>10.1e}'

    for av in av_ratios:
        D_adj = calculate_adjusted_diffusivity(species, H)
        k_mt = (D_adj / MASS_TRANSFER.delta_x_liq) * av
        tau = 1/k_mt if k_mt > 0 else float('inf')
        row += f'{tau:>12.1f}'

    print(row)

# Part 3: Concentration after 60s
print()
print('Part 3: Concentration after 60s [mol/L]')
print('-'*80)
print(f'{"Species":8} {"C_eq":>12}' + ''.join([f'{av:>12}' for av in av_ratios]))
print('-'*80)

for species in species_test:
    H = HENRY_CONSTANTS.get(species, 1.0)
    gas_conc = gas_concs[species]
    C_eq = H * molecules_to_molar(gas_conc)

    row = f'{species:8} {C_eq:>12.2e}'

    for av in av_ratios:
        D_adj = calculate_adjusted_diffusivity(species, H)
        k_mt = (D_adj / MASS_TRANSFER.delta_x_liq) * av

        C_aq = 0.0
        dt = 0.1
        for _ in range(600):
            dC = k_mt * (C_eq - C_aq) * dt
            C_aq += dC

        row += f'{C_aq:>12.2e}'

    print(row)

# Part 4: Percentage of equilibrium
print()
print('Part 4: Percentage of equilibrium reached after 60s')
print('-'*80)
print(f'{"Species":8} {"C_eq":>12}' + ''.join([f'{av:>12}' for av in av_ratios]))
print('-'*80)

for species in species_test:
    H = HENRY_CONSTANTS.get(species, 1.0)
    gas_conc = gas_concs[species]
    C_eq = H * molecules_to_molar(gas_conc)

    row = f'{species:8} {C_eq:>12.2e}'

    for av in av_ratios:
        D_adj = calculate_adjusted_diffusivity(species, H)
        k_mt = (D_adj / MASS_TRANSFER.delta_x_liq) * av

        C_aq = 0.0
        dt = 0.1
        for _ in range(600):
            dC = k_mt * (C_eq - C_aq) * dt
            C_aq += dC

        pct = (C_aq/C_eq*100) if C_eq > 0 else 0
        row += f'{pct:>11.1f}%'

    print(row)

# Summary
print()
print('='*80)
print('Summary: Required A/V to reach 1e-6 M after 60s')
print('='*80)
target = 1e-6

for species in species_test:
    H = HENRY_CONSTANTS.get(species, 1.0)
    gas_conc = gas_concs[species]
    C_eq = H * molecules_to_molar(gas_conc)

    if C_eq < target:
        print(f'{species}: C_eq = {C_eq:.2e} < target. Need higher gas concentration.')
    else:
        ratio = target / C_eq
        if ratio >= 1:
            print(f'{species}: Already at/above target with C_eq = {C_eq:.2e}')
        else:
            k_mt_needed = -np.log(1 - ratio) / 60
            D_adj = calculate_adjusted_diffusivity(species, H)
            av_needed = k_mt_needed * MASS_TRANSFER.delta_x_liq / D_adj
            depth_needed = 1000 / av_needed
            print(f'{species}: Need A/V >= {av_needed:.0f} m^-1 (depth <= {depth_needed:.3f} mm)')
