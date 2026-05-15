#!/usr/bin/env python3
"""Verify two specific claims from the V-shape discussion using the CSV
already dumped by diag_o3_param_dump.py:

  (1) "Sink is strongest at surface, not in mid" -> compute R32 rate per
      depth at multiple times.
  (2) "Surface O3 stays high because BC supply balances sink at cell 0"
      -> tabulate surface-cell mass balance: chem_net, div_diff,
      dCdt_actual, residual. residual is the surface MT supply (since
      the dump excluded MT term in chem_net).

  Also dump NO2-(z) at each snapshot to check whether the profile is
  really 'flat from surface to ~1 mm' as the user observed visually.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CSV = (Path(__file__).resolve().parent
       / 'diag_o3_param_dump_3.6kV_HONOvar.csv')
K_R32 = 5.0e5


def main():
    df = pd.read_csv(CSV)
    df['R32_rate'] = -K_R32 * df['C_O3'] * df['C_NO2-']

    print(f'rows={len(df)}, times={sorted(df["t_min"].unique())}')

    # ------------------------------------------------------------------
    # Claim 1 verification: NO2- profile flat surface->1mm? Dump NO2- per z.
    # ------------------------------------------------------------------
    print('\n=== NO2-(z) profile at each snapshot (M) ===')
    print(f'{"z(mm)":>7s}  ' + '  '.join(
        f'{f"t={t:.1f}m":>10s}' for t in sorted(df["t_min"].unique())))
    cells_show = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    z_by_cell = df.groupby('cell_idx')['z_mm'].first()
    for j in cells_show:
        row = []
        for t in sorted(df['t_min'].unique()):
            v = df[(df['cell_idx'] == j) & (df['t_min'] == t)]['C_NO2-'].iloc[0]
            row.append(f'{v:>10.2e}')
        print(f'{z_by_cell[j]:>7.3f}  ' + '  '.join(row))

    # ------------------------------------------------------------------
    # Claim 1: R32 sink rate per depth (k * O3 * NO2-) — where is it actually
    # strongest?
    # ------------------------------------------------------------------
    print('\n=== R32 sink rate (M/s) per depth at each snapshot ===')
    print(f'{"z(mm)":>7s}  ' + '  '.join(
        f'{f"t={t:.1f}m":>11s}' for t in sorted(df["t_min"].unique())))
    for j in cells_show:
        row = []
        for t in sorted(df['t_min'].unique()):
            v = df[(df['cell_idx'] == j) & (df['t_min'] == t)]['R32_rate'].iloc[0]
            row.append(f'{abs(v):>11.2e}')
        print(f'{z_by_cell[j]:>7.3f}  ' + '  '.join(row))

    # ------------------------------------------------------------------
    # Where is the per-time peak sink located?
    # ------------------------------------------------------------------
    print('\n=== Cell with maximum |R32_rate| at each snapshot ===')
    for t in sorted(df['t_min'].unique()):
        sub = df[df['t_min'] == t].copy()
        idx_max = sub['R32_rate'].abs().idxmax()
        row = sub.loc[idx_max]
        print(f't={t:5.1f}m   max |R32| at z={row["z_mm"]:6.3f}mm  '
              f'(cell {int(row["cell_idx"]):2d})   '
              f'rate={abs(row["R32_rate"]):.3e}   '
              f'O3={row["C_O3"]:.2e}   NO2-={row["C_NO2-"]:.2e}')

    # ------------------------------------------------------------------
    # Claim 2 verification: surface (cell 0) mass balance.
    # The dump did NOT include the MT (gas->liquid) flux in chem_net,
    # so for cell 0 the residual = dCdt - chem_net - div_diff is exactly
    # the MT supply per cell (M/s).
    # ------------------------------------------------------------------
    print('\n=== Surface cell (j=0) mass balance ===')
    print(f'{"t(min)":>8s}  {"C_O3":>10s}  {"C_NO2-":>10s}  '
          f'{"chem_net":>11s}  {"div_diff":>11s}  '
          f'{"dCdt":>11s}  {"residual=MT":>12s}  '
          f'{"|chem|/|MT|":>11s}')
    surf = df[df['cell_idx'] == 0].sort_values('t_min')
    for _, r in surf.iterrows():
        ratio_chem_mt = (abs(r['chem_net']) /
                         max(abs(r['residual']), 1e-30))
        print(f'{r["t_min"]:>8.1f}  {r["C_O3"]:>10.2e}  '
              f'{r["C_NO2-"]:>10.2e}  '
              f'{r["chem_net"]:>+11.2e}  {r["div_diff"]:>+11.2e}  '
              f'{r["dCdt_actual"]:>+11.2e}  {r["residual"]:>+12.2e}  '
              f'{ratio_chem_mt:>11.2e}')

    # ------------------------------------------------------------------
    # Compare R32 contribution on cell 0 (single reaction) vs total chem_net
    # vs MT supply. If R32 is the dominant chemistry sink, then
    # |R32_rate| ~ |chem_net| at cell 0.
    # ------------------------------------------------------------------
    print('\n=== Cell 0: R32 rate vs total chem_net vs MT supply ===')
    print(f'{"t(min)":>8s}  {"R32_rate":>11s}  {"chem_net":>11s}  '
          f'{"R32/chem":>10s}  {"MT (resid)":>12s}  {"R32/MT":>10s}')
    for _, r in surf.iterrows():
        r32 = r['R32_rate']
        cn = r['chem_net']
        mt = r['residual']
        print(f'{r["t_min"]:>8.1f}  {r32:>+11.2e}  {cn:>+11.2e}  '
              f'{abs(r32)/max(abs(cn),1e-30):>10.2e}  '
              f'{mt:>+12.2e}  '
              f'{abs(r32)/max(abs(mt),1e-30):>10.2e}')

    # ------------------------------------------------------------------
    # Why is c_O3(surface) finite if R32 is so strong? Direct check:
    # the *hypothetical* steady-state surface concentration if MT_in is
    # given and only R32 sink acts at cell 0:
    # 0 = MT_in/dz0 - k_R32 * NO2- * c_O3   ==>   c_O3_ss = MT_in /
    # (dz0 * k_R32 * NO2-)
    # Use cell-0 dz0 and per-cell rate (M/s). Use abs(MT) = abs(residual).
    # ------------------------------------------------------------------
    print('\n=== Steady-state c_O3 at cell 0 from MT vs R32 only ===')
    dz0 = 5e-6  # production grid surface cell width
    print(f'{"t(min)":>8s}  {"|MT|":>11s}  {"NO2-":>11s}  '
          f'{"c_ss = MT/(k*NO2-)":>22s}  {"actual c_O3":>13s}  '
          f'{"ratio":>9s}')
    for _, r in surf.iterrows():
        mt_abs = abs(r['residual'])
        no2 = r['C_NO2-']
        if no2 > 1e-30 and mt_abs > 1e-30:
            c_ss = mt_abs / (K_R32 * no2)
            ratio = r['C_O3'] / c_ss if c_ss > 0 else 0.0
        else:
            c_ss = 0.0
            ratio = 0.0
        print(f'{r["t_min"]:>8.1f}  {mt_abs:>11.2e}  {no2:>11.2e}  '
              f'{c_ss:>22.2e}  {r["C_O3"]:>13.2e}  {ratio:>9.2e}')


if __name__ == '__main__':
    main()
