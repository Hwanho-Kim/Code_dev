#!/usr/bin/env python3
"""Phase E: cell-40 dydt decomposition over time.

Identify mechanism keeping O3 at z≈4mm at 1.83e-10 M in 3.6 kV baseline.

For each snapshot:
  - Chemistry rates per reaction at cell 40 (top contributors)
  - SG transport flux at cell 40 (in from cell 39, out to cell 41)
  - Net dC/dt vs ΔC/Δt finite difference
  - Find which reactions dominate O3 production/consumption
"""
from __future__ import annotations

import functools
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "3.6kV"
TARGET_CELL = 40  # z ≈ 4.07 mm
TARGET_SP = "O3"


def main():
    import gen_all_figures as gaf

    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = VOLTAGE
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[VOLTAGE]
    times, gas_conc = gaf.load_gas_data()
    solver = gaf._get_solver(times, gas_conc)
    chem = solver.chem

    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])  # (nt, N_z, N_s)
    dz = np.asarray(d["dz_cells"])
    z_mm = np.asarray(d["z_centers"]) * 1e3

    iO3 = AQUEOUS_SPECIES.index("O3")
    iHp = chem.species_idx["H+"]
    j = TARGET_CELL
    print(f"Target cell {j} at z={z_mm[j]:.3f} mm")
    print(f"O3 trajectory: t=60→ {snap_y[30, j, iO3]:.3e}, "
          f"t=480→ {snap_y[240, j, iO3]:.3e}")

    nt = len(snap_t)
    n_rxn = len(chem.reactions)

    # Per-reaction rates at cell j (and neighbors j-1, j+1 for transport)
    rates_O3 = defaultdict(lambda: np.zeros(nt))  # rxn label → rate(t) at cell j
    transport_in = np.zeros(nt)   # flux from cell j-1 → cell j
    transport_out = np.zeros(nt)  # flux from cell j → cell j+1
    chem_total = np.zeros(nt)     # net chemistry dC/dt at cell j

    D_O3 = float(solver.D_species[iO3])
    inv_h = solver.inv_h_faces  # face inverse distances
    inv_dz_j = solver.inv_dz_cells[j]

    for t_idx in range(nt):
        y_cell = snap_y[t_idx, j].copy()
        y_cell = np.clip(y_cell, chem.trace, 1.0)
        y_cell[iHp] = max(y_cell[iHp], 1e-14)
        spec = chem.speciate(y_cell)
        for ri, rxn_d in enumerate(chem._rxn_data):
            r = chem._compute_single_rate(rxn_d, y_cell, spec)
            rxn = chem.reactions[ri]
            label = rxn.get("label", f"R{ri}")
            # Net effect on O3:
            n_consume = rxn["reactants"].get("O3", 0)
            n_produce = rxn.get("products", {}).get("O3", 0)
            net = n_produce - n_consume
            if net != 0:
                rates_O3[label][t_idx] += net * r
                chem_total[t_idx] += net * r

        # SG transport at cell j (α=0 for neutral O3): J = D/h × (c_j - c_{j+1})
        c_jm1 = snap_y[t_idx, j-1, iO3]
        c_j = snap_y[t_idx, j, iO3]
        c_jp1 = snap_y[t_idx, j+1, iO3] if j+1 < snap_y.shape[1] else 0.0
        J_in = D_O3 * inv_h[j-1] * (c_jm1 - c_j)        # face j-1/2
        J_out = D_O3 * inv_h[j] * (c_j - c_jp1)          # face j+1/2
        transport_in[t_idx] = J_in * inv_dz_j           # flux/dz contribution to dC/dt
        transport_out[t_idx] = -J_out * inv_dz_j

    # Net transport contribution to dC/dt = -(J_out - J_in)/dz_j = transport_in + transport_out
    transport_net = transport_in + transport_out

    # Finite-difference ΔC/Δt
    o3_t = snap_y[:, j, iO3]
    dt_snap = np.diff(snap_t)
    dC_dt_fd = np.zeros(nt)
    if nt > 1:
        diffs = np.diff(o3_t) / dt_snap
        dC_dt_fd[0] = diffs[0]
        dC_dt_fd[-1] = diffs[-1]
        dC_dt_fd[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])

    # Identify top reactions by max |rate|
    rxn_peak = {label: float(np.max(np.abs(arr))) for label, arr in rates_O3.items()}
    top_rxns = sorted(rxn_peak.items(), key=lambda x: -x[1])[:8]
    print(f"\nTop reactions affecting O3 at cell {j}:")
    for label, peak in top_rxns:
        print(f"  {label}: peak |rate| = {peak:.3e} M/s")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    t_min = snap_t / 60.0

    # (a) O3 concentration at cell j
    ax = axes[0]
    ax.plot(t_min, np.maximum(o3_t, 1e-30), color="#d62728", lw=2.0,
            label=f"O3 cell {j} (z={z_mm[j]:.2f}mm)")
    ax.set_yscale("log")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title(f"(a) [O₃] time series at z={z_mm[j]:.2f}mm",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # (b) dC/dt breakdown (chemistry + transport + net)
    ax = axes[1]
    ax.plot(t_min, transport_in, color="#1f77b4", lw=1.4, label="diff in (j-1→j)")
    ax.plot(t_min, transport_out, color="#9467bd", lw=1.4, label="diff out (j→j+1)")
    ax.plot(t_min, transport_net, color="#1f77b4", ls="--", lw=1.6,
            label="diff net")
    ax.plot(t_min, chem_total, color="#2ca02c", lw=1.6, label="chemistry net")
    ax.plot(t_min, transport_net + chem_total, color="k", lw=1.8,
            label="rxn+diff total")
    ax.plot(t_min, dC_dt_fd, color="orange", ls=":", lw=2.0,
            label="ΔC/Δt (finite-diff)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("dC/dt (M/s)")
    ax.set_title("(b) dC/dt budget at cell j", fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # (c) Top reactions
    ax = axes[2]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(top_rxns)))
    for ci, (label, _) in enumerate(top_rxns):
        ax.plot(t_min, rates_O3[label], color=cmap[ci], lw=1.3,
                label=label[:42])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Reaction rate × stoich (M/s)")
    ax.set_xlabel("Time (min)")
    ax.set_title(f"(c) Per-reaction O₃ source/sink at cell {j}",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"Phase E ({VOLTAGE}): cell {j} O₃ dC/dt budget — "
        "what makes 4mm peak persist?",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_e_cell{j}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary at key time points
    print("\n=== dC/dt budget at cell {} (z={:.2f}mm) ===".format(j, z_mm[j]))
    print(f"{'t(s)':>5} {'O3':>11} {'diff_in':>11} {'diff_out':>11} {'diff_net':>11} "
          f"{'chem_net':>11} {'rxn+diff':>11} {'ΔC/Δt':>11}")
    for tt in [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
        i = int(np.argmin(np.abs(snap_t - tt)))
        print(f"{snap_t[i]:>5.0f} {o3_t[i]:>11.3e} {transport_in[i]:>+11.3e} "
              f"{transport_out[i]:>+11.3e} {transport_net[i]:>+11.3e} "
              f"{chem_total[i]:>+11.3e} {transport_net[i]+chem_total[i]:>+11.3e} "
              f"{dC_dt_fd[i]:>+11.3e}")


if __name__ == "__main__":
    main()
