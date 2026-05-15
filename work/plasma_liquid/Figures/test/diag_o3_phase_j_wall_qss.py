#!/usr/bin/env python3
"""Phase J: Problem 3 wall value vs QSS prediction.

User analysis (2026-05-06): 4mm value is physical (delayed diffusion × chemistry),
but wall values (0.2-1.3mm at 1e-22) need verification — atol band noise vs real
deep sink.

Procedure:
  1. Load 3.6 kV baseline (3 voltages: 2.6, 3.2, 3.6) at t=480s
  2. Compute NO2⁻(z) profile (via HONO_total speciation)
  3. λ(z) = sqrt(D_O3 / (k_R32 × NO2⁻(z)))
  4. Cumulative ∫₀^z dz'/λ(z') via trapezoid rule
  5. QSS prediction: c(z) ≈ c_surf × exp(-∫₀^z dz'/λ(z'))
  6. Compare sim vs QSS, plot both linear and log scale

Output:
  fig_diag_o3_j.{png,pdf}: 3-panel figure (linear, log, ∫dz/λ)
  diag_o3_j.txt: numerical comparison at key cells
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))

from config_1d import ACID_BASE_PAIRS, AQUEOUS_SPECIES, LIQUID_DIFFUSIVITY  # noqa: E402

VOLTAGES = ("2.6kV", "3.2kV", "3.6kV")
COLORS = {"2.6kV": "#d62728", "3.2kV": "#1f77b4", "3.6kV": "#2ca02c"}
K_R32 = 5.0e5  # M⁻¹s⁻¹
D_O3 = LIQUID_DIFFUSIVITY.get("O3", 1.75e-9)
PKA_HONO = ACID_BASE_PAIRS["HONO_total"][2]
KA_HONO = 10.0 ** (-PKA_HONO)


def load_at_t(voltage, t_target=480.0):
    fp = (_root / "Figures" / "DIW results"
          / f"{voltage}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    z_mm = np.asarray(d["z_centers"]) * 1e3

    si = int(np.argmin(np.abs(snap_t - t_target)))
    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")

    o3 = snap_y[si, :, iO3]
    hono_t = snap_y[si, :, iHONO_t]
    hp = np.maximum(snap_y[si, :, iHp], 1e-14)
    no2m = hono_t * KA_HONO / (hp + KA_HONO)
    return {
        "voltage": voltage, "t": float(snap_t[si]),
        "z_mm": z_mm, "o3": o3, "no2m": no2m, "hp": hp,
    }


def compute_qss(z_mm, no2m, c_surf):
    """Compute QSS reactive-penetration prediction.
    λ(z) = sqrt(D_O3 / (k_R32 × NO2⁻(z)))
    c(z) = c_surf × exp(-∫₀^z dz'/λ(z'))
    """
    z_m = z_mm * 1e-3
    no2m_clip = np.maximum(no2m, 1e-30)
    lam = np.sqrt(D_O3 / (K_R32 * no2m_clip))  # m
    integrand = 1.0 / lam  # 1/m
    # trapezoid cumulative
    dz = np.diff(z_m)
    integ = np.zeros_like(z_m)
    for i in range(1, len(z_m)):
        integ[i] = integ[i-1] + 0.5 * (integrand[i-1] + integrand[i]) * dz[i-1]
    c_qss = c_surf * np.exp(-integ)
    return lam, integ, c_qss


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    print("=" * 80)
    print("Phase J: wall vs QSS prediction at t=480s")
    print("=" * 80)
    summ = ["Phase J wall vs QSS prediction (t=480s):", ""]

    for ci, V in enumerate(VOLTAGES):
        d = load_at_t(V, t_target=480.0)
        c_surf = d["o3"][0]
        lam, integ, c_qss = compute_qss(d["z_mm"], d["no2m"], c_surf)

        z = d["z_mm"]
        o3 = d["o3"]

        # (top) log scale
        ax = axes[0, ci]
        ax.plot(z, np.maximum(o3, 1e-40), color=COLORS[V], lw=1.6,
                marker="o", ms=3, label=f"sim {V}")
        ax.plot(z, np.maximum(c_qss, 1e-40), "k--", lw=1.2,
                label="QSS prediction")
        ax.axhline(1e-15, color="orange", ls=":", lw=1.0,
                   label="atol=1e-15")
        ax.set_yscale("log")
        ax.set_ylim(1e-30, 1e-3)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("[O₃] (M)")
        ax.set_title(f"({chr(97+ci)}) {V} log scale @ t=480s",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

        # (bottom) linear scale, restricted range
        ax = axes[1, ci]
        # Use linear y-axis but clip to top ~10% of c_surf
        ax.plot(z, o3 * 1e6, color=COLORS[V], lw=1.6, marker="o", ms=3,
                label=f"sim {V}")
        ax.plot(z, c_qss * 1e6, "k--", lw=1.2, label="QSS")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("[O₃] (µM)")
        ax.set_title(f"({chr(100+ci)}) {V} linear scale @ t=480s",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.5)  # zoom on wall region

        # Summary text
        print(f"\n--- {V} t=480s ---")
        line0 = (f"  c_surf = {c_surf:.3e} M, ∫₀^L dz/λ = {integ[-1]:.2f}")
        print(line0)
        summ.append(f"\n--- {V} t=480s ---")
        summ.append(line0)
        header = f"{'z(mm)':>7s} {'sim O3':>12s} {'QSS O3':>12s} {'NO2-':>11s} {'λ(µm)':>8s} {'∫dz/λ':>8s}"
        print(header)
        summ.append(header)
        for zt in [0.003, 0.04, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 2.3, 4.0, 7.0, 9.8]:
            j = int(np.argmin(np.abs(z - zt)))
            line = (f"{z[j]:>7.3f} {o3[j]:>+12.3e} {c_qss[j]:>+12.3e} "
                    f"{d['no2m'][j]:>+11.3e} {lam[j]*1e6:>8.2f} "
                    f"{integ[j]:>8.2f}")
            print(line)
            summ.append(line)

    fig.suptitle("Phase J: simulation O₃ vs QSS reactive-penetration prediction",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_j.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"\nsaved: {p}")
    (out / "diag_o3_j.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
