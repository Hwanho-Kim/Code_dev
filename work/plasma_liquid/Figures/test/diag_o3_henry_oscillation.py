#!/usr/bin/env python3
"""Phase A: O3 oscillation root-cause diagnosis (3 voltages comparison).

Hypothesis: 2.6 kV bulk O3 oscillation appears at Henry equilibrium reaching
point (MT flux convergence). 3.2/3.6 kV stay smooth despite same gas input
SG smoothing → input smoothing is NOT the cause. Real cause must be tied
to chemistry sink strength (R32 dependency on NO2-).

Output:
  - fig_diag_o3_oscillation.{png,pdf}  (4×3 panel grid)
  - Console: Henry-eq reaching times, oscillation period, RMS amplitudes
  - diag_o3_oscillation.txt  (numerical summary)

Reads existing HONOvar caches; no re-simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from config_1d import ACID_BASE_PAIRS, AQUEOUS_SPECIES, HENRY_CONSTANTS  # noqa: E402

VOLTAGES = ("2.6kV", "3.2kV", "3.6kV")
COLORS = {"2.6kV": "#d62728", "3.2kV": "#1f77b4", "3.6kV": "#2ca02c"}
CACHE_FMT = (
    "{root}/Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar/"
    "cache/three_film_abspecies_dg0.0100.npz"
)

DRIVING_THRESHOLD = 0.10  # |dC|/C_eq below this = "Henry eq reached"
PROBE_DEPTHS_MM = (0.0, 0.1, 1.0, 5.0)


def load_cache(voltage: str) -> dict:
    fp = CACHE_FMT.format(root=_root, V=voltage)
    d = dict(np.load(fp, allow_pickle=True))
    return d


def build_solver_for_voltage(voltage: str):
    """Set up gen_all_figures.py state to instantiate solver for given voltage."""
    import gen_all_figures as gaf  # type: ignore

    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[voltage]

    times, gas_conc = gaf.load_gas_data()
    solver = gaf._get_solver(times, gas_conc)
    return solver, times, gas_conc


def cell_at_depth(z_centers_mm: np.ndarray, target_mm: float) -> int:
    return int(np.argmin(np.abs(z_centers_mm - target_mm)))


def bulk_only_avg(snap_y: np.ndarray, dz: np.ndarray, z_mm: np.ndarray,
                  sp_idx: int, surface_skip_mm: float = 0.1) -> np.ndarray:
    """Average over cells with z > surface_skip_mm."""
    mask = z_mm > surface_skip_mm
    dz_b = dz[mask]
    L_b = float(dz_b.sum())
    return np.array(
        [np.dot(snap_y[k, mask, sp_idx], dz_b) / L_b for k in range(len(snap_y))]
    )


def vol_weighted_avg(snap_y: np.ndarray, dz: np.ndarray, sp_idx: int,
                     L: float) -> np.ndarray:
    return np.array(
        [np.dot(snap_y[k, :, sp_idx], dz) / L for k in range(len(snap_y))]
    )


def compute_o3_mt_diagnostics(voltage: str) -> dict:
    """Returns dict with all O3-related time series for the given voltage."""
    cache = load_cache(voltage)
    snap_t = np.asarray(cache["snap_t"])
    snap_y = np.asarray(cache["snap_y"])
    dz = np.asarray(cache["dz_cells"])
    L = float(cache["L"])
    z_mm = np.asarray(cache["z_centers"]) * 1e3

    sp_idx_map = {sp: i for i, sp in enumerate(AQUEOUS_SPECIES)}
    iO3 = sp_idx_map["O3"]
    iHONO_t = sp_idx_map.get("HONO_total", -1)  # NO2- lives in HONO_total pool
    iHp = sp_idx_map["H+"]
    pKa_hono = ACID_BASE_PAIRS["HONO_total"][2]
    Ka_hono = 10.0 ** (-pKa_hono)

    # Solver-side: C_eq(t), k_mt(O3)
    solver, times, gas_conc = build_solver_for_voltage(voltage)
    iface = solver._interface_species  # [(aq_idx, k_mt, gas_sp, H, Ka), ...]
    k_mt_O3 = None
    H_O3 = HENRY_CONSTANTS.get("O3", 1.0)
    aq_O3_idx = solver.species_idx["O3"]
    for aq_idx, k_mt, gas_sp, H, Ka in iface:
        if gas_sp == "O3":
            k_mt_O3 = k_mt
            break
    if k_mt_O3 is None:
        raise RuntimeError("O3 not in interface species")

    C_eq_t = np.array([solver._get_C_eq_fast("O3", t) for t in snap_t])

    # Surface, depth-resolved, bulk-only, vol-weighted
    surf = snap_y[:, 0, iO3]
    depth_series = {
        d: snap_y[:, cell_at_depth(z_mm, d), iO3] for d in PROBE_DEPTHS_MM
    }
    bulk_avg = bulk_only_avg(snap_y, dz, z_mm, iO3, surface_skip_mm=0.1)
    vol_avg = vol_weighted_avg(snap_y, dz, iO3, L)

    # MT flux (M/s): k_mt × (C_eq − c_surf) / L  (no Ka for O3)
    mt_flux = k_mt_O3 * (C_eq_t - surf) / L

    # Driving force
    drive = (C_eq_t - surf) / np.maximum(C_eq_t, 1e-30)

    # NO2- via speciation HONO_total × Ka/(H+ + Ka), bulk-only (z>0.1mm)
    if iHONO_t >= 0:
        mask_b = z_mm > 0.1
        dz_b = dz[mask_b]
        L_b = float(dz_b.sum())
        hono_t_avg = np.array(
            [np.dot(snap_y[k, mask_b, iHONO_t], dz_b) / L_b
             for k in range(len(snap_t))]
        )
        hp_avg = np.array(
            [np.dot(snap_y[k, mask_b, iHp], dz_b) / L_b
             for k in range(len(snap_t))]
        )
        hp_avg = np.maximum(hp_avg, 1e-14)
        no2m_avg = hono_t_avg * Ka_hono / (hp_avg + Ka_hono)
    else:
        no2m_avg = np.zeros_like(snap_t)

    # "Transient-end time" — driving force minimum (where chemistry-MT
    # coupling switches from buildup to dissipation).
    # Smoothed to avoid single-step glitches. NB: this is NOT Henry equilibrium
    # since drive remains > 0 throughout (PDE liquid-side resistance dominant).
    win_smooth = min(30, len(drive) // 4)
    if win_smooth >= 3:
        kernel = np.ones(win_smooth) / win_smooth
        drive_smooth = np.convolve(drive, kernel, mode="same")
    else:
        drive_smooth = drive
    # Skip first 5s of warmup
    skip = max(int(5 / max(np.median(np.diff(snap_t)), 1)), 1)
    he_idx = int(skip + np.argmin(drive_smooth[skip:]))

    return {
        "voltage": voltage,
        "snap_t": snap_t,
        "z_mm": z_mm,
        "surface": surf,
        "depth": depth_series,
        "bulk_avg": bulk_avg,
        "vol_avg": vol_avg,
        "C_eq": C_eq_t,
        "drive": drive,
        "mt_flux": mt_flux,
        "no2m_avg": no2m_avg,
        "k_mt_O3": k_mt_O3,
        "H_O3": H_O3,
        "he_idx": he_idx,
        "he_time": float(snap_t[he_idx]) if he_idx >= 0 else float("nan"),
    }


def fft_post_henry(t: np.ndarray, y: np.ndarray, t_start: float
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """FFT of y[t > t_start], linear-detrended. Period in seconds."""
    if not np.isfinite(t_start) or t_start >= t[-1] - 60:
        return np.array([]), np.array([]), float("nan")
    mask = t > t_start
    yy = y[mask].astype(float)
    tt = t[mask].astype(float)
    if len(yy) < 16:
        return np.array([]), np.array([]), float("nan")
    # Linear detrend (drift removal)
    p = np.polyfit(tt, yy, 1)
    yy = yy - np.polyval(p, tt)
    dt = float(np.median(np.diff(tt)))
    Y = np.fft.rfft(yy)
    freqs = np.fft.rfftfreq(len(yy), d=dt)
    power = np.abs(Y) ** 2
    if len(freqs) > 1:
        f_dom = freqs[1 + np.argmax(power[1:])]
        period = 1.0 / f_dom if f_dom > 0 else float("nan")
    else:
        period = float("nan")
    return freqs, power, period


def make_plot(diags: dict[str, dict]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=False)
    plt.rcParams.update({"font.family": "serif"})

    for ci, V in enumerate(VOLTAGES):
        d = diags[V]
        t_min = d["snap_t"] / 60.0
        col = COLORS[V]

        # Row 0: Surface, bulk-only avg, vol-avg, C_eq overlay
        ax = axes[0, ci]
        ax.plot(t_min, d["surface"] * 1e9, color="#999", lw=1.0,
                label="surface (z=0)")
        ax.plot(t_min, d["bulk_avg"] * 1e9, color=col, lw=2.0,
                label="bulk-only avg (z>0.1mm)")
        ax.plot(t_min, d["vol_avg"] * 1e9, color=col, lw=1.0, ls="--",
                label="vol-weighted avg")
        ax.plot(t_min, d["C_eq"] * 1e9, color="k", lw=0.8, ls=":", label="C_eq(t)")
        if np.isfinite(d["he_time"]):
            ax.axvline(d["he_time"] / 60, color="orange", lw=1, ls="-.")
            ax.text(d["he_time"] / 60 + 0.1, ax.get_ylim()[1] * 0.9,
                    f"Henry eq\nt={d['he_time']:.0f}s",
                    fontsize=8, color="orange")
        ax.set_ylabel("[O₃] (nM)")
        ax.set_title(f"{V}  Surface/Bulk/Vol-avg O₃", fontweight="bold", fontsize=11)
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t_min[-1])

        # Row 1: O3 at probed depths
        ax = axes[1, ci]
        depth_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(PROBE_DEPTHS_MM)))
        for di, dmm in enumerate(PROBE_DEPTHS_MM):
            arr = np.where(d["depth"][dmm] > 0, d["depth"][dmm], np.nan)
            ax.plot(t_min, arr * 1e9, color=depth_colors[di], lw=1.3,
                    label=f"z={dmm:g}mm")
        ax.set_ylabel("[O₃] (nM)")
        ax.set_yscale("log")
        ax.set_title(f"{V}  Depth-resolved [O₃]", fontweight="bold", fontsize=11)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, t_min[-1])

        # Row 2: Driving force
        ax = axes[2, ci]
        ax.plot(t_min, d["drive"], color=col, lw=1.5)
        ax.axhline(DRIVING_THRESHOLD, color="orange", lw=0.8, ls="--",
                   label=f"|drive|<{DRIVING_THRESHOLD}")
        ax.axhline(-DRIVING_THRESHOLD, color="orange", lw=0.8, ls="--")
        if np.isfinite(d["he_time"]):
            ax.axvline(d["he_time"] / 60, color="orange", lw=1, ls="-.")
        ax.set_ylabel("(C_eq − c_surf)/C_eq")
        ax.set_title(f"{V}  Driving force", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t_min[-1])

        # Row 3: MT flux + FFT inset
        ax = axes[3, ci]
        ax.plot(t_min, d["mt_flux"], color=col, lw=1.5)
        ax.axhline(0, color="k", lw=0.5)
        if np.isfinite(d["he_time"]):
            ax.axvline(d["he_time"] / 60, color="orange", lw=1, ls="-.")
        ax.set_ylabel("MT flux (M/s)")
        ax.set_xlabel("Time (min)")
        ax.set_title(f"{V}  MT flux (volumetric)", fontweight="bold", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t_min[-1])

        # FFT inset (post-Henry-eq)
        freqs, power, period = fft_post_henry(d["snap_t"], d["bulk_avg"],
                                              d["he_time"])
        if len(freqs) > 0:
            ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
            ax_in.semilogy(freqs[1:] * 60, power[1:], color=col, lw=1.0)
            ax_in.set_xlabel("Freq (1/min)", fontsize=7)
            ax_in.set_ylabel("Power", fontsize=7)
            ax_in.set_title(f"FFT bulk O₃ (post-HE)\nT_dom={period:.1f}s",
                            fontsize=7)
            ax_in.tick_params(labelsize=6)
            ax_in.grid(alpha=0.3)

    fig.suptitle(
        "Phase A diagnosis — O₃ Henry-equilibrium oscillation across voltages "
        "(DIW, three_film, HONOvar)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out_dir = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_diag_o3_oscillation.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")


def write_summary(diags: dict[str, dict]) -> None:
    out = Path(__file__).parent / "diag_o3_oscillation.txt"
    lines = ["Phase A: O3 Henry-eq oscillation diagnosis", "=" * 70, ""]
    for V in VOLTAGES:
        d = diags[V]
        t = d["snap_t"]
        bulk = d["bulk_avg"]
        no2m = d["no2m_avg"]
        # Post-HE statistics
        if np.isfinite(d["he_time"]):
            mask_post = t > d["he_time"]
            bulk_post = bulk[mask_post]
            mean_b = float(np.mean(bulk_post))
            std_b = float(np.std(bulk_post))
            cv = (std_b / mean_b * 100) if mean_b > 0 else float("nan")
        else:
            mean_b = std_b = cv = float("nan")
        # FFT period
        _, _, period = fft_post_henry(t, d["bulk_avg"], d["he_time"])
        # NO2- final
        no2m_final = float(no2m[-1] * 1e6)
        # k_mt
        kmt = d["k_mt_O3"]
        # R32 effective rate at final = k×O3_avg×NO2m_avg
        r32 = 5.0e5 * float(bulk[-1]) * float(no2m[-1])
        lines.append(f"--- {V} ---")
        lines.append(f"  Henry-eq reaching time:  t = {d['he_time']:.1f} s")
        lines.append(f"  k_mt (O3, three_film):   {kmt:.3e} m/s")
        lines.append(f"  H_cc (O3):               {d['H_O3']:.3e}")
        lines.append(f"  Bulk-only avg [O3] post-HE:")
        lines.append(f"     mean = {mean_b*1e9:.3f} nM, "
                     f"std = {std_b*1e9:.3f} nM, CV = {cv:.2f}%")
        lines.append(f"  Dominant osc period (post-HE FFT): {period:.1f} s")
        lines.append(f"  NO2- final (vol avg):  {no2m_final:.4f} µM")
        lines.append(f"  R32 effective rate at final:  {r32:.3e} M/s")
        # τ_chem = 1 / (k×NO2-), τ_MT = L/k_mt
        if no2m[-1] > 0:
            tau_chem = 1.0 / (5.0e5 * no2m[-1])
            lines.append(f"  τ_chem (1/(k_R32·[NO2-])):  {tau_chem:.2e} s")
        tau_mt = 0.01 / kmt
        lines.append(f"  τ_MT (L/k_mt):              {tau_mt:.2e} s")
        lines.append("")
    txt = "\n".join(lines)
    out.write_text(txt)
    print(txt)
    print(f"\nsaved: {out}")


def main() -> None:
    diags = {V: compute_o3_mt_diagnostics(V) for V in VOLTAGES}
    make_plot(diags)
    write_summary(diags)


if __name__ == "__main__":
    main()
