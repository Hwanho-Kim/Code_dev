"""Verify analytical Jacobian against finite-difference Jacobian.

Usage:
    source ~/work/.venv/bin/activate && cd ~/work
    python -m plasma0d_v2.test_jacobian --config plasma0d_v2/config.yaml
"""

import argparse
import numpy as np
import os
import sys

from .config import load_config, setup_simulation


def fd_jacobian(rhs_func, t, y, eps_rel=1e-6):
    """Compute finite-difference Jacobian."""
    n = len(y)
    J = np.zeros((n, n))
    f0 = rhs_func(t, y)

    for j in range(n):
        y_pert = y.copy()
        h = max(abs(y[j]) * eps_rel, 1e-20)
        y_pert[j] += h
        f_pert = rhs_func(t, y_pert)
        J[:, j] = (f_pert - f0) / h

    return J


def compare_jacobians(J_ana, J_fd, label=""):
    """Compare two Jacobian matrices and print statistics."""
    n = J_ana.shape[0]

    # Element-wise relative error (where FD is nonzero)
    mask = np.abs(J_fd) > 1e-25
    if mask.any():
        rel_err = np.abs(J_ana[mask] - J_fd[mask]) / np.abs(J_fd[mask])
        print(f"\n  {label} Jacobian comparison ({n}x{n}):")
        print(f"    Non-zero FD elements: {mask.sum()}")
        print(f"    Relative error: median={np.median(rel_err):.2e}, "
              f"mean={np.mean(rel_err):.2e}, max={np.max(rel_err):.2e}")
        print(f"    Elements with >10% error: {(rel_err > 0.1).sum()}")
        print(f"    Elements with >100% error: {(rel_err > 1.0).sum()}")

        # Find worst elements
        worst_idx = np.unravel_index(np.argmax(np.abs(J_ana - J_fd)), J_ana.shape)
        print(f"    Worst absolute error at [{worst_idx[0]}, {worst_idx[1]}]: "
              f"ana={J_ana[worst_idx]:.4e}, fd={J_fd[worst_idx]:.4e}")
    else:
        print(f"\n  {label}: No non-zero FD elements found")

    # Block-wise summary
    n_sp = n - 2
    idx_e = n_sp  # ne_eps
    idx_T = n_sp + 1  # T_gas

    blocks = [
        ("Species×Species", slice(0, n_sp), slice(0, n_sp)),
        ("Species vs ne_eps", slice(0, n_sp), slice(idx_e, idx_e+1)),
        ("Species vs T_gas", slice(0, n_sp), slice(idx_T, idx_T+1)),
        ("ne_eps row", slice(idx_e, idx_e+1), slice(0, n)),
        ("T_gas row", slice(idx_T, idx_T+1), slice(0, n)),
    ]

    print(f"\n    Block-wise Frobenius norms:")
    for name, r, c in blocks:
        norm_ana = np.linalg.norm(J_ana[r, c])
        norm_fd = np.linalg.norm(J_fd[r, c])
        norm_diff = np.linalg.norm(J_ana[r, c] - J_fd[r, c])
        rel = norm_diff / max(norm_fd, 1e-30)
        print(f"      {name:25s}: ||ana||={norm_ana:.3e}, ||fd||={norm_fd:.3e}, "
              f"||diff||={norm_diff:.3e}, rel={rel:.3e}")


def main():
    parser = argparse.ArgumentParser(description='Test analytical Jacobian')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)

    print("\n" + "=" * 60)
    print("  JACOBIAN VERIFICATION")
    print("=" * 60)

    # Test at initial conditions (t=0)
    t_test = 0.0
    print(f"\n  Test point: t = {t_test*1e6:.1f} µs (initial conditions)")
    print(f"  State vector size: {len(y0)}")
    print(f"  c_e = {y0[0]:.4e}, ne_eps = {y0[solver.sm.idx_energy]:.4e}, "
          f"T_gas = {y0[solver.sm.idx_Tgas]:.1f}")

    # Analytical Jacobian
    print("\n  Computing analytical Jacobian...")
    J_ana = solver.jacobian(t_test, y0)
    print(f"    Shape: {J_ana.shape}, NaN count: {np.isnan(J_ana).sum()}, "
          f"Inf count: {np.isinf(J_ana).sum()}")

    # FD Jacobian
    print("  Computing FD Jacobian (65 perturbations)...")
    J_fd = fd_jacobian(solver.rhs, t_test, y0, eps_rel=1e-6)
    print(f"    Shape: {J_fd.shape}, NaN count: {np.isnan(J_fd).sum()}, "
          f"Inf count: {np.isinf(J_fd).sum()}")

    compare_jacobians(J_ana, J_fd, label="t=0 (IC)")

    # Detailed element-by-element comparison of the worst block
    n_sp = solver.sm.n_species
    print(f"\n  Detailed: Species×Species block worst elements (top 20):")
    sp_ana = J_ana[:n_sp, :n_sp]
    sp_fd = J_fd[:n_sp, :n_sp]
    abs_diff = np.abs(sp_ana - sp_fd)
    # Filter to elements where FD is nonzero
    fd_mask = np.abs(sp_fd) > 1e-25
    if fd_mask.any():
        indices = np.argwhere(fd_mask)
        errors = []
        for idx in indices:
            i, j = idx
            rel = abs_diff[i, j] / max(abs(sp_fd[i, j]), 1e-50)
            errors.append((i, j, sp_ana[i, j], sp_fd[i, j], rel))
        errors.sort(key=lambda x: -x[4])
        for rank, (i, j, a, f, r) in enumerate(errors[:20]):
            sp_i = solver.sm.names[i] if i < len(solver.sm.names) else f"sp{i}"
            sp_j = solver.sm.names[j] if j < len(solver.sm.names) else f"sp{j}"
            print(f"      [{i:2d},{j:2d}] ({sp_i:8s} vs {sp_j:8s}): "
                  f"ana={a:+12.4e}  fd={f:+12.4e}  rel={r:.3e}")

    # Same for species vs ne_eps
    print(f"\n  Detailed: Species vs ne_eps column (top 20):")
    idx_e = solver.sm.idx_energy
    for i in range(n_sp):
        a = J_ana[i, idx_e]
        f = J_fd[i, idx_e]
        if abs(f) > 1e-25 or abs(a) > 1e-25:
            rel = abs(a - f) / max(abs(f), 1e-50) if abs(f) > 1e-25 else float('inf')
            sp_i = solver.sm.names[i]
            print(f"      [{i:2d},ne_eps] ({sp_i:8s}): "
                  f"ana={a:+12.4e}  fd={f:+12.4e}  rel={rel:.3e}")

    # Test at a slightly perturbed state (simulate elevated electrons)
    y_mid = y0.copy()
    y_mid[0] *= 1e5          # c_e ×100000 (pulse-like)
    y_mid[solver.sm.idx_energy] *= 1e5
    t_mid = 5e-6  # 5 µs (during a pulse)
    print(f"\n  Test point: t = {t_mid*1e6:.1f} µs (elevated n_e)")
    print(f"  c_e = {y_mid[0]:.4e}, ne_eps = {y_mid[solver.sm.idx_energy]:.4e}")

    J_ana2 = solver.jacobian(t_mid, y_mid)
    J_fd2 = fd_jacobian(solver.rhs, t_mid, y_mid, eps_rel=1e-6)
    compare_jacobians(J_ana2, J_fd2, label="t=5µs (elevated)")

    print(f"\n  Verification complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
