#!/usr/bin/env python3
"""Phase K6: Bernoulli function precision.

K5 already showed SG = FD exactly when E_half=0, which means B(0)=1 is
correctly handled. This phase further inspects the Bernoulli function
at small but non-zero alpha (where Taylor cancellation could matter)
and at extreme alpha (where exp() overflow could matter), to verify
the implementation is robust if Poisson is ever turned on.

We compare the project's _bernoulli against:
  (a) double-precision exact:  alpha / (exp(alpha) - 1)
  (b) Taylor series for |alpha|<small_threshold
"""
import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D


def bernoulli_exact(alpha):
    """Direct double-precision evaluation. Diverges as alpha -> 0 due
    to cancellation; safe only for |alpha| > ~1e-8."""
    a = np.asarray(alpha, dtype=float)
    out = np.where(np.abs(a) < 1e-300, 1.0, a / np.expm1(a))
    return out


def bernoulli_taylor(alpha):
    """Taylor: B(a) = 1 - a/2 + a^2/12 - a^4/720 + ... (Bernoulli numbers).
    Accurate for |a| << 1."""
    a = np.asarray(alpha, dtype=float)
    a2 = a * a
    a4 = a2 * a2
    return 1.0 - a / 2.0 + a2 / 12.0 - a4 / 720.0


def main():
    print('Phase K6: Bernoulli function precision in SG flux')

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, fixed_cation_conc=0.0,
        bc_type='three_film', alpha_b=None, delta_gas=0.01,
    )

    # Test grid of alpha values from numerical zero to large
    alphas = np.array([
        0.0, 1e-20, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        1e-1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 300.0, 500.0,
        -1e-15, -1e-8, -1e-2, -1.0, -10.0, -100.0,
    ])

    print(f'\n{"alpha":>12s}  {"_bernoulli":>15s}  {"exact":>15s}  '
          f'{"taylor":>15s}  {"|impl-exact|":>15s}')

    bproj = solver._bernoulli(alphas)
    for i, a in enumerate(alphas):
        if abs(a) < 1e-8:
            ref = bernoulli_taylor(a)
            ref_label = 'taylor'
        else:
            ref = float(bernoulli_exact(a))
            ref_label = 'exact'
        diff = abs(float(bproj[i]) - float(ref))
        print(f'{a:>12.3e}  {float(bproj[i]):>15.10e}  '
              f'{float(bernoulli_exact(a)):>15.10e}  '
              f'{float(bernoulli_taylor(a)):>15.10e}  {diff:>15.3e}')

    print('\nB(0) exact value should be 1.0')
    print(f'  _bernoulli(0) = {float(solver._bernoulli(np.array([0.0]))[0]):.16e}')
    print(f'  taylor(0)     = {float(bernoulli_taylor(0.0)):.16e}')

    # Decisive symmetry test: B(a) - B(-a) should equal -a (identity for B)
    # i.e. B(a) - B(-a) = -a   (because B(a) + a/2 is even in a)
    print('\nIdentity check:  B(a) - B(-a) should equal -a')
    print(f'{"alpha":>12s}  {"B(a)-B(-a)":>16s}  {"-alpha":>16s}  {"diff":>12s}')
    for a in [1e-10, 1e-5, 1e-2, 1.0, 10.0, 100.0]:
        ba = float(solver._bernoulli(np.array([a]))[0])
        bma = float(solver._bernoulli(np.array([-a]))[0])
        diff = (ba - bma) - (-a)
        print(f'{a:>12.3e}  {ba - bma:>16.6e}  {-a:>16.6e}  {diff:>12.3e}')


if __name__ == '__main__':
    main()
