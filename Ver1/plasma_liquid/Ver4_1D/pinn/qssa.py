"""
Quasi-Steady-State Approximation (QSSA) for OH and HO₂.

Both species have lifetimes << 2s observation interval:
  τ_OH  ~ 1 ms  (dominated by k8·[O₃] ~ 730 s⁻¹)
  τ_HO₂ ~ 50 ms (dominated by k_NEW3·[O₃] + k_NEW2·[NO] ~ 20-100 s⁻¹)

All operations are PyTorch-differentiable for backpropagation through PINN.
"""
import torch

from .reactions import (
    K_R8, K_R9, K_R10, K_R12, K_R13, K_R14,
    K_R7, K_R11, K_NEW1, K_NEW2, K_NEW3,
)

# Small epsilon to avoid division by zero / negative sqrt
_EPS = 1e-30


def solve_oh_qssa(
    o3: torch.Tensor,
    no2: torch.Tensor,
    hono: torch.Tensor,
    hono2: torch.Tensor,
    h2o2: torch.Tensor,
    no: torch.Tensor,
    ho2: torch.Tensor,
    q_oh: torch.Tensor,
) -> torch.Tensor:
    """
    Solve d[OH]/dt = 0 for [OH].

    Production:
        Q_OH + k_NEW2·[HO₂]·[NO] + k_NEW3·[HO₂]·[O₃]

    Destruction (linear in [OH]):
        k8·[O₃] + k7·[NO₂] + k_NEW1·[NO] + k12·[HONO] + k13·[HONO₂]
        + k14·[H₂O₂] + k10·[HO₂]

    Destruction (quadratic in [OH]):
        2·k9·[OH]

    Equation: a·OH² + b·OH - c = 0
    Solution: OH = (-b + sqrt(b² + 4ac)) / (2a)
    """
    # Quadratic coefficient (OH self-reaction)
    a = 2.0 * K_R9

    # Linear consumption coefficient
    b = (K_R8 * o3
         + K_R7 * no2
         + K_NEW1 * no
         + K_R12 * hono
         + K_R13 * hono2
         + K_R14 * h2o2
         + K_R10 * ho2)

    # Production (independent of OH)
    c = q_oh + K_NEW2 * ho2 * no + K_NEW3 * ho2 * o3

    # Quadratic formula: OH = (-b + sqrt(b² + 4ac)) / (2a)
    discriminant = b * b + 4.0 * a * c
    oh = (-b + torch.sqrt(discriminant.clamp(min=_EPS))) / (2.0 * a + _EPS)

    return oh.clamp(min=0.0)


def solve_ho2_qssa(
    o3: torch.Tensor,
    h2o2: torch.Tensor,
    no: torch.Tensor,
    oh: torch.Tensor,
) -> torch.Tensor:
    """
    Solve d[HO₂]/dt = 0 for [HO₂].

    Production:
        k8·[OH]·[O₃] + k14·[OH]·[H₂O₂]

    Destruction (linear):
        k10·[OH] + k_NEW2·[NO] + k_NEW3·[O₃]

    Destruction (quadratic):
        2·k11·[HO₂]

    Equation: a·HO₂² + b·HO₂ - c = 0
    """
    a = 2.0 * K_R11

    b = K_R10 * oh + K_NEW2 * no + K_NEW3 * o3

    c = K_R8 * oh * o3 + K_R14 * oh * h2o2

    discriminant = b * b + 4.0 * a * c
    ho2 = (-b + torch.sqrt(discriminant.clamp(min=_EPS))) / (2.0 * a + _EPS)

    return ho2.clamp(min=0.0)


def solve_oh_ho2_coupled(
    o3: torch.Tensor,
    no2: torch.Tensor,
    hono: torch.Tensor,
    hono2: torch.Tensor,
    h2o2: torch.Tensor,
    no: torch.Tensor,
    q_oh: torch.Tensor,
    n_iter: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iteratively solve coupled OH-HO₂ QSSA.

    OH depends on HO₂ (via NEW2, NEW3, R10).
    HO₂ depends on OH (via R8, R14, R10).

    Start with HO₂=0, iterate OH→HO₂→OH until convergence (~3 iterations).

    Returns:
        (oh, ho2) tensors, same shape as inputs
    """
    # Initial guess: HO₂ = 0
    ho2 = torch.zeros_like(o3)

    for _ in range(n_iter):
        oh = solve_oh_qssa(o3, no2, hono, hono2, h2o2, no, ho2, q_oh)
        ho2 = solve_ho2_qssa(o3, h2o2, no, oh)

    return oh, ho2
