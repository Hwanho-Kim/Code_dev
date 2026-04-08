"""
PINN loss functions.

L_total = λ_d·L_data + λ_p·L_physics + λ_h·L_HONO_cross
          + λ_c·L_conservation + λ_s·L_smooth

All losses operate on autograd-enabled tensors for backpropagation.
"""
import torch

from .model import (
    IDX_O3, IDX_NO2, IDX_NO3, IDX_N2O5, IDX_HONO,
    IDX_NO, IDX_HONO2, IDX_H2O2, IDX_Q_OH, IDX_Q_HONO, IDX_Q_NO,
    N_MEASURED,
)
from .qssa import solve_oh_ho2_coupled
from .reactions import compute_rates, compute_n2o4


def l_data(
    log_pred: torch.Tensor,
    log_target: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between predicted and measured log₁₀ concentrations.
    Only first 5 columns (measured species).

    Species-specific weights (inverse variance of log₁₀ noise):
      O₃: high SNR (weight=1), NO₂: medium (1), NO₃: noisy (0.5),
      N₂O₅: medium (1), HONO: noisy at low RH (0.3)
    """
    weights = torch.tensor([1.0, 1.0, 0.5, 1.0, 0.3],
                           device=log_pred.device, dtype=log_pred.dtype)

    diff = log_pred[:, :N_MEASURED] - log_target[:, :N_MEASURED]
    return (weights * diff * diff).mean()


def _compute_dcdt_autograd(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    output_idx: int,
) -> torch.Tensor:
    """
    Compute d(log₁₀ c)/dt via autograd, then convert to dc/dt in linear scale.

    d(c)/dt = c · ln(10) · d(log₁₀ c)/d(t̃) · (1/T_MAX)

    where t̃ = t/T_MAX is the normalized time (input[:, 0]).
    """
    T_MAX = 600.0
    inputs_req = inputs.detach().requires_grad_(True)
    log_out = model(inputs_req)

    log_c = log_out[:, output_idx]
    grad = torch.autograd.grad(
        log_c, inputs_req,
        grad_outputs=torch.ones_like(log_c),
        create_graph=True,
    )[0]

    # d(log₁₀ c)/dt̃ — gradient w.r.t. normalized time (column 0)
    dlog_dt_norm = grad[:, 0]

    # Convert: dc/dt = c · ln(10) · dlog/dt̃ / T_MAX
    c = 10.0 ** log_c
    dcdt = c * 2.302585 * dlog_dt_norm / T_MAX

    return dcdt


def l_physics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Rate residuals for 3 unmeasured species: NO, HONO₂, H₂O₂.

    L = Σ || dc/dt (autograd) - R(c) (kinetics) ||²

    Normalized by characteristic rate scale to balance species.
    """
    inputs_req = inputs.detach().requires_grad_(True)
    log_out = model(inputs_req)
    conc = 10.0 ** log_out

    o3    = conc[:, IDX_O3]
    no2   = conc[:, IDX_NO2]
    hono  = conc[:, IDX_HONO]
    no    = conc[:, IDX_NO]
    hono2 = conc[:, IDX_HONO2]
    h2o2  = conc[:, IDX_H2O2]
    q_oh  = conc[:, IDX_Q_OH]
    q_hono = conc[:, IDX_Q_HONO]
    q_no  = conc[:, IDX_Q_NO]

    oh, ho2 = solve_oh_ho2_coupled(o3, no2, hono, hono2, h2o2, no, q_oh)

    rates = compute_rates(
        o3, no2, conc[:, IDX_NO3], conc[:, IDX_N2O5],
        hono, no, hono2, h2o2, oh, ho2, q_oh, q_hono, q_no,
    )

    T_MAX = 600.0

    # dc/dt via autograd for each unmeasured species
    loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
    for idx, rate_key, scale in [
        (IDX_NO,    'dNO_dt',    1e13),
        (IDX_HONO2, 'dHONO2_dt', 1e13),
        (IDX_H2O2,  'dH2O2_dt',  1e12),
    ]:
        log_c = log_out[:, idx]
        grad = torch.autograd.grad(
            log_c, inputs_req,
            grad_outputs=torch.ones_like(log_c),
            create_graph=True,
            retain_graph=True,
        )[0]
        c = 10.0 ** log_c
        dcdt_nn = c * 2.302585 * grad[:, 0] / T_MAX

        residual = (dcdt_nn - rates[rate_key]) / scale
        loss = loss + (residual ** 2).mean()

    return loss / 3.0


def l_hono_cross(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    HONO cross-validation: measured HONO rate vs kinetic prediction.

    HONO is measured (constrained by L_data), so d[HONO]/dt from autograd
    should match R_HONO = Q_HONO + NEW1(NO·OH) + R6 - R12(OH·HONO).

    This independently validates the OH estimate (through Q_OH).
    """
    inputs_req = inputs.detach().requires_grad_(True)
    log_out = model(inputs_req)
    conc = 10.0 ** log_out

    o3    = conc[:, IDX_O3]
    no2   = conc[:, IDX_NO2]
    hono  = conc[:, IDX_HONO]
    no    = conc[:, IDX_NO]
    hono2 = conc[:, IDX_HONO2]
    h2o2  = conc[:, IDX_H2O2]
    q_oh  = conc[:, IDX_Q_OH]
    q_hono = conc[:, IDX_Q_HONO]
    q_no  = conc[:, IDX_Q_NO]

    oh, ho2 = solve_oh_ho2_coupled(o3, no2, hono, hono2, h2o2, no, q_oh)

    rates = compute_rates(
        o3, no2, conc[:, IDX_NO3], conc[:, IDX_N2O5],
        hono, no, hono2, h2o2, oh, ho2, q_oh, q_hono, q_no,
    )

    T_MAX = 600.0

    log_hono = log_out[:, IDX_HONO]
    grad = torch.autograd.grad(
        log_hono, inputs_req,
        grad_outputs=torch.ones_like(log_hono),
        create_graph=True,
    )[0]
    dhono_dt_nn = conc[:, IDX_HONO] * 2.302585 * grad[:, 0] / T_MAX

    scale = 1e11
    residual = (dhono_dt_nn - rates['dHONO_dt']) / scale
    return (residual ** 2).mean()


def l_conservation(
    log_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Nitrogen atom conservation.

    N_total = [NO] + [NO₂] + [NO₃] + 2[N₂O₄] + 2[N₂O₅] + [HONO] + [HONO₂]

    Within each condition, dN_total/dt should be ≈ Q_N (plasma production).
    We penalize large d²N_total/dt² (smooth production rate).
    """
    conc = 10.0 ** log_pred

    no2  = conc[:, IDX_NO2]
    n2o4 = compute_n2o4(no2)

    n_total = (conc[:, IDX_NO]
               + no2
               + conc[:, IDX_NO3]
               + 2.0 * n2o4
               + 2.0 * conc[:, IDX_N2O5]
               + conc[:, IDX_HONO]
               + conc[:, IDX_HONO2])

    # Penalize second derivative (smoothness of N production)
    # Using finite differences along batch (assumes sorted by time within condition)
    if len(n_total) < 3:
        return torch.tensor(0.0, device=log_pred.device)

    d2n = n_total[2:] - 2.0 * n_total[1:-1] + n_total[:-2]
    scale = n_total.mean().detach().clamp(min=1e10)
    return ((d2n / scale) ** 2).mean()


def l_smooth(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Temporal smoothness for unmeasured species.
    Penalizes d²(log₁₀ c)/dt̃² to prevent oscillations.
    """
    log_pred = model(inputs)

    loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
    for idx in [IDX_NO, IDX_HONO2, IDX_H2O2]:
        vals = log_pred[:, idx]
        if len(vals) < 3:
            continue
        d2 = vals[2:] - 2.0 * vals[1:-1] + vals[:-2]
        loss = loss + (d2 ** 2).mean()

    return loss / 3.0


class PINNLoss:
    """Combined loss with curriculum scheduling."""

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_hono_cross: float = 1.0,
        lambda_conservation: float = 10.0,
        lambda_smooth: float = 0.001,
    ):
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_hono_cross = lambda_hono_cross
        self.lambda_conservation = lambda_conservation
        self.lambda_smooth = lambda_smooth
        self._physics_scale = 0.0  # curriculum: 0 → 1

    def set_curriculum(self, progress: float):
        """
        progress: 0.0 (start) → 1.0 (end of Phase 2)
        Ramps physics/hono_cross from 0 to their nominal values.
        """
        self._physics_scale = min(1.0, max(0.0, progress))

    def __call__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        log_pred: torch.Tensor,
        log_target: torch.Tensor,
        phase: int = 1,
    ) -> dict[str, torch.Tensor]:

        losses = {}
        losses['data'] = self.lambda_data * l_data(log_pred, log_target)

        if phase >= 2:
            scale = self._physics_scale
            losses['physics'] = self.lambda_physics * scale * l_physics(model, inputs)
            losses['hono_cross'] = self.lambda_hono_cross * scale * l_hono_cross(model, inputs)

        if phase >= 3:
            losses['conservation'] = self.lambda_conservation * l_conservation(log_pred)
            losses['smooth'] = self.lambda_smooth * l_smooth(model, inputs)

        losses['total'] = sum(losses.values())
        return losses
