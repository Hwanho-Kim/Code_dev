"""PlasmaChemPINN: [32,64,64,32] → 11 outputs, log₁₀ encoded."""
import torch
import torch.nn as nn


# Output indices
IDX_O3    = 0
IDX_NO2   = 1
IDX_NO3   = 2
IDX_N2O5  = 3
IDX_HONO  = 4
IDX_NO    = 5
IDX_HONO2 = 6
IDX_H2O2  = 7
IDX_Q_OH  = 8
IDX_Q_HONO = 9
IDX_Q_NO  = 10

N_MEASURED   = 5   # O3, NO2, NO3, N2O5, HONO
N_UNMEASURED = 3   # NO, HONO2, H2O2
N_SOURCE     = 3   # Q_OH, Q_HONO, Q_NO
N_OUTPUT     = N_MEASURED + N_UNMEASURED + N_SOURCE  # 11


class PlasmaChemPINN(nn.Module):
    """
    Input:  (t̃, Ṽ, R̃H) — 3 normalized scalars
    Output: 11 values (log₁₀ scale)
      [0:5]  measured species log₁₀ concentrations (cm⁻³)
      [5:8]  unmeasured species log₁₀ concentrations
      [8:11] learnable source terms log₁₀ (cm⁻³ s⁻¹)

    Non-negativity guaranteed by log₁₀ encoding: c = 10^(output).
    """

    def __init__(self, hidden_sizes=(32, 64, 64, 32)):
        super().__init__()

        layers = []
        in_dim = 3
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, N_OUTPUT))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3) — normalized (t, V, RH)
        Returns:
            (batch, 11) — log₁₀ values
        """
        return self.net(x)

    def predict_concentrations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass → named concentration tensors in linear scale (cm⁻³)."""
        log_out = self.forward(x)
        conc = 10.0 ** log_out

        return {
            'O3':    conc[:, IDX_O3],
            'NO2':   conc[:, IDX_NO2],
            'NO3':   conc[:, IDX_NO3],
            'N2O5':  conc[:, IDX_N2O5],
            'HONO':  conc[:, IDX_HONO],
            'NO':    conc[:, IDX_NO],
            'HONO2': conc[:, IDX_HONO2],
            'H2O2':  conc[:, IDX_H2O2],
            'Q_OH':  conc[:, IDX_Q_OH],
            'Q_HONO': conc[:, IDX_Q_HONO],
            'Q_NO':  conc[:, IDX_Q_NO],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
