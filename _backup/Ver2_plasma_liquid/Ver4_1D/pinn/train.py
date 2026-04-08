#!/usr/bin/env python3
"""
3-phase curriculum training for PlasmaChemPINN.

Phase 1 (warm-up):   L_data only — NN learns measured species shapes
Phase 2 (physics):   + L_physics, L_HONO_cross (curriculum ramp 0→1)
Phase 3 (refine):    + L_conservation, L_smooth — full physics consistency

Usage:
    python -m pinn.train                    # full training
    python -m pinn.train --epochs 100       # quick test
    python -m pinn.train --loo 55_3.2       # leave-one-out (RH=55, V=3.2)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .data_loader import PlasmaOASDataset
from .model import PlasmaChemPINN
from .loss import PINNLoss


SAVE_DIR = Path(__file__).parent / 'checkpoints'


def train(
    epochs_phase1: int = 5000,
    epochs_phase2: int = 20000,
    epochs_phase3: int = 10000,
    lr_init: float = 1e-3,
    lr_final: float = 1e-5,
    weight_decay: float = 1e-4,
    exclude_condition: tuple | None = None,
    save_every: int = 5000,
    print_every: int = 500,
    device: str = 'cpu',
):
    SAVE_DIR.mkdir(exist_ok=True)

    ds = PlasmaOASDataset(exclude_condition=exclude_condition)
    print(ds.summary())
    print()

    inputs  = ds.inputs.to(device)
    targets = ds.targets.to(device)

    model = PlasmaChemPINN().to(device)
    print(f'Parameters: {model.count_parameters():,}')

    criterion = PINNLoss()

    total_epochs = epochs_phase1 + epochs_phase2 + epochs_phase3
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_init, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=lr_final,
    )

    history = []
    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, total_epochs + 1):
        # Determine phase
        if epoch <= epochs_phase1:
            phase = 1
            criterion.set_curriculum(0.0)
        elif epoch <= epochs_phase1 + epochs_phase2:
            phase = 2
            progress = (epoch - epochs_phase1) / epochs_phase2
            criterion.set_curriculum(progress)
        else:
            phase = 3
            criterion.set_curriculum(1.0)

        optimizer.zero_grad()
        log_pred = model(inputs)
        losses = criterion(model, inputs, log_pred, targets, phase=phase)
        losses['total'].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_val = losses['total'].item()

        if epoch % print_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            parts = [f"[{epoch:6d}/{total_epochs}] P{phase}"]
            for k, v in losses.items():
                parts.append(f"{k}={v.item():.4f}")
            parts.append(f"lr={lr:.1e}")
            parts.append(f"({time.time()-t0:.0f}s)")
            print('  '.join(parts))

        if epoch % save_every == 0:
            record = {'epoch': epoch, 'phase': phase}
            for k, v in losses.items():
                record[k] = v.item()
            history.append(record)

        if total_val < best_loss:
            best_loss = total_val
            torch.save(model.state_dict(), SAVE_DIR / 'best.pt')

    # Final save
    torch.save(model.state_dict(), SAVE_DIR / 'final.pt')

    with open(SAVE_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t0
    print(f'\nTraining complete: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'Best loss: {best_loss:.6f}')
    print(f'Saved: {SAVE_DIR}/best.pt, final.pt, history.json')

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train PlasmaChemPINN')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override total epochs (splits 15/60/25%%)')
    parser.add_argument('--epochs1', type=int, default=5000)
    parser.add_argument('--epochs2', type=int, default=20000)
    parser.add_argument('--epochs3', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loo', type=str, default=None,
                        help='Leave-one-out: "RH_V" e.g. "55_3.2"')
    parser.add_argument('--print-every', type=int, default=500)
    args = parser.parse_args()

    exclude = None
    if args.loo:
        parts = args.loo.split('_')
        exclude = (int(parts[0]), float(parts[1]))
        print(f'Leave-one-out: excluding RH={exclude[0]}%, V={exclude[1]}kV')

    if args.epochs:
        e1 = int(args.epochs * 0.15)
        e2 = int(args.epochs * 0.60)
        e3 = args.epochs - e1 - e2
    else:
        e1, e2, e3 = args.epochs1, args.epochs2, args.epochs3

    train(
        epochs_phase1=e1, epochs_phase2=e2, epochs_phase3=e3,
        lr_init=args.lr, exclude_condition=exclude,
        print_every=args.print_every,
    )


if __name__ == '__main__':
    main()
