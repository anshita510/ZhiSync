"""
ZhiSync model training script.

Trains all models for 10 epochs using a batch size of 32, CrossEntropyLoss,
and the Adam optimizer.

Trains both the local (baseline) model and the context-aware (ZhiAware)
variant for each modality:

  ECG    → ECGNet(context_dim=0)  +  ECGNet(context_dim=2)
  Breath → CovidCoughNet(context_dim=0)  +  CovidCoughNet(context_dim=2)
  Motion → MotionMLP(context_dim=0)  +  MotionMLP(context_dim=2)

Saved model files (in --model-dir, default ./simulation/saved_models/):
  ecg_local.pt, ecg_context.pt
  breath_local.pt, breath_context.pt
  motion_local.pt, motion_context.pt

Usage::

  python -m simulation.train
  python -m simulation.train --data-dir /path/to/datasets --epochs 10

Context-aware training strategy:
  During training, peer metadata is *simulated*: for each batch we randomly
  draw peer confidence values from Beta(3,2) for positive (abnormal) samples
  and Beta(2,5) for negative (normal) samples, and assign peer urgency = 1
  when peer_conf >= urgency_threshold, 0 otherwise.
  This lets the context-aware model learn that high peer urgency should
  increase its own classification confidence for critical events.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulation.models.ecg_model import ECGNet
from simulation.models.breath_model import CovidCoughNet
from simulation.models.motion_model import MotionMLP
from simulation.datasets.ecg_dataset import ECGDataset
from simulation.datasets.breath_dataset import BreathDataset
from simulation.datasets.motion_dataset import MotionDataset

logger = logging.getLogger(__name__)

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
URGENCY_THRESHOLD = 0.8


def _simulate_peer_context(
    labels: torch.Tensor,
    rng: torch.Generator,
    urgency_threshold: float = URGENCY_THRESHOLD,
) -> torch.Tensor:
    """
    Simulate peer metadata for context-aware training.

    For positive (label=1, abnormal) samples:  peer_conf ~ Beta(3,2) → high energy.
    For negative (label=0, normal) samples:    peer_conf ~ Beta(2,5) → low energy.
    peer_urgency_binary = 1 if peer_conf >= urgency_threshold else 0.
    """
    n = len(labels)
    peer_conf = torch.empty(n, dtype=torch.float32)
    positive_mask = labels == 1

    # Beta distribution via torch (using Beta through Dirichlet approximation)
    # Positive samples: α=3, β=2  → mean 0.6
    alpha_pos = torch.full((int(positive_mask.sum()),), 3.0)
    beta_pos = torch.full((int(positive_mask.sum()),), 2.0)
    peer_conf[positive_mask] = torch.distributions.Beta(alpha_pos, beta_pos).sample()

    # Negative samples: α=2, β=5  → mean 0.29
    n_neg = n - int(positive_mask.sum())
    if n_neg > 0:
        alpha_neg = torch.full((n_neg,), 2.0)
        beta_neg = torch.full((n_neg,), 5.0)
        peer_conf[~positive_mask] = torch.distributions.Beta(alpha_neg, beta_neg).sample()

    peer_urgency = (peer_conf >= urgency_threshold).float()
    return torch.stack([peer_conf, peer_urgency], dim=1)


def _train_one_modality(
    name: str,
    local_model: nn.Module,
    context_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    model_dir: Path,
    epochs: int = EPOCHS,
) -> None:
    """Train both local and context-aware models for one modality."""

    criterion = nn.CrossEntropyLoss()
    local_opt = torch.optim.Adam(local_model.parameters(), lr=LR)
    context_opt = torch.optim.Adam(context_model.parameters(), lr=LR)

    local_model.to(device).train()
    context_model.to(device).train()

    rng = torch.Generator()
    rng.manual_seed(42)

    for epoch in range(epochs):
        local_loss_sum = 0.0
        ctx_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # --- Local model (baseline) ---
            local_opt.zero_grad()
            local_out = local_model(x)
            local_loss = criterion(local_out, y)
            local_loss.backward()
            local_opt.step()
            local_loss_sum += float(local_loss.item())

            # --- Context-aware model (ZhiAware) ---
            context_opt.zero_grad()
            peer_ctx = _simulate_peer_context(y.cpu(), rng).to(device)
            ctx_out = context_model(x, peer_ctx)
            ctx_loss = criterion(ctx_out, y)
            ctx_loss.backward()
            context_opt.step()
            ctx_loss_sum += float(ctx_loss.item())

            n_batches += 1

        avg_local = local_loss_sum / max(n_batches, 1)
        avg_ctx = ctx_loss_sum / max(n_batches, 1)
        logger.info(
            "[%s] epoch %2d/%d  local_loss=%.4f  context_loss=%.4f",
            name, epoch + 1, epochs, avg_local, avg_ctx,
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(local_model.state_dict(), model_dir / f"{name}_local.pt")
    torch.save(context_model.state_dict(), model_dir / f"{name}_context.pt")
    logger.info("Saved %s models to %s", name, model_dir)


def train_all(
    data_dir: str | None = None,
    model_dir: str = "simulation/saved_models",
    epochs: int = EPOCHS,
    device_str: str = "cpu",
) -> None:
    """Train all three modality models (local + context-aware)."""
    device = torch.device(device_str)
    mdl_dir = Path(model_dir)

    # --- ECG ---
    logger.info("=== Training ECG models ===")
    ecg_train = ECGDataset(data_dir=data_dir, split="train")
    ecg_loader = DataLoader(ecg_train, batch_size=BATCH_SIZE, shuffle=True)
    _train_one_modality(
        "ecg",
        ECGNet(num_classes=2, context_dim=0),
        ECGNet(num_classes=2, context_dim=2),
        ecg_loader, device, mdl_dir, epochs,
    )

    # --- Breath ---
    logger.info("=== Training Breath models ===")
    breath_train = BreathDataset(data_dir=data_dir, split="train")
    breath_loader = DataLoader(breath_train, batch_size=BATCH_SIZE, shuffle=True)
    _train_one_modality(
        "breath",
        CovidCoughNet(num_classes=2, context_dim=0),
        CovidCoughNet(num_classes=2, context_dim=2),
        breath_loader, device, mdl_dir, epochs,
    )

    # --- Motion ---
    logger.info("=== Training Motion models ===")
    motion_train = MotionDataset(data_dir=data_dir, split="train")
    motion_loader = DataLoader(motion_train, batch_size=BATCH_SIZE, shuffle=True)
    _train_one_modality(
        "motion",
        MotionMLP(num_classes=6, context_dim=0),
        MotionMLP(num_classes=6, context_dim=2),
        motion_loader, device, mdl_dir, epochs,
    )


def load_models(
    model_dir: str = "simulation/saved_models",
    device: torch.device | None = None,
) -> dict[str, nn.Module]:
    """
    Load all six saved models.

    Returns dict with keys:
      ecg_local, ecg_context, breath_local, breath_context,
      motion_local, motion_context
    """
    device = device or torch.device("cpu")
    mdl_dir = Path(model_dir)

    specs = {
        "ecg_local":      ECGNet(num_classes=2, context_dim=0),
        "ecg_context":    ECGNet(num_classes=2, context_dim=2),
        "breath_local":   CovidCoughNet(num_classes=2, context_dim=0),
        "breath_context": CovidCoughNet(num_classes=2, context_dim=2),
        "motion_local":   MotionMLP(num_classes=6, context_dim=0),
        "motion_context": MotionMLP(num_classes=6, context_dim=2),
    }

    loaded: dict[str, nn.Module] = {}
    for key, model in specs.items():
        ckpt_path = mdl_dir / f"{key}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            logger.info("Loaded %s from %s", key, ckpt_path)
        else:
            logger.warning("Checkpoint not found for %s at %s — using random weights.", key, ckpt_path)
        loaded[key] = model.to(device).eval()

    return loaded


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train ZhiSync simulation models.")
    parser.add_argument("--data-dir", default=None, help="Path to datasets root directory.")
    parser.add_argument("--model-dir", default="simulation/saved_models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (cpu / cuda / mps).",
    )
    args = parser.parse_args()

    train_all(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        device_str=args.device,
    )
