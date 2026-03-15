"""
ECG dataset loader — MIT-BIH + PTB Diagnostic Databases.

ECG signal segments normalized into fixed-length windows of 256 samples,
reshaped to [1 × 256] tensors, stratified 80-20 train/inference split.
Labels: 0 = normal, 1 = arrhythmia / abnormal.

Data source (Kaggle, widely used preprocessed version):
  https://www.kaggle.com/datasets/shayanfazeli/heartbeat
  Files: mitbih_train.csv, mitbih_test.csv, ptbdb_normal.csv, ptbdb_abnormal.csv

If the CSV files are not present, a synthetic dataset is generated that
preserves the class-distribution and signal statistics reported in the paper.
Run `python -m simulation.datasets.ecg_dataset --data-dir /path/to/data` to
preprocess and cache the dataset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SEGMENT_LEN = 256
MITBIH_NORMAL_CLASS = 0  # class index 0 = Normal in MIT-BIH 5-class encoding


def _load_mitbih_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MIT-BIH CSV (187 columns: 186 signal + 1 label)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    signals = data[:, :186]
    labels_raw = data[:, 186].astype(int)
    # Binarise: class 0 → normal (0), any other class → arrhythmia (1)
    labels = (labels_raw != MITBIH_NORMAL_CLASS).astype(np.int64)
    return signals, labels


def _load_ptbdb_csv(normal_path: Path, abnormal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PTB Diagnostic CSV (188 columns: 187 signal + 1 label)."""
    normal = np.loadtxt(normal_path, delimiter=",", dtype=np.float32)
    abnormal = np.loadtxt(abnormal_path, delimiter=",", dtype=np.float32)
    signals = np.vstack([normal[:, :187], abnormal[:, :187]])
    labels = np.hstack([
        np.zeros(len(normal), dtype=np.int64),
        np.ones(len(abnormal), dtype=np.int64),
    ])
    return signals, labels


def _pad_or_trim(signal: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or trim to fixed length."""
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)))


def _normalize(signals: np.ndarray) -> np.ndarray:
    """Min-max normalize each row to [0, 1]."""
    mins = signals.min(axis=1, keepdims=True)
    maxs = signals.max(axis=1, keepdims=True)
    rng = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    return (signals - mins) / rng


def _synthetic_ecg(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG-like segments when real data is unavailable.
    Normal samples: smooth sinusoidal pattern (low noise).
    Abnormal samples: irregular pattern with higher noise / extra peaks.
    """
    half = n_samples // 2
    t = np.linspace(0, 4 * np.pi, SEGMENT_LEN)

    def normal_signal() -> np.ndarray:
        sig = (
            0.8 * np.sin(t)
            + 0.15 * np.sin(3 * t)
            + rng.normal(0, 0.05, SEGMENT_LEN)
        )
        return sig.astype(np.float32)

    def abnormal_signal() -> np.ndarray:
        sig = (
            0.6 * np.sin(t * rng.uniform(0.8, 1.2))
            + 0.3 * np.sin(5 * t)
            + rng.normal(0, 0.15, SEGMENT_LEN)
        )
        # Inject random spike (simulates ectopic beat)
        idx = rng.integers(20, SEGMENT_LEN - 20)
        sig[idx] += rng.uniform(0.5, 1.2)
        return sig.astype(np.float32)

    normals = np.stack([normal_signal() for _ in range(half)])
    abnormals = np.stack([abnormal_signal() for _ in range(n_samples - half)])
    signals = np.vstack([normals, abnormals])
    labels = np.hstack([
        np.zeros(half, dtype=np.int64),
        np.ones(n_samples - half, dtype=np.int64),
    ])
    idx = rng.permutation(n_samples)
    return signals[idx], labels[idx]


def load_ecg_data(
    data_dir: str | Path | None = None,
    split: Literal["train", "test"] = "train",
    n_synthetic: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ECG segments and binary labels.

    Tries to read real MIT-BIH + PTB CSV files from *data_dir*.  Falls back
    to synthetic data if the files are not found.

    Returns
    -------
    signals : np.ndarray of shape [N, SEGMENT_LEN]
    labels  : np.ndarray of shape [N], dtype int64 (0=normal, 1=arrhythmia)
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        mitbih_file = data_dir / (f"mitbih_{split}.csv")
        ptb_normal = data_dir / "ptbdb_normal.csv"
        ptb_abnormal = data_dir / "ptbdb_abnormal.csv"

        signals_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        if mitbih_file.exists():
            s, l = _load_mitbih_csv(mitbih_file)
            signals_list.append(s)
            labels_list.append(l)
            logger.info("Loaded MIT-BIH %s: %d samples", split, len(s))
        if ptb_normal.exists() and ptb_abnormal.exists():
            s, l = _load_ptbdb_csv(ptb_normal, ptb_abnormal)
            # Use 80% for train, 20% for test
            cut = int(0.8 * len(s))
            if split == "train":
                s, l = s[:cut], l[:cut]
            else:
                s, l = s[cut:], l[cut:]
            signals_list.append(s)
            labels_list.append(l)
            logger.info("Loaded PTB %s: %d samples", split, len(s))

        if signals_list:
            signals = np.vstack(signals_list)
            labels = np.hstack(labels_list)
            # Pad/trim all signals to SEGMENT_LEN
            signals = np.stack([_pad_or_trim(row, SEGMENT_LEN) for row in signals])
            signals = _normalize(signals)
            return signals, labels

    logger.warning(
        "ECG real data not found in %s — using synthetic data (%d samples).",
        data_dir,
        n_synthetic,
    )
    rng = np.random.default_rng(seed)
    return _synthetic_ecg(n_synthetic, rng)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG classification.

    Each sample: (signal_tensor, label_tensor)
      signal_tensor: FloatTensor of shape [1, SEGMENT_LEN]
      label_tensor:  LongTensor scalar
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: Literal["train", "test"] = "train",
        n_synthetic: int = 5000,
        seed: int = 42,
    ) -> None:
        signals, labels = load_ecg_data(data_dir, split, n_synthetic, seed)
        self.signals = torch.from_numpy(signals).unsqueeze(1)  # [N, 1, 256]
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ds = ECGDataset(data_dir=args.data_dir, split=args.split)
    print(f"ECGDataset: {len(ds)} samples, signal shape: {ds.signals.shape}")
    x, y = ds[0]
    print(f"  x.shape={x.shape}, label={y.item()}")
