"""
Breath dataset loader — COUGHVID crowdsourcing dataset.

Audio samples from COUGHVID preprocessed with MFCC + Chroma + Spectral Contrast.
70-30 stratified train/inference split.
Labels: 0 = normal breathing, 1 = cough / abnormal.

Data source:
  https://zenodo.org/record/4048312  (or via Kaggle: lnstagram/coughvid)

Expected directory layout::

    <data_dir>/
        *.wav   (or *.ogg / *.webm — librosa handles all)
        metadata.csv   (columns: uuid, status, ...)

If data is absent, synthetic feature vectors are generated to allow training
and simulation without downloading the ~6 GB dataset.

Feature extraction produces a 59-dimensional vector per audio clip:
  MFCC (n_mfcc=40) mean over time — 40 dims
  Chroma mean over time              — 12 dims
  Spectral Contrast mean over time   — 7  dims
  Total: 59 dims  (matches simulation/models/breath_model.py FEATURE_DIM)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_DIM = 59     # must match breath_model.FEATURE_DIM
SAMPLE_RATE = 22050
N_MFCC = 40


def _extract_features(audio_path: Path) -> np.ndarray | None:
    """Extract 59-D feature vector from a single audio file using librosa."""
    try:
        import librosa  # optional import — only needed with real data

        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=5.0)
        if len(y) < 512:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)         # [40, T]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)                 # [12, T]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)         # [7, T]

        features = np.concatenate([
            mfcc.mean(axis=1),      # 40
            chroma.mean(axis=1),    # 12
            contrast.mean(axis=1),  # 7
        ])
        return features.astype(np.float32)
    except Exception as exc:
        logger.debug("Feature extraction failed for %s: %s", audio_path, exc)
        return None


def _load_coughvid_dir(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk data_dir for audio files and metadata.csv, extract features, split.
    """
    import csv

    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.int64)

    # Read labels from metadata: 'status' column — COVID-19/symptomatic → 1, healthy → 0
    label_map: dict[str, int] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uuid", "").strip()
            status = row.get("status", "").strip().lower()
            label_map[uid] = 0 if status == "healthy" else 1

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    audio_exts = {".wav", ".ogg", ".webm", ".flac", ".mp3"}
    for audio_path in sorted(data_dir.iterdir()):
        if audio_path.suffix.lower() not in audio_exts:
            continue
        uid = audio_path.stem
        if uid not in label_map:
            continue
        feat = _extract_features(audio_path)
        if feat is None:
            continue
        features_list.append(feat)
        labels_list.append(label_map[uid])

    if not features_list:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.int64)

    feats = np.stack(features_list)
    labels = np.array(labels_list, dtype=np.int64)

    # 70-30 stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=0.30, stratify=labels, random_state=42
    )
    if split == "train":
        return X_train, y_train
    return X_test, y_test


def _synthetic_breath(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic 59-D feature vectors mimicking MFCC+Chroma+Spectral Contrast
    statistics for normal breathing vs cough.
    Normal: low-energy, smooth spectrum (lower MFCC values).
    Cough:  high-energy, irregular spectrum (higher MFCC, more contrast).
    """
    half = n_samples // 2

    # Normal breathing: MFCC centered around small negative values (quiet audio)
    normal_mfcc = rng.normal(-15, 8, (half, N_MFCC)).astype(np.float32)
    normal_mfcc[:, 0] = rng.normal(-200, 20, half)   # first MFCC = overall energy
    normal_chroma = rng.uniform(0.3, 0.6, (half, 12)).astype(np.float32)
    normal_contrast = rng.normal(20, 5, (half, 7)).astype(np.float32)
    normal_feats = np.hstack([normal_mfcc, normal_chroma, normal_contrast])

    # Cough: higher energy, more irregular
    n_ab = n_samples - half
    cough_mfcc = rng.normal(-5, 15, (n_ab, N_MFCC)).astype(np.float32)
    cough_mfcc[:, 0] = rng.normal(-100, 30, n_ab)
    cough_chroma = rng.uniform(0.2, 0.9, (n_ab, 12)).astype(np.float32)
    cough_contrast = rng.normal(35, 8, (n_ab, 7)).astype(np.float32)
    cough_feats = np.hstack([cough_mfcc, cough_chroma, cough_contrast])

    feats = np.vstack([normal_feats, cough_feats])
    labels = np.hstack([
        np.zeros(half, dtype=np.int64),
        np.ones(n_ab, dtype=np.int64),
    ])
    idx = rng.permutation(n_samples)
    return feats[idx], labels[idx]


def load_breath_data(
    data_dir: str | Path | None = None,
    split: Literal["train", "test"] = "train",
    n_synthetic: int = 3000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load breath feature vectors and binary labels.

    Returns
    -------
    features : np.ndarray of shape [N, FEATURE_DIM]
    labels   : np.ndarray of shape [N], dtype int64 (0=normal, 1=cough/abnormal)
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        if data_dir.exists():
            feats, labels = _load_coughvid_dir(data_dir, split)
            if len(feats) > 0:
                logger.info("Loaded COUGHVID %s: %d samples", split, len(feats))
                return feats, labels

    logger.warning(
        "COUGHVID real data not found in %s — using synthetic data (%d samples).",
        data_dir,
        n_synthetic,
    )
    rng = np.random.default_rng(seed)
    all_feats, all_labels = _synthetic_breath(n_synthetic, rng)
    cut = int(0.70 * n_synthetic)
    if split == "train":
        return all_feats[:cut], all_labels[:cut]
    return all_feats[cut:], all_labels[cut:]


class BreathDataset(Dataset):
    """
    PyTorch Dataset for breath/cough classification.

    Each sample: (feature_tensor, label_tensor)
      feature_tensor: FloatTensor of shape [FEATURE_DIM]
      label_tensor:   LongTensor scalar
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: Literal["train", "test"] = "train",
        n_synthetic: int = 3000,
        seed: int = 42,
    ) -> None:
        feats, labels = load_breath_data(data_dir, split, n_synthetic, seed)
        self.features = torch.from_numpy(feats)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ds = BreathDataset(data_dir=args.data_dir, split=args.split)
    print(f"BreathDataset: {len(ds)} samples, feature dim: {ds.features.shape[1]}")
    x, y = ds[0]
    print(f"  x.shape={x.shape}, label={y.item()}")
