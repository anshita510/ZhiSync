"""
CovidCoughNet — Inception + DeepConvNet architecture for breath/cough classification.

Trained on the COUGHVID dataset.
Features: MFCC + Chroma + Spectral Contrast (59-dimensional per sample after
time-averaging).
Architecture: Inception feature extractor with DeepConvNet blocks.
Output: binary classification (normal breathing vs cough/abnormal).

Context-aware variant (ZhiAware):
  Peer metadata is appended to the feature vector before the classification
  head.  Pass context_dim=2 to enable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of hand-crafted audio features extracted per sample.
# MFCC: 40, Chroma: 12, Spectral Contrast: 7  →  total 59
FEATURE_DIM = 59


class _Inception1DBlock(nn.Module):
    """Parallel 1-D convolutions with kernel sizes 1, 3, 5 — concatenated."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Distribute output channels roughly evenly across three paths
        c1 = out_ch // 3
        c3 = out_ch // 3
        c5 = out_ch - c1 - c3
        self.path1 = nn.Conv1d(in_ch, c1, kernel_size=1)
        self.path3 = nn.Conv1d(in_ch, c3, kernel_size=3, padding=1)
        self.path5 = nn.Conv1d(in_ch, c5, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(torch.cat([self.path1(x), self.path3(x), self.path5(x)], dim=1)))


class _DeepConvBlock(nn.Module):
    """Two stacked 1-D convolutions with residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CovidCoughNet(nn.Module):
    """
    Inception + DeepConvNet model for breath/cough audio classification.

    The model operates on fixed-length feature vectors (not raw waveforms).
    Each sample is represented as a 59-dimensional vector derived from
    MFCC, Chroma, and Spectral Contrast features averaged over time frames.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: normal breath / abnormal cough).
    context_dim : int
        Peer-context feature dimensions appended before the classification
        head (0 = baseline, 2 = ZhiAware context-aware).
    feature_dim : int
        Input feature dimensionality (default 59).
    """

    def __init__(
        self,
        num_classes: int = 2,
        context_dim: int = 0,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim

        # Reshape [batch, feature_dim] → [batch, 1, feature_dim] for 1-D conv
        self.inception = _Inception1DBlock(1, 64)            # [batch, 64, feature_dim]
        self.deep_conv1 = _DeepConvBlock(64)
        self.deep_conv2 = _DeepConvBlock(64)
        self.pool = nn.AdaptiveAvgPool1d(1)                  # [batch, 64, 1]

        self._flat_dim = 64

        # Classification head
        self.fc1 = nn.Linear(self._flat_dim + context_dim, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, feature_dim] → returns [batch, 64] feature vector."""
        x = x.unsqueeze(1)                       # [batch, 1, feature_dim]
        x = self.inception(x)
        x = self.deep_conv1(x)
        x = self.deep_conv2(x)
        x = self.pool(x).squeeze(-1)             # [batch, 64]
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape [batch, feature_dim]
        context : Tensor of shape [batch, context_dim] or None.
        """
        features = self.extract_features(x)
        if self.context_dim > 0 and context is not None:
            features = torch.cat([features, context], dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(features))))

    def predict_confidence(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x, context)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
        return predicted, confidence
