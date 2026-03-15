"""
ECG_CNN — lightweight 1D convolutional neural network for arrhythmia classification.

Trained on MIT-BIH + PTB diagnostic databases.
Input: fixed-length windows of 256 samples per segment, reshaped to [1 × 256].
Output: binary classification (normal vs arrhythmia).

Context-aware variant (ZhiAware):
  The ZhiAware algorithm augments the feature vector extracted by the convolutional
  backbone with peer metadata (confidence + urgency_binary) before the classification
  head: z_t = [features(x_t); μ_t.confidence, μ_t.urgency].
  Pass context_dim=2 to enable this variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGNet(nn.Module):
    """
    Lightweight 1-D CNN for ECG arrhythmia classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: normal / arrhythmia).
    context_dim : int
        Number of peer-context features appended to the CNN feature vector
        before the FC head.  Set to 0 for the local-only (baseline) model and
        to 2 for the ZhiAware context-aware model ([peer_conf, peer_urgency]).
    """

    def __init__(self, num_classes: int = 2, context_dim: int = 0) -> None:
        super().__init__()
        self.context_dim = context_dim

        # --- Convolutional backbone ---
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)

        # After three pool operations on length-256 input:
        # 256 → 128 → 64 → 32  (3 × MaxPool1d(2))
        # channels: 128  =>  feature dim = 128 * 32 = 4096
        self._flat_dim = 128 * 32

        # --- Classification head ---
        self.fc1 = nn.Linear(self._flat_dim + context_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened CNN feature vector (before the FC head)."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.view(x.size(0), -1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape [batch, 1, 256]
        context : Tensor of shape [batch, context_dim] or None.
            Peer metadata (peer_confidence, peer_urgency_binary).
            Required when context_dim > 0, ignored otherwise.
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
        """
        Returns (predicted_class, confidence).
        confidence = max softmax probability (paper eq. for c_t).
        """
        with torch.no_grad():
            logits = self.forward(x, context)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
        return predicted, confidence
