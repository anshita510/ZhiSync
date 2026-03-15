"""
MotionMLP — fully connected multilayer perceptron for human activity recognition.

Trained on the WISDM Human Activity Recognition dataset.
Input: 561-dimensional statistical and frequency-domain features from
accelerometer readings.
Output: 6 activity classes (walking, jogging, sitting, standing,
climbing stairs, lying).

Context-aware variant (ZhiAware):
  Peer metadata appended to the input feature vector before the first FC layer:
  z_t = [x_t; μ_t.confidence, μ_t.urgency_binary].
  Pass context_dim=2 to enable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# WISDM: 561 statistical + frequency features per time window
FEATURE_DIM = 561
NUM_CLASSES = 6


class MotionMLP(nn.Module):
    """
    Fully-connected MLP for accelerometer-based activity recognition.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6).
    context_dim : int
        Peer-context feature dimensions prepended to the input vector
        (0 = baseline, 2 = ZhiAware context-aware).
    feature_dim : int
        Input feature dimensionality (default 561 for WISDM).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        context_dim: int = 0,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        in_dim = feature_dim + context_dim

        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

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
            For ZhiAware: [peer_confidence, peer_urgency_binary].
            Appended to raw features as per Algorithm 1.
        """
        if self.context_dim > 0 and context is not None:
            x = torch.cat([x, context], dim=1)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

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
