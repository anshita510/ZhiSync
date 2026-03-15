"""Motion device — wearable motion sensor (d_mtn) using MotionMLP."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from simulation.models.motion_model import MotionMLP
from simulation.datasets.motion_dataset import MotionDataset
from zhisync.transport import MetadataTransport

from .base_device import BaseDevice


class MotionDevice(BaseDevice):
    """
    Simulates a wearable motion / accelerometer sensor (d_mtn).

    Local model:   MotionMLP(context_dim=0) — baseline.
    Context model: MotionMLP(context_dim=2) — ZhiAware.

    Paper: "fully connected MLP (MotionMLP), 6-class activity recognition."
    The binary urgency signal from motion is still meaningful: e.g.
    jogging or stair-climbing → higher urgency if combined with ECG alert.
    """

    def __init__(
        self,
        transport: MetadataTransport,
        local_model: MotionMLP,
        context_model: MotionMLP,
        data_dir: str | None = None,
        split: str = "test",
        urgency_threshold: float = 0.8,
        staleness_seconds: float = 2.0,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            device_id="Motion",
            transport=transport,
            local_model=local_model,
            context_model=context_model,
            urgency_threshold=urgency_threshold,
            staleness_seconds=staleness_seconds,
            device=device,
        )
        dataset = MotionDataset(data_dir=data_dir, split=split, seed=seed)
        self._loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        self._iter = iter(self._loader)

    def _next_sample(self) -> tuple[torch.Tensor, int]:
        try:
            x, y = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            x, y = next(self._iter)
        return x.to(self.torch_device), int(y.item())
