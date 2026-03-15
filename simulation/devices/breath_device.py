"""Breath device — stationary breath analyzer (d_br) using CovidCoughNet."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from simulation.models.breath_model import CovidCoughNet
from simulation.datasets.breath_dataset import BreathDataset
from zhisync.transport import MetadataTransport

from .base_device import BaseDevice


class BreathDevice(BaseDevice):
    """
    Simulates a stationary breath/cough analyzer (d_br).

    Local model:   CovidCoughNet(context_dim=0) — baseline.
    Context model: CovidCoughNet(context_dim=2) — ZhiAware.

    Paper: "CovidCoughNet combining Inception + DeepConvNet blocks."
    """

    def __init__(
        self,
        transport: MetadataTransport,
        local_model: CovidCoughNet,
        context_model: CovidCoughNet,
        data_dir: str | None = None,
        split: str = "test",
        urgency_threshold: float = 0.8,
        staleness_seconds: float = 2.0,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            device_id="Breath",
            transport=transport,
            local_model=local_model,
            context_model=context_model,
            urgency_threshold=urgency_threshold,
            staleness_seconds=staleness_seconds,
            device=device,
        )
        dataset = BreathDataset(data_dir=data_dir, split=split, seed=seed)
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
