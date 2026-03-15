"""ECG device — wearable ECG monitor (d_ecg) using ECGNet."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from simulation.models.ecg_model import ECGNet
from simulation.datasets.ecg_dataset import ECGDataset
from zhisync.transport import MetadataTransport

from .base_device import BaseDevice


class ECGDevice(BaseDevice):
    """
    Simulates a wearable ECG monitor (d_ecg).

    Local model:   ECGNet(context_dim=0) — baseline, no peer input.
    Context model: ECGNet(context_dim=2) — ZhiAware, appends [peer_conf, peer_urgency].

    Paper: "lightweight 1D CNN (ECG_CNN), final output = normal vs arrhythmia."
    """

    def __init__(
        self,
        transport: MetadataTransport,
        local_model: ECGNet,
        context_model: ECGNet,
        data_dir: str | None = None,
        split: str = "test",
        urgency_threshold: float = 0.8,
        staleness_seconds: float = 2.0,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            device_id="ECG",
            transport=transport,
            local_model=local_model,
            context_model=context_model,
            urgency_threshold=urgency_threshold,
            staleness_seconds=staleness_seconds,
            device=device,
        )
        dataset = ECGDataset(data_dir=data_dir, split=split, seed=seed)
        # Infinite cycling loader (one sample at a time)
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
