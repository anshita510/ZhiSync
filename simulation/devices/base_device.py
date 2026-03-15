"""
BaseDevice — abstract device class implementing the ZhiAware inference loop.

Implements Algorithm 1 from the paper:
  1. Capture local signal window x_t
  2. Query local buffer for freshest peer metadata μ_t
  3. Augment input: z_t = [x_t; μ_t.confidence, μ_t.urgency]  (if peers available)
  4. Run inference: y_hat_t = f_θ(z_t),  c_t = max(softmax(y_hat_t))
  5. Derive urgency u_t from c_t and threshold τ
  6. Broadcast ZhiTag_t = {device_id, c_t, u_t, t} to peers

Each concrete device subclass provides:
  - `local_model`   : base model  f_θ(x_t)
  - `context_model` : context-aware model  f_θ(z_t)  (ZhiAware variant)
  - `_next_sample()`  : returns the next (x_t, true_label) pair
  - `_build_context_tensor()` : builds the [peer_conf, peer_urgency] tensor
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import MetadataTransport, UdpJsonTransport


@dataclass
class DeviceResult:
    """One inference step result from a device."""
    step: int
    device_id: str
    local_confidence: float
    final_confidence: float
    confidence_gain: float
    urgency: str               # "high" or "low"
    peer_urgency: str          # majority peer urgency at this step
    context_used: bool
    true_label: int
    predicted_label: int
    peer_count: int
    timestamp: float


class BaseDevice(ABC):
    """
    Abstract device that wires together:
      - a PyTorch local model (no peer context)
      - a PyTorch context-aware model (peer context appended to features)
      - a ZhiSyncNode (manages ZhiTag exchange and urgency/staleness logic)

    Parameters
    ----------
    device_id : str
    transport : MetadataTransport
    local_model : nn.Module   — f_θ(x_t)
    context_model : nn.Module — f_θ(z_t) where z_t = [features; peer_conf; peer_urgency]
    urgency_threshold : float — τ (default 0.8)
    staleness_seconds : float — K (default 2.0)
    device : torch.device
    """

    def __init__(
        self,
        device_id: str,
        transport: MetadataTransport,
        local_model: nn.Module,
        context_model: nn.Module,
        urgency_threshold: float = 0.8,
        staleness_seconds: float = 2.0,
        device: torch.device | None = None,
    ) -> None:
        self.device_id = device_id
        self.torch_device = device or torch.device("cpu")

        self.local_model = local_model.to(self.torch_device).eval()
        self.context_model = context_model.to(self.torch_device).eval()

        self.node = ZhiSyncNode(
            options=NodeOptions(
                node_id=device_id,
                urgency_threshold=urgency_threshold,
                staleness_seconds=staleness_seconds,
                metadata_enabled=True,
            ),
            transport=transport,
        )
        self.transport = transport

    # ------------------------------------------------------------------
    # Abstract interface — implement in each concrete device
    # ------------------------------------------------------------------

    @abstractmethod
    def _next_sample(self) -> tuple[torch.Tensor, int]:
        """
        Return (x_t, true_label) for the next inference step.
        x_t is a torch.Tensor already on self.torch_device.
        """

    # ------------------------------------------------------------------
    # ZhiAware inference (Algorithm 1)
    # ------------------------------------------------------------------

    def _build_context_tensor(self, peer_conf: float, peer_urgency: str) -> torch.Tensor:
        """Build [peer_conf, peer_urgency_binary] tensor for context-aware model."""
        urgency_binary = 1.0 if peer_urgency == "high" else 0.0
        return torch.tensor(
            [[peer_conf, urgency_binary]],
            dtype=torch.float32,
            device=self.torch_device,
        )

    def _run_local(self, x: torch.Tensor) -> tuple[int, float]:
        """Run local model. Returns (predicted_class, confidence)."""
        with torch.no_grad():
            logits = self.local_model(x.unsqueeze(0) if x.dim() == 1 else x)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
        return int(pred.item()), float(conf.item())

    def _run_context(self, x: torch.Tensor, context: torch.Tensor) -> tuple[int, float]:
        """Run context-aware model. Returns (predicted_class, confidence)."""
        with torch.no_grad():
            logits = self.context_model(
                x.unsqueeze(0) if x.dim() == 1 else x,
                context,
            )
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
        return int(pred.item()), float(conf.item())

    def step(self, step_idx: int, zhisync_enabled: bool = True) -> DeviceResult:
        """
        Execute one ZhiAware inference step (Algorithm 1).

        Parameters
        ----------
        step_idx : int
        zhisync_enabled : bool
            If False, run in baseline mode (local model only, no peer context).
        """
        x_t, true_label = self._next_sample()
        now_ts = time.time()

        # --- Step 1-2: local inference + fetch peer metadata ---
        pred_local, local_conf = self._run_local(x_t)

        peer_tags = self.node.get_fresh_peer_tags(timestamp=now_ts)
        peer_conf_val = 0.0
        peer_urgency_str = "low"
        context_used = False

        if zhisync_enabled and peer_tags:
            # Use max peer confidence and majority urgency (paper: single peer buffer)
            peer_confs = [t.confidence for t in peer_tags.values()]
            peer_urgencies = [t.urgency for t in peer_tags.values()]
            peer_conf_val = max(peer_confs)
            peer_urgency_str = "high" if peer_urgencies.count("high") > len(peer_urgencies) // 2 else "low"

            # --- Step 3-4: augment input and run context-aware model ---
            context_tensor = self._build_context_tensor(peer_conf_val, peer_urgency_str)
            pred_ctx, ctx_conf = self._run_context(x_t, context_tensor)
            context_used = True
        else:
            ctx_conf = local_conf
            pred_ctx = pred_local

        # --- Step 5-6: derive urgency and broadcast via ZhiSyncNode ---
        decision = self.node.process(
            local_confidence=local_conf,
            context_confidence=ctx_conf if zhisync_enabled else None,
            timestamp=now_ts,
            publish=True,
        )

        return DeviceResult(
            step=step_idx,
            device_id=self.device_id,
            local_confidence=local_conf,
            final_confidence=decision.final_confidence,
            confidence_gain=decision.confidence_gain,
            urgency=decision.urgency,
            peer_urgency=peer_urgency_str,
            context_used=context_used,
            true_label=true_label,
            predicted_label=pred_ctx,
            peer_count=decision.peer_count,
            timestamp=now_ts,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.node.start()

    def stop(self) -> None:
        self.node.stop()
