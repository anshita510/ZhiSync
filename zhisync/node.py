import time
from dataclasses import dataclass

from .fusion import clamp01, derive_urgency, fresh_peer_tags, summarize_peer_context
from .transport import MetadataTransport, NullTransport
from .types import InferenceDecision, ZhiTag


@dataclass(frozen=True)
class NodeOptions:
    node_id: str
    urgency_threshold: float = 0.8
    staleness_seconds: float = 2.0
    metadata_enabled: bool = True


class ZhiSyncNode:
    """
    Integratable runtime object.

    Call `process(local_confidence=...)` on each local model inference.
    It returns final confidence + urgency and publishes a ZhiTag to peers.
    """

    def __init__(
        self,
        options: NodeOptions,
        transport: MetadataTransport | None = None,
    ) -> None:
        self.options = options
        self.transport = transport or NullTransport()
        self._seq = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.transport.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.transport.stop()
        self._started = False

    def process(
        self,
        local_confidence: float,
        *,
        context_confidence: float | None = None,
        timestamp: float | None = None,
        publish: bool = True,
    ) -> InferenceDecision:
        if not self._started:
            raise RuntimeError("Node not started. Call start() before process().")

        now_ts = float(timestamp if timestamp is not None else time.time())
        local_conf = clamp01(float(local_confidence))
        peer_map = self.transport.get_latest_peer_tags()
        peers = fresh_peer_tags(
            peer_tags=peer_map,
            now_ts=now_ts,
            staleness_seconds=self.options.staleness_seconds,
        )
        stale_count = len(peer_map) - len(peers)
        if stale_count > 0:
            self.transport.record_stale_drop(stale_count)

        context = summarize_peer_context(peers)
        if self.options.metadata_enabled and context_confidence is not None:
            final_conf = clamp01(float(context_confidence))
        else:
            # Paper-aligned fallback: local-only confidence if context-aware model output is not provided.
            final_conf = local_conf

        urgency = derive_urgency(final_conf, self.options.urgency_threshold)
        self._seq += 1
        tag = ZhiTag(
            device_id=self.options.node_id,
            confidence=final_conf,
            urgency=urgency,
            timestamp=now_ts,
            seq=self._seq,
        )
        if publish:
            self.transport.broadcast(tag)

        return InferenceDecision(
            node_id=self.options.node_id,
            local_confidence=local_conf,
            final_confidence=final_conf,
            confidence_gain=final_conf - local_conf,
            urgency=urgency,
            context_used=(self.options.metadata_enabled and len(peers) > 0 and context_confidence is not None),
            peer_count=int(context["peer_count"]),
            peer_high_count=int(context["peer_high_count"]),
            peer_max_confidence=float(context["peer_max_confidence"]),
            peer_mean_confidence=float(context["peer_mean_confidence"]),
            timestamp=now_ts,
            seq=self._seq,
        )

    def get_fresh_peer_tags(self, *, timestamp: float | None = None) -> dict[str, ZhiTag]:
        if not self._started:
            raise RuntimeError("Node not started. Call start() before reading peer metadata.")
        now_ts = float(timestamp if timestamp is not None else time.time())
        peer_map = self.transport.get_latest_peer_tags()
        peers = fresh_peer_tags(
            peer_tags=peer_map,
            now_ts=now_ts,
            staleness_seconds=self.options.staleness_seconds,
        )
        return {tag.device_id: tag for tag in peers}
