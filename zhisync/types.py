from dataclasses import asdict, dataclass
from typing import Literal

Urgency = Literal["high", "low"]


@dataclass(frozen=True)
class ZhiTag:
    """Lightweight metadata shared between peers."""

    device_id: str
    confidence: float
    urgency: Urgency
    timestamp: float
    seq: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, object]) -> "ZhiTag":
        urgency_raw = str(data["urgency"]).lower()
        urgency: Urgency = "high" if urgency_raw == "high" else "low"
        return ZhiTag(
            device_id=str(data["device_id"]),
            confidence=float(data["confidence"]),
            urgency=urgency,
            timestamp=float(data["timestamp"]),
            seq=int(data["seq"]),
        )


@dataclass(frozen=True)
class InferenceDecision:
    """Final decision returned to the caller's inference loop."""

    node_id: str
    local_confidence: float
    final_confidence: float
    confidence_gain: float
    urgency: Urgency
    context_used: bool
    peer_count: int
    peer_high_count: int
    peer_max_confidence: float
    peer_mean_confidence: float
    timestamp: float
    seq: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
