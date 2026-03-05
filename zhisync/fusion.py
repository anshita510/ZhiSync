from .types import ZhiTag


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def derive_urgency(confidence: float, threshold: float) -> str:
    return "high" if confidence >= threshold else "low"


def fresh_peer_tags(
    peer_tags: dict[str, ZhiTag], now_ts: float, staleness_seconds: float
) -> list[ZhiTag]:
    return [tag for tag in peer_tags.values() if (now_ts - tag.timestamp) <= staleness_seconds]


def summarize_peer_context(peers: list[ZhiTag]) -> dict[str, float]:
    if not peers:
        return {
            "peer_count": 0.0,
            "peer_high_count": 0.0,
            "peer_max_confidence": 0.0,
            "peer_mean_confidence": 0.0,
        }
    peer_max = max(tag.confidence for tag in peers)
    peer_mean = sum(tag.confidence for tag in peers) / float(len(peers))
    peer_high_count = sum(1 for tag in peers if tag.urgency == "high")
    return {
        "peer_count": float(len(peers)),
        "peer_high_count": float(peer_high_count),
        "peer_max_confidence": peer_max,
        "peer_mean_confidence": peer_mean,
    }
