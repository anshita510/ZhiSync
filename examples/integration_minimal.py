"""
Minimal integration example for existing model inference loops.

This example uses the in-memory transport, so it runs in one process.
Replace `fake_local_confidence` and `fake_context_aware_confidence` with your real model calls.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import InMemoryBus, InMemoryTransport
from zhisync.types import ZhiTag


def fake_local_confidence(rng: random.Random) -> float:
    return max(0.0, min(1.0, 0.65 + rng.uniform(-0.2, 0.2)))


def fake_context_aware_confidence(local_conf: float, peers: dict[str, ZhiTag]) -> float:
    # Placeholder: in your deployment, compute this using your model with peer context as input.
    _ = peers
    return local_conf


def main() -> None:
    bus = InMemoryBus()
    ecg_node = ZhiSyncNode(
        options=NodeOptions(node_id="ECG"),
        transport=InMemoryTransport(node_id="ECG", bus=bus),
    )
    breath_node = ZhiSyncNode(
        options=NodeOptions(node_id="Breath"),
        transport=InMemoryTransport(node_id="Breath", bus=bus),
    )

    ecg_rng = random.Random(11)
    breath_rng = random.Random(17)

    ecg_node.start()
    breath_node.start()
    try:
        for step in range(10):
            ecg_local = fake_local_confidence(ecg_rng)
            breath_local = fake_local_confidence(breath_rng)

            ecg_peers = ecg_node.get_fresh_peer_tags()
            breath_peers = breath_node.get_fresh_peer_tags()
            ecg_context_conf = fake_context_aware_confidence(ecg_local, ecg_peers)
            breath_context_conf = fake_context_aware_confidence(breath_local, breath_peers)

            ecg_decision = ecg_node.process(ecg_local, context_confidence=ecg_context_conf)
            breath_decision = breath_node.process(breath_local, context_confidence=breath_context_conf)

            print(
                f"step={step:02d} "
                f"ECG(local={ecg_decision.local_confidence:.3f},final={ecg_decision.final_confidence:.3f},"
                f"gain={ecg_decision.confidence_gain:.3f},urg={ecg_decision.urgency}) "
                f"Breath(local={breath_decision.local_confidence:.3f},final={breath_decision.final_confidence:.3f},"
                f"gain={breath_decision.confidence_gain:.3f},urg={breath_decision.urgency})"
            )
            time.sleep(0.2)
    finally:
        ecg_node.stop()
        breath_node.stop()


if __name__ == "__main__":
    main()
