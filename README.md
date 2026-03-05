# ZhiSync SDK

ZhiSync is an integration SDK for metadata-driven collaborative inference across decentralized edge devices.

Use ZhiSync when you have multiple edge devices that should coordinate in real time without sending raw sensor data to a central server.

Typical reasons:
- improve decision stability for uncertain local predictions
- keep communication overhead low (metadata only)
- preserve privacy by avoiding raw signal sharing
- add cross-device context without retraining all models together

## What Is Included

- `zhisync/node.py`: `ZhiSyncNode` runtime for per-inference fusion
- `zhisync/transport.py`: pluggable metadata transports
- `zhisync/fusion.py`: urgency and peer-context utilities
- `zhisync/types.py`: `ZhiTag` and `InferenceDecision` types
- `examples/integration_minimal.py`: in-process integration example
- `examples/udp_node.py`: one-node UDP runtime example
- `scripts/run_udp_demo.sh`: 3-node UDP demo launcher

## Install

```bash
python3 -m pip install -e .
```

## How To Use

```python
from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import UdpJsonTransport, UdpPeer

transport = UdpJsonTransport(
    node_id="ECG",
    bind_host="0.0.0.0",
    bind_port=6001,
    peers=[UdpPeer("10.0.0.12", 6003), UdpPeer("10.0.0.13", 6005)],
)

node = ZhiSyncNode(
    options=NodeOptions(
        node_id="ECG",
        urgency_threshold=0.8,
        staleness_seconds=2.0,
        metadata_enabled=True,
    ),
    transport=transport,
)

node.start()
try:
    local_conf = 0.74  # from local-only model
    peer_tags = node.get_fresh_peer_tags()
    # Compute this from your context-aware model that uses local input + peer metadata.
    context_conf = local_conf
    decision = node.process(local_confidence=local_conf, context_confidence=context_conf)
    print(decision.final_confidence, decision.urgency, decision.confidence_gain)
finally:
    node.stop()
```

## Local Examples

```bash
python3 examples/integration_minimal.py
./scripts/run_udp_demo.sh
```

`run_udp_demo.sh` writes logs to:
- `runs/udp_ecg.log`
- `runs/udp_motion.log`
- `runs/udp_breath.log`

## Integration Contract

- Input: local model confidence (`0.0` to `1.0`) and optional context-aware confidence
- Output: `InferenceDecision` with fused confidence and urgency
- Side effect: publishes/receives lightweight peer metadata (`ZhiTag`)

## Notes

- This repository is focused only on the reusable SDK.
