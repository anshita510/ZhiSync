"""
Run one ZhiSync node over UDP.

Open multiple terminals and run this script for each node to form a mesh.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import UdpJsonTransport, UdpPeer


def _parse_peers(peers_csv: str) -> list[UdpPeer]:
    if not peers_csv.strip():
        return []
    peers: list[UdpPeer] = []
    for item in peers_csv.split(","):
        host_port = item.strip()
        if not host_port:
            continue
        host, port_str = host_port.split(":")
        peers.append(UdpPeer(host=host, port=int(port_str)))
    return peers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single ZhiSync UDP node.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--peers", default="", help="Comma-separated host:port list.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--base-confidence", type=float, default=0.65)
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    peers = _parse_peers(args.peers)
    transport = UdpJsonTransport(
        node_id=args.node_id,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        peers=peers,
    )
    node = ZhiSyncNode(
        options=NodeOptions(node_id=args.node_id),
        transport=transport,
    )

    node_offset = sum(ord(ch) for ch in args.node_id) % 1000
    rng = random.Random(args.seed + node_offset)
    node.start()
    try:
        for step in range(args.steps):
            local = max(0.0, min(1.0, args.base_confidence + rng.uniform(-args.volatility, args.volatility)))
            decision = node.process(local_confidence=local, context_confidence=local)
            print(
                f"[{args.node_id}] step={step:03d} local={decision.local_confidence:.3f} "
                f"final={decision.final_confidence:.3f} gain={decision.confidence_gain:.3f} "
                f"urg={decision.urgency} peers={decision.peer_count}"
            )
            time.sleep(args.sleep_seconds)
    finally:
        node.stop()


if __name__ == "__main__":
    main()
