"""
ZhiSync scalability test.

Measures each node's average total byte rate as the number of peers increases,
under two ZhiTag broadcast frequencies (2 Hz and 5 Hz).

N = 4..10 nodes: 3 modality devices (ECG, Breath, Motion) + (N-3) simulated
peer nodes that participate in ZhiTag exchange without running a full model.
Metric: per-node total byte rate (kB/s) = (bytes_sent + bytes_received) /
        runtime_s / 1024.

Coalescing is enabled at each frequency by setting coalesce_ms = 1000/freq_hz,
which reduces redundant sends at higher broadcast rates.

Usage::

  python -m simulation.scalability_test
  python -m simulation.scalability_test --duration-s 30 --runs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import torch

from simulation.devices.ecg_device import ECGDevice
from simulation.devices.breath_device import BreathDevice
from simulation.devices.motion_device import MotionDevice
from simulation.metrics import WallTimer
from simulation.train import load_models, train_all
from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import UdpJsonTransport, UdpPeer

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
BASE_PORT = 8100        # scalability test uses a separate port range
MODEL_DIR = "simulation/saved_models"

FREQS_HZ = [2.0, 5.0]
NODE_COUNTS = list(range(4, 11))   # 4, 5, 6, 7, 8, 9, 10
DEFAULT_DURATION_S = 30            # seconds per experiment point
DEFAULT_RUNS = 3


# ---------------------------------------------------------------------------
# Simulated peer node — lightweight ZhiTag broadcaster for scalability tests
# ---------------------------------------------------------------------------

class SimulatedNode:
    """
    Lightweight peer node that broadcasts ZhiTags at a fixed frequency.

    Represents additional IoT devices participating in the ZhiSync mesh
    during scalability experiments.  Each node runs its own ZhiSyncNode
    and UDP transport, contributing realistic network load without
    executing a full modality-specific model.
    """

    def __init__(
        self,
        node_id: str,
        port: int,
        peers: list[UdpPeer],
        freq_hz: float,
        coalesce_ms: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.node_id = node_id
        self.freq_hz = freq_hz
        self._transport = UdpJsonTransport(
            node_id=node_id,
            bind_host=HOST,
            bind_port=port,
            peers=peers,
            coalesce_ms=coalesce_ms,
        )
        self._node = ZhiSyncNode(
            options=NodeOptions(node_id=node_id, urgency_threshold=0.8),
            transport=self._transport,
        )
        self._rng = random.Random(seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the transport and begin broadcasting ZhiTags."""
        self._node.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop broadcasting and shut down the transport."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._node.stop()

    def get_stats(self) -> dict[str, int]:
        """Return communication counters from the underlying transport."""
        return self._transport.get_stats()

    def _loop(self) -> None:
        interval = 1.0 / self.freq_hz
        while not self._stop.is_set():
            conf = self._rng.uniform(0.3, 0.95)
            self._node.process(local_confidence=conf)
            time.sleep(interval)


# ---------------------------------------------------------------------------
# One scalability data point
# ---------------------------------------------------------------------------

def _assign_ports(n_nodes: int, base_port: int) -> dict[str, int]:
    """Return a mapping of node_id → UDP port for all N nodes."""
    names = ["ECG", "Breath", "Motion"] + [f"Peer{i}" for i in range(1, n_nodes - 2)]
    return {name: base_port + i for i, name in enumerate(names)}


def _build_peer_list(node_id: str, port_map: dict[str, int]) -> list[UdpPeer]:
    """Return UdpPeer entries for every node in port_map except node_id."""
    return [UdpPeer(HOST, port) for name, port in port_map.items() if name != node_id]


def run_scalability_point(
    n_nodes: int,
    freq_hz: float,
    duration_s: float,
    models: dict[str, torch.nn.Module],
    data_dir: str | None,
    device: torch.device,
    base_port: int,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run one scalability experiment point for a given node count and broadcast frequency.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the mesh (3 modality devices + simulated peers).
    freq_hz : float
        ZhiTag broadcast frequency in Hz.
    duration_s : float
        Duration of the experiment in seconds.
    models : dict
        Loaded model weights (from load_models).
    data_dir : str or None
        Path to real datasets; None uses synthetic data.
    device : torch.device
    base_port : int
        Starting UDP port for this experiment point.
    seed : int

    Returns
    -------
    dict with per-node byte rates and coalescing statistics.
    """
    coalesce_ms = 1000.0 / freq_hz
    port_map = _assign_ports(n_nodes, base_port)
    n_steps = max(1, int(duration_s * freq_hz))
    sleep_s = 1.0 / freq_hz

    # Build real modality devices
    ecg_transport = UdpJsonTransport(
        node_id="ECG", bind_host=HOST, bind_port=port_map["ECG"],
        peers=_build_peer_list("ECG", port_map),
        coalesce_ms=coalesce_ms,
    )
    breath_transport = UdpJsonTransport(
        node_id="Breath", bind_host=HOST, bind_port=port_map["Breath"],
        peers=_build_peer_list("Breath", port_map),
        coalesce_ms=coalesce_ms,
    )
    motion_transport = UdpJsonTransport(
        node_id="Motion", bind_host=HOST, bind_port=port_map["Motion"],
        peers=_build_peer_list("Motion", port_map),
        coalesce_ms=coalesce_ms,
    )

    ecg_dev = ECGDevice(
        transport=ecg_transport, local_model=models["ecg_local"],
        context_model=models["ecg_context"], data_dir=data_dir,
        split="test", device=device, seed=seed,
    )
    breath_dev = BreathDevice(
        transport=breath_transport, local_model=models["breath_local"],
        context_model=models["breath_context"], data_dir=data_dir,
        split="test", device=device, seed=seed,
    )
    motion_dev = MotionDevice(
        transport=motion_transport, local_model=models["motion_local"],
        context_model=models["motion_context"], data_dir=data_dir,
        split="test", device=device, seed=seed,
    )

    # Build simulated peer nodes for scalability beyond the 3 modality devices
    peer_nodes: list[SimulatedNode] = []
    for i in range(1, n_nodes - 2):
        peer_id = f"Peer{i}"
        peer = SimulatedNode(
            node_id=peer_id,
            port=port_map[peer_id],
            peers=_build_peer_list(peer_id, port_map),
            freq_hz=freq_hz,
            coalesce_ms=coalesce_ms,
            seed=seed + i,
        )
        peer_nodes.append(peer)

    timer = WallTimer()
    stop_event = threading.Event()
    all_results: list = []

    def _run_device(dev, steps, step_sleep, results):
        """Run a modality device for the given number of inference steps."""
        dev.start()
        try:
            for step in range(steps):
                if stop_event.is_set():
                    break
                results.append(dev.step(step, zhisync_enabled=True))
                time.sleep(step_sleep)
        finally:
            dev.stop()

    threads = [
        threading.Thread(target=_run_device, args=(ecg_dev, n_steps, sleep_s, all_results), daemon=True),
        threading.Thread(target=_run_device, args=(breath_dev, n_steps, sleep_s, all_results), daemon=True),
        threading.Thread(target=_run_device, args=(motion_dev, n_steps, sleep_s, all_results), daemon=True),
    ]

    timer.start()
    for peer in peer_nodes:
        peer.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for peer in peer_nodes:
        peer.stop()
    runtime_s = timer.stop()

    # Collect per-node byte rates
    node_stats: list[dict] = []
    for node_id, transport in [
        ("ECG", ecg_transport),
        ("Breath", breath_transport),
        ("Motion", motion_transport),
    ]:
        raw = transport.get_stats()
        total_bytes = raw["bytes_sent"] + raw["bytes_received"]
        rate_kbps = (total_bytes / 1024) / max(runtime_s, 1e-6)
        node_stats.append({
            "node_id": node_id,
            "bytes_sent": raw["bytes_sent"],
            "bytes_received": raw["bytes_received"],
            "rate_kbps": rate_kbps,
            "coalesce_suppressed": raw["coalesce_suppressed"],
            "stale_drops": raw["stale_drops"],
        })

    for peer in peer_nodes:
        raw = peer.get_stats()
        total_bytes = raw["bytes_sent"] + raw["bytes_received"]
        rate_kbps = (total_bytes / 1024) / max(runtime_s, 1e-6)
        node_stats.append({
            "node_id": peer.node_id,
            "bytes_sent": raw["bytes_sent"],
            "bytes_received": raw["bytes_received"],
            "rate_kbps": rate_kbps,
            "coalesce_suppressed": raw["coalesce_suppressed"],
            "stale_drops": raw["stale_drops"],
        })

    avg_rate_kbps = sum(s["rate_kbps"] for s in node_stats) / len(node_stats) if node_stats else 0.0
    total_coalesce = sum(s["coalesce_suppressed"] for s in node_stats)
    avg_coalesce_per_node_per_s = total_coalesce / n_nodes / max(runtime_s, 1e-6)

    return {
        "n_nodes": n_nodes,
        "freq_hz": freq_hz,
        "runtime_s": runtime_s,
        "avg_rate_kbps": avg_rate_kbps,
        "avg_coalesce_per_node_per_s": avg_coalesce_per_node_per_s,
        "node_stats": node_stats,
    }


# ---------------------------------------------------------------------------
# Full scalability sweep
# ---------------------------------------------------------------------------

def run_scalability_experiment(
    node_counts: list[int] = NODE_COUNTS,
    freqs_hz: list[float] = FREQS_HZ,
    duration_s: float = DEFAULT_DURATION_S,
    n_runs: int = DEFAULT_RUNS,
    data_dir: str | None = None,
    model_dir: str = MODEL_DIR,
    output_dir: str = "simulation/results",
    device_str: str = "cpu",
    retrain: bool = False,
) -> None:
    """
    Run the full scalability sweep over all node counts and frequencies.

    Results are saved to output_dir/scalability_results.json and printed
    as a summary table matching Figure 7 in the paper.
    """
    device = torch.device(device_str)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if retrain or not Path(model_dir).exists() or not list(Path(model_dir).glob("*.pt")):
        train_all(data_dir=data_dir, model_dir=model_dir, device_str=device_str)
    models = load_models(model_dir=model_dir, device=device)

    all_results: list[dict] = []

    for freq_hz in freqs_hz:
        for n_nodes in node_counts:
            run_results: list[dict] = []
            for run_idx in range(n_runs):
                base_port = BASE_PORT + run_idx * 100 + node_counts.index(n_nodes) * 20
                logger.info(
                    "Scalability: N=%d, freq=%.0fHz, run %d/%d",
                    n_nodes, freq_hz, run_idx + 1, n_runs,
                )
                result = run_scalability_point(
                    n_nodes=n_nodes,
                    freq_hz=freq_hz,
                    duration_s=duration_s,
                    models=models,
                    data_dir=data_dir,
                    device=device,
                    base_port=base_port,
                    seed=run_idx,
                )
                run_results.append(result)
                time.sleep(0.5)

            avg_rate = sum(r["avg_rate_kbps"] for r in run_results) / len(run_results)
            avg_coalesce = sum(r["avg_coalesce_per_node_per_s"] for r in run_results) / len(run_results)
            point = {
                "n_nodes": n_nodes,
                "freq_hz": freq_hz,
                "avg_rate_kbps_mean": round(avg_rate, 4),
                "avg_coalesce_per_node_per_s_mean": round(avg_coalesce, 4),
                "runs": run_results,
            }
            all_results.append(point)
            print(
                f"  N={n_nodes:2d}, {freq_hz:.0f}Hz → "
                f"avg_rate={avg_rate:.3f} kB/s, "
                f"coalesce/node/s={avg_coalesce:.2f}"
            )

    out_path = out_dir / "scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Scalability results saved to %s", out_path)

    _print_scalability_table(all_results)


def _print_scalability_table(results: list[dict]) -> None:
    """Print a per-node byte rate summary table matching Figure 7 of the paper."""
    print("\n" + "=" * 60)
    print("Scalability Results (avg per-node kB/s)")
    print(f"{'N':>4}  {'2 Hz':>10}  {'5 Hz':>10}  {'Coalesce@5Hz':>14}")
    print("-" * 60)

    by_n: dict[int, dict[float, dict]] = {}
    for r in results:
        by_n.setdefault(r["n_nodes"], {})[r["freq_hz"]] = r

    for n in sorted(by_n.keys()):
        rate_2 = by_n[n].get(2.0, {}).get("avg_rate_kbps_mean", 0.0)
        rate_5 = by_n[n].get(5.0, {}).get("avg_rate_kbps_mean", 0.0)
        coal_5 = by_n[n].get(5.0, {}).get("avg_coalesce_per_node_per_s_mean", 0.0)
        print(f"  {n:2d}  {rate_2:10.3f}  {rate_5:10.3f}  {coal_5:14.2f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ZhiSync scalability experiment.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--output-dir", default="simulation/results")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    run_scalability_experiment(
        duration_s=args.duration_s,
        n_runs=args.runs,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        retrain=args.retrain,
    )
