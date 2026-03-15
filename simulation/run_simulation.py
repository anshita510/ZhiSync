"""
ZhiSync main simulation runner.

Runs 10 repeated experiments under two conditions — With ZhiSync
(metadata-enabled) and Without ZhiSync (baseline) — capturing confidence
gain, urgency-aware behavior, CPU utilization, RAM usage, runtime, and
communication bytes.

Three devices (ECG, Breath, Motion) run as independent threads communicating
via UDP, reproducing the multi-device deployment described in the paper.

Usage::

  # Train models first (uses synthetic data if real datasets absent):
  python -m simulation.train

  # Run simulation (10 runs × 2 conditions):
  python -m simulation.run_simulation

  # With real datasets:
  python -m simulation.run_simulation --data-dir /path/to/datasets

  # Quick test (fewer steps):
  python -m simulation.run_simulation --steps 50 --runs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from simulation.devices.ecg_device import ECGDevice
from simulation.devices.breath_device import BreathDevice
from simulation.devices.motion_device import MotionDevice
from simulation.devices.base_device import DeviceResult
from simulation.metrics import ResourceSampler, WallTimer, CommStats, aggregate_run_stats
from simulation.train import load_models, train_all
from zhisync.transport import UdpJsonTransport, UdpPeer

logger = logging.getLogger(__name__)

# UDP ports matching the paper's 3-device setup
PORTS = {"ECG": 7001, "Breath": 7005, "Motion": 7003}
HOST = "127.0.0.1"

# Paper: 10 repeated experiments
DEFAULT_RUNS = 10
# Steps per device per run — paper ran ~360 s at ~1 step/s = ~360 steps
DEFAULT_STEPS = 360
DEFAULT_SLEEP_S = 1.0   # 1 Hz inference rate
MODEL_DIR = "simulation/saved_models"


# ---------------------------------------------------------------------------
# Device thread runner
# ---------------------------------------------------------------------------

def _run_device_thread(
    device: ECGDevice | BreathDevice | MotionDevice,
    n_steps: int,
    sleep_s: float,
    zhisync_enabled: bool,
    results: list[DeviceResult],
    stop_event: threading.Event,
) -> None:
    """Run a device for n_steps inference cycles and append results to the shared list."""
    device.start()
    try:
        for step in range(n_steps):
            if stop_event.is_set():
                break
            result = device.step(step, zhisync_enabled=zhisync_enabled)
            results.append(result)
            time.sleep(sleep_s)
    finally:
        device.stop()


# ---------------------------------------------------------------------------
# Transport factory
# ---------------------------------------------------------------------------

def _make_transport(node_id: str, port_offset: int = 0) -> UdpJsonTransport:
    """Create a UDP transport for one device with peers set to the other two."""
    peers = [
        UdpPeer(HOST, port + port_offset)
        for name, port in PORTS.items()
        if name != node_id
    ]
    return UdpJsonTransport(
        node_id=node_id,
        bind_host=HOST,
        bind_port=PORTS[node_id] + port_offset,
        peers=peers,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one_experiment(
    models: dict[str, torch.nn.Module],
    n_steps: int,
    sleep_s: float,
    zhisync_enabled: bool,
    data_dir: str | None,
    device: torch.device,
    port_offset: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run one experiment with all three devices in separate threads.
    Returns a dict of metrics for this run.
    """
    mode_label = "with_zhisync" if zhisync_enabled else "without_zhisync"
    logger.info("Starting %s run  (steps=%d, sleep=%.2fs)", mode_label, n_steps, sleep_s)

    # Build devices
    ecg_transport = _make_transport("ECG", port_offset)
    breath_transport = _make_transport("Breath", port_offset)
    motion_transport = _make_transport("Motion", port_offset)

    ecg_dev = ECGDevice(
        transport=ecg_transport,
        local_model=models["ecg_local"],
        context_model=models["ecg_context"],
        data_dir=data_dir,
        split="test",
        device=device,
        seed=seed,
    )
    breath_dev = BreathDevice(
        transport=breath_transport,
        local_model=models["breath_local"],
        context_model=models["breath_context"],
        data_dir=data_dir,
        split="test",
        device=device,
        seed=seed,
    )
    motion_dev = MotionDevice(
        transport=motion_transport,
        local_model=models["motion_local"],
        context_model=models["motion_context"],
        data_dir=data_dir,
        split="test",
        device=device,
        seed=seed,
    )

    ecg_results: list[DeviceResult] = []
    breath_results: list[DeviceResult] = []
    motion_results: list[DeviceResult] = []
    stop_event = threading.Event()

    # Start resource sampler and wall timer
    sampler = ResourceSampler(pid=os.getpid(), interval_s=1.0)
    timer = WallTimer()

    sampler.start()
    timer.start()

    # Launch device threads
    threads = [
        threading.Thread(
            target=_run_device_thread,
            args=(ecg_dev, n_steps, sleep_s, zhisync_enabled, ecg_results, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=_run_device_thread,
            args=(breath_dev, n_steps, sleep_s, zhisync_enabled, breath_results, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=_run_device_thread,
            args=(motion_dev, n_steps, sleep_s, zhisync_enabled, motion_results, stop_event),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    runtime_s = timer.stop()
    resource_stats = sampler.stop()

    # Collect communication stats
    def _comm_stats(dev_id: str, transport: UdpJsonTransport) -> CommStats:
        """Build a CommStats snapshot from a transport's counters."""
        raw = transport.get_stats()
        return CommStats(
            device_id=dev_id,
            total_bytes_sent=raw["bytes_sent"],
            total_bytes_received=raw["bytes_received"],
            total_packets_sent=raw["packets_sent"],
            total_packets_received=raw["packets_received"],
            coalesce_suppressed=raw["coalesce_suppressed"],
            stale_drops=raw["stale_drops"],
            runtime_s=runtime_s,
        )

    ecg_comm = _comm_stats("ECG", ecg_transport)
    breath_comm = _comm_stats("Breath", breath_transport)
    motion_comm = _comm_stats("Motion", motion_transport)

    all_results = ecg_results + breath_results + motion_results

    # --- Aggregate confidence gain stats ---
    gains = [r.confidence_gain for r in all_results]
    high_urgency_gains = [
        r.confidence_gain for r in all_results if r.peer_urgency == "high"
    ]
    low_urgency_gains = [
        r.confidence_gain for r in all_results if r.peer_urgency == "low"
    ]
    positive_gain_pct = (
        sum(1 for g in gains if g > 0) / len(gains) * 100 if gains else 0.0
    )

    return {
        "mode": mode_label,
        "n_steps_per_device": n_steps,
        "runtime_s": runtime_s,
        "resource": resource_stats.to_dict(),

        "confidence_gain": {
            "mean": _safe_mean(gains),
            "positive_pct": positive_gain_pct,
            "high_urgency_mean": _safe_mean(high_urgency_gains),
            "high_urgency_positive_pct": (
                sum(1 for g in high_urgency_gains if g > 0) / len(high_urgency_gains) * 100
                if high_urgency_gains else 0.0
            ),
            "low_urgency_mean": _safe_mean(low_urgency_gains),
            "low_urgency_positive_pct": (
                sum(1 for g in low_urgency_gains if g > 0) / len(low_urgency_gains) * 100
                if low_urgency_gains else 0.0
            ),
        },

        "comm": {
            "ECG": ecg_comm.to_dict(),
            "Breath": breath_comm.to_dict(),
            "Motion": motion_comm.to_dict(),
            "overall_upload_kbps": (
                ecg_comm.upload_kbps + breath_comm.upload_kbps + motion_comm.upload_kbps
            ) / 3,
            "overall_download_kbps": (
                ecg_comm.download_kbps + breath_comm.download_kbps + motion_comm.download_kbps
            ) / 3,
        },
    }


# ---------------------------------------------------------------------------
# Multi-run experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_runs: int = DEFAULT_RUNS,
    n_steps: int = DEFAULT_STEPS,
    sleep_s: float = DEFAULT_SLEEP_S,
    data_dir: str | None = None,
    model_dir: str = MODEL_DIR,
    output_dir: str = "simulation/results",
    device_str: str = "cpu",
    retrain: bool = False,
) -> None:
    device = torch.device(device_str)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train or load models
    if retrain or not Path(model_dir).exists() or not list(Path(model_dir).glob("*.pt")):
        logger.info("Training models...")
        train_all(data_dir=data_dir, model_dir=model_dir, device_str=device_str)
    models = load_models(model_dir=model_dir, device=device)

    with_results: list[dict] = []
    without_results: list[dict] = []

    for run_idx in range(n_runs):
        logger.info("=== Run %d/%d ===", run_idx + 1, n_runs)
        port_offset = run_idx * 20  # avoid port conflicts between runs

        # With ZhiSync
        with_res = run_one_experiment(
            models, n_steps, sleep_s,
            zhisync_enabled=True,
            data_dir=data_dir,
            device=device,
            port_offset=port_offset,
            seed=run_idx,
        )
        with_results.append(with_res)

        time.sleep(1.0)  # brief pause between modes

        # Without ZhiSync (baseline)
        without_res = run_one_experiment(
            models, n_steps, sleep_s,
            zhisync_enabled=False,
            data_dir=data_dir,
            device=device,
            port_offset=port_offset + 10,
            seed=run_idx,
        )
        without_results.append(without_res)

    # Save raw results
    raw_path = out_dir / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump({"with_zhisync": with_results, "without_zhisync": without_results}, f, indent=2)
    logger.info("Raw results saved to %s", raw_path)

    # Print summary
    _print_summary(with_results, without_results)


def _print_summary(
    with_results: list[dict],
    without_results: list[dict],
) -> None:
    """Print a Table I-style summary comparing With ZhiSync and Without ZhiSync runs."""
    print("\n" + "=" * 60)
    print("ZhiSync Simulation Results Summary")
    print("=" * 60)

    def stat(key_path: list[str], results: list[dict]) -> str:
        """Extract a nested metric from each run dict and return median [IQR] string."""
        vals = []
        for r in results:
            obj = r
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, (int, float)):
                vals.append(obj)
        if not vals:
            return "N/A"
        import statistics
        med = statistics.median(vals)
        iqr = [_safe_percentile(vals, 25), _safe_percentile(vals, 75)]
        return f"{med:.4f} [{iqr[0]:.4f}–{iqr[1]:.4f}]"

    print("\n--- Confidence Gain (With ZhiSync) ---")
    print(f"  Mean gain:           {stat(['confidence_gain', 'mean'], with_results)}")
    print(f"  Positive gain %:     {stat(['confidence_gain', 'positive_pct'], with_results)}")
    print(f"  High-urgency mean:   {stat(['confidence_gain', 'high_urgency_mean'], with_results)}")
    print(f"  High-urgency pos %:  {stat(['confidence_gain', 'high_urgency_positive_pct'], with_results)}")
    print(f"  Low-urgency mean:    {stat(['confidence_gain', 'low_urgency_mean'], with_results)}")

    print("\n--- Runtime (seconds) ---")
    print(f"  With ZhiSync:        {stat(['runtime_s'], with_results)}")
    print(f"  Without ZhiSync:     {stat(['runtime_s'], without_results)}")

    print("\n--- CPU % (median) ---")
    print(f"  With ZhiSync:        {stat(['resource', 'cpu_median'], with_results)}")
    print(f"  Without ZhiSync:     {stat(['resource', 'cpu_median'], without_results)}")

    print("\n--- RSS Memory MB (median) ---")
    print(f"  With ZhiSync:        {stat(['resource', 'rss_median_mb'], with_results)}")
    print(f"  Without ZhiSync:     {stat(['resource', 'rss_median_mb'], without_results)}")

    print("\n--- Communication kB/s (upload / download, overall avg) ---")
    print(f"  With ZhiSync upload: {stat(['comm', 'overall_upload_kbps'], with_results)}")
    print(f"  Without ZhiSync upl: {stat(['comm', 'overall_upload_kbps'], without_results)}")
    print("=" * 60 + "\n")


def _safe_mean(vals: list[float]) -> float:
    """Return the arithmetic mean of vals, or 0.0 if vals is empty."""
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _safe_percentile(data: list[float], pct: float) -> float:
    """Return the pct-th percentile of data using linear interpolation."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return sorted_data[lo] * (1 - (k - lo)) + sorted_data[hi] * (k - lo)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run ZhiSync simulation experiments.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--output-dir", default="simulation/results")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--sleep-s", type=float, default=DEFAULT_SLEEP_S)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    run_experiment(
        n_runs=args.runs,
        n_steps=args.steps,
        sleep_s=args.sleep_s,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        retrain=args.retrain,
    )
