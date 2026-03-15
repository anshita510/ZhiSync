"""
ZhiSync Metrics — psutil-based CPU/RAM sampler + communication statistics.

Monitors per-process CPU utilization and memory usage at 1 Hz using psutil,
recording instantaneous CPU% and resident set size (RSS, MB).

Every UDP packet transmission and reception is logged with its payload size
(bytes) and packet count per port.

Wall-clock execution time is measured using time.perf_counter() (end minus start).
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import psutil  # optional — graceful degradation if not installed
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ResourceSample:
    timestamp: float
    cpu_pct: float   # instantaneous CPU %
    rss_mb: float    # resident set size in MB


@dataclass
class ResourceStats:
    """Summary statistics over a collection of ResourceSamples."""
    n_samples: int = 0
    cpu_median: float = 0.0
    cpu_mean: float = 0.0
    cpu_p25: float = 0.0
    cpu_p75: float = 0.0
    rss_median: float = 0.0
    rss_mean: float = 0.0
    rss_p25: float = 0.0
    rss_p75: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "cpu_median": round(self.cpu_median, 2),
            "cpu_mean": round(self.cpu_mean, 2),
            "cpu_iqr": [round(self.cpu_p25, 2), round(self.cpu_p75, 2)],
            "rss_median_mb": round(self.rss_median, 2),
            "rss_mean_mb": round(self.rss_mean, 2),
            "rss_iqr_mb": [round(self.rss_p25, 2), round(self.rss_p75, 2)],
        }


@dataclass
class CommStats:
    """Communication statistics for one device over a run."""
    device_id: str = ""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets_sent: int = 0
    total_packets_received: int = 0
    coalesce_suppressed: int = 0
    stale_drops: int = 0
    runtime_s: float = 0.0

    @property
    def upload_kbps(self) -> float:
        if self.runtime_s <= 0:
            return 0.0
        return (self.total_bytes_sent / 1024) / self.runtime_s

    @property
    def download_kbps(self) -> float:
        if self.runtime_s <= 0:
            return 0.0
        return (self.total_bytes_received / 1024) / self.runtime_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "bytes_sent": self.total_bytes_sent,
            "bytes_received": self.total_bytes_received,
            "packets_sent": self.total_packets_sent,
            "packets_received": self.total_packets_received,
            "upload_kbps": round(self.upload_kbps, 4),
            "download_kbps": round(self.download_kbps, 4),
            "coalesce_suppressed": self.coalesce_suppressed,
            "stale_drops": self.stale_drops,
            "runtime_s": round(self.runtime_s, 3),
        }


# ---------------------------------------------------------------------------
# Resource sampler
# ---------------------------------------------------------------------------

class ResourceSampler:
    """
    Background thread that samples CPU % and RSS memory of a process at a
    fixed interval (default 1 Hz), matching the paper's monitoring approach.

    Usage::

        sampler = ResourceSampler(pid=os.getpid())
        sampler.start()
        # ... run simulation ...
        stats = sampler.stop()
    """

    def __init__(self, pid: int | None = None, interval_s: float = 1.0) -> None:
        self.pid = pid if pid is not None else os.getpid()
        self.interval_s = interval_s
        self._samples: list[ResourceSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if not _PSUTIL_AVAILABLE:
            return
        self._samples.clear()
        self._stop_event.clear()
        # Prime the CPU percent baseline (first call always returns 0)
        proc = psutil.Process(self.pid)
        proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True, args=(proc,))
        self._thread.start()

    def stop(self) -> ResourceStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
        return self._summarize()

    def _loop(self, proc: "psutil.Process") -> None:
        while not self._stop_event.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                self._samples.append(ResourceSample(
                    timestamp=time.time(),
                    cpu_pct=cpu,
                    rss_mb=rss_mb,
                ))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _summarize(self) -> ResourceStats:
        if not self._samples:
            return ResourceStats()
        cpus = [s.cpu_pct for s in self._samples]
        rsses = [s.rss_mb for s in self._samples]
        return ResourceStats(
            n_samples=len(self._samples),
            cpu_median=statistics.median(cpus),
            cpu_mean=statistics.mean(cpus),
            cpu_p25=float(_percentile(cpus, 25)),
            cpu_p75=float(_percentile(cpus, 75)),
            rss_median=statistics.median(rsses),
            rss_mean=statistics.mean(rsses),
            rss_p25=float(_percentile(rsses, 25)),
            rss_p75=float(_percentile(rsses, 75)),
        )

    @property
    def samples(self) -> list[ResourceSample]:
        return list(self._samples)


# ---------------------------------------------------------------------------
# Wall-clock timer
# ---------------------------------------------------------------------------

class WallTimer:
    """
    Thin wrapper around time.perf_counter() matching the paper's
    "wall-clock execution time computed using time.perf_counter()".
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        self._end = time.perf_counter()
        return self.elapsed_s

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_run_stats(
    run_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate per-run metric dicts into median ± IQR summary, matching
    Table I format from the paper.
    """
    if not run_stats:
        return {}

    keys = list(run_stats[0].keys())
    summary: dict[str, Any] = {}
    for key in keys:
        vals = [r[key] for r in run_stats if isinstance(r.get(key), (int, float))]
        if vals:
            summary[key] = {
                "median": round(statistics.median(vals), 4),
                "iqr": [round(_percentile(vals, 25), 4), round(_percentile(vals, 75), 4)],
                "mean": round(statistics.mean(vals), 4),
                "se": round(_sem(vals), 4),
            }
        else:
            summary[key] = run_stats[0].get(key)
    return summary


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of data using linear interpolation."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _sem(data: list[float]) -> float:
    """Return the standard error of the mean for data."""
    n = len(data)
    if n < 2:
        return 0.0
    import math
    return statistics.stdev(data) / math.sqrt(n)
