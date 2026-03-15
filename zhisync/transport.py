import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from .types import ZhiTag


class MetadataTransport(ABC):
    """Transport contract used by ZhiSyncNode."""

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def broadcast(self, tag: ZhiTag) -> None:
        pass

    @abstractmethod
    def get_latest_peer_tags(self) -> dict[str, ZhiTag]:
        pass

    def get_stats(self) -> dict[str, int]:
        """Return communication counters. Subclasses may override."""
        return {
            "bytes_sent": 0,
            "bytes_received": 0,
            "packets_sent": 0,
            "packets_received": 0,
            "coalesce_suppressed": 0,
            "stale_drops": 0,
        }

    def reset_stats(self) -> None:
        """Reset counters to zero. Subclasses may override."""

    def record_stale_drop(self, count: int = 1) -> None:
        """Called by ZhiSyncNode when staleness filter discards peer tags."""


class NullTransport(MetadataTransport):
    """No-op transport for single-node or unit-test integration."""

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def broadcast(self, tag: ZhiTag) -> None:
        _ = tag
        return

    def get_latest_peer_tags(self) -> dict[str, ZhiTag]:
        return {}


@dataclass(frozen=True)
class UdpPeer:
    host: str
    port: int


class UdpJsonTransport(MetadataTransport):
    """
    UDP JSON metadata transport with communication statistics.

    Tracks bytes sent/received, packet counts, coalesced-suppressed sends
    (COALESCE_MS), and stale-drop events (STALENESS_MS).

    Parameters
    ----------
    coalesce_ms:
        Minimum interval in milliseconds between successive broadcasts from
        this node.  Any broadcast attempt arriving sooner than this interval
        is suppressed and counted in ``coalesce_suppressed``.  Set to 0
        (default) to disable coalescing.
    """

    def __init__(
        self,
        node_id: str,
        bind_host: str,
        bind_port: int,
        peers: Iterable[UdpPeer],
        recv_buffer_bytes: int = 4096,
        socket_timeout_seconds: float = 0.2,
        coalesce_ms: float = 0.0,
    ) -> None:
        self.node_id = node_id
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.peers = tuple(peers)
        self.recv_buffer_bytes = recv_buffer_bytes
        self.socket_timeout_seconds = socket_timeout_seconds
        self.coalesce_ms = coalesce_ms

        self._latest_by_peer: dict[str, ZhiTag] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recv_sock: socket.socket | None = None
        self._send_sock: socket.socket | None = None

        # --- communication statistics ---
        self._stats_lock = threading.Lock()
        self._bytes_sent: int = 0
        self._bytes_received: int = 0
        self._packets_sent: int = 0
        self._packets_received: int = 0
        self._coalesce_suppressed: int = 0  # COALESCE_MS counter
        self._stale_drops: int = 0          # STALENESS_MS counter
        self._last_broadcast_ts: float = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind((self.bind_host, self.bind_port))
        self._recv_sock.settimeout(self.socket_timeout_seconds)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._recv_sock is not None:
            self._recv_sock.close()
        if self._send_sock is not None:
            self._send_sock.close()
        self._recv_sock = None
        self._send_sock = None

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def broadcast(self, tag: ZhiTag) -> None:
        if self._send_sock is None:
            raise RuntimeError("Transport not started. Call start() first.")

        # Coalescing: suppress if within COALESCE_MS of last send
        if self.coalesce_ms > 0.0:
            now = time.monotonic()
            elapsed_ms = (now - self._last_broadcast_ts) * 1000.0
            if elapsed_ms < self.coalesce_ms:
                with self._stats_lock:
                    self._coalesce_suppressed += 1
                return
            self._last_broadcast_ts = now

        payload = json.dumps(tag.to_dict(), separators=(",", ":")).encode("utf-8")
        n_peers = len(self.peers)
        for peer in self.peers:
            self._send_sock.sendto(payload, (peer.host, peer.port))
        with self._stats_lock:
            self._bytes_sent += len(payload) * n_peers
            self._packets_sent += n_peers

    def get_latest_peer_tags(self) -> dict[str, ZhiTag]:
        with self._lock:
            return dict(self._latest_by_peer)

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of communication counters."""
        with self._stats_lock:
            return {
                "bytes_sent": self._bytes_sent,
                "bytes_received": self._bytes_received,
                "packets_sent": self._packets_sent,
                "packets_received": self._packets_received,
                "coalesce_suppressed": self._coalesce_suppressed,
                "stale_drops": self._stale_drops,
            }

    def reset_stats(self) -> None:
        """Reset all counters to zero."""
        with self._stats_lock:
            self._bytes_sent = 0
            self._bytes_received = 0
            self._packets_sent = 0
            self._packets_received = 0
            self._coalesce_suppressed = 0
            self._stale_drops = 0
            self._last_broadcast_ts = 0.0

    def record_stale_drop(self, count: int = 1) -> None:
        """Increment stale-drop counter (called by ZhiSyncNode after freshness filter)."""
        with self._stats_lock:
            self._stale_drops += count

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._recv_sock is not None
        while not self._stop_event.is_set():
            try:
                raw, _ = self._recv_sock.recvfrom(self.recv_buffer_bytes)
            except socket.timeout:
                continue
            except OSError:
                break

            n_bytes = len(raw)
            try:
                data = json.loads(raw.decode("utf-8"))
                tag = ZhiTag.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

            if tag.device_id == self.node_id:
                continue

            with self._stats_lock:
                self._bytes_received += n_bytes
                self._packets_received += 1

            with self._lock:
                self._latest_by_peer[tag.device_id] = tag


class InMemoryBus:
    """Simple in-process bus for integration tests and local demos."""

    def __init__(self) -> None:
        self._transports: dict[str, "InMemoryTransport"] = {}
        self._lock = threading.Lock()

    def register(self, transport: "InMemoryTransport") -> None:
        with self._lock:
            self._transports[transport.node_id] = transport

    def unregister(self, node_id: str) -> None:
        with self._lock:
            self._transports.pop(node_id, None)

    def broadcast(self, sender_id: str, tag: ZhiTag) -> None:
        with self._lock:
            targets = [t for node_id, t in self._transports.items() if node_id != sender_id]
        for transport in targets:
            transport._ingest(tag)


class InMemoryTransport(MetadataTransport):
    """Thread-safe in-memory transport for SDK-level tests."""

    def __init__(self, node_id: str, bus: InMemoryBus) -> None:
        self.node_id = node_id
        self.bus = bus
        self._running = False
        self._latest_by_peer: dict[str, ZhiTag] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.bus.register(self)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.bus.unregister(self.node_id)

    def broadcast(self, tag: ZhiTag) -> None:
        if not self._running:
            raise RuntimeError("Transport not started. Call start() first.")
        self.bus.broadcast(sender_id=self.node_id, tag=tag)

    def get_latest_peer_tags(self) -> dict[str, ZhiTag]:
        with self._lock:
            return dict(self._latest_by_peer)

    def _ingest(self, tag: ZhiTag) -> None:
        with self._lock:
            self._latest_by_peer[tag.device_id] = tag
