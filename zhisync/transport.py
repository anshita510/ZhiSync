import json
import socket
import threading
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
    UDP JSON metadata transport.

    Each node binds one local UDP port and broadcasts ZhiTag JSON packets
    to configured peers.
    """

    def __init__(
        self,
        node_id: str,
        bind_host: str,
        bind_port: int,
        peers: Iterable[UdpPeer],
        recv_buffer_bytes: int = 4096,
        socket_timeout_seconds: float = 0.2,
    ) -> None:
        self.node_id = node_id
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.peers = tuple(peers)
        self.recv_buffer_bytes = recv_buffer_bytes
        self.socket_timeout_seconds = socket_timeout_seconds

        self._latest_by_peer: dict[str, ZhiTag] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recv_sock: socket.socket | None = None
        self._send_sock: socket.socket | None = None

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

    def broadcast(self, tag: ZhiTag) -> None:
        if self._send_sock is None:
            raise RuntimeError("Transport not started. Call start() first.")
        payload = json.dumps(tag.to_dict(), separators=(",", ":")).encode("utf-8")
        for peer in self.peers:
            self._send_sock.sendto(payload, (peer.host, peer.port))

    def get_latest_peer_tags(self) -> dict[str, ZhiTag]:
        with self._lock:
            return dict(self._latest_by_peer)

    def _recv_loop(self) -> None:
        assert self._recv_sock is not None
        while not self._stop_event.is_set():
            try:
                raw, _ = self._recv_sock.recvfrom(self.recv_buffer_bytes)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                data = json.loads(raw.decode("utf-8"))
                tag = ZhiTag.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

            if tag.device_id == self.node_id:
                continue
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
