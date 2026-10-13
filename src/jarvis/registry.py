"""Device registry. Polls each Ollama node and swaps in an immutable snapshot.

Requests read the snapshot and never take a lock. A device is online after
one successful poll and offline after two failures in a row. The poll timeout
stays short so a sleeping laptop cannot stall the others.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import requests

log = logging.getLogger("jarvis.registry")


@dataclass(frozen=True)
class DeviceView:
    name: str
    online: bool
    pulled: tuple[str, ...]
    loaded: tuple[tuple[str, int | None, int | None], ...]
    consecutive_failures: int


@dataclass(frozen=True)
class Snapshot:
    devices: Mapping[str, DeviceView]

    def as_dict(self) -> dict:
        return {
            "devices": {
                name: {
                    "online": view.online,
                    "pulled": list(view.pulled),
                    "loaded": [
                        {"name": model, "size": size, "size_vram": vram} for model, size, vram in view.loaded
                    ],
                    "consecutive_failures": view.consecutive_failures,
                }
                for name, view in self.devices.items()
            }
        }


_bound: contextvars.ContextVar[Snapshot | None] = contextvars.ContextVar(
    "jarvis_bound_snapshot", default=None
)


def _model_names(payload: dict) -> tuple[str, ...]:
    found: list[str] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value not in found:
                found.append(value)
    return tuple(found)


def _loaded_models(payload: dict) -> tuple[tuple[str, int | None, int | None], ...]:
    rows = []
    for item in payload.get("models") or []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            rows.append((item["name"], item.get("size"), item.get("size_vram")))
    return tuple(rows)


async def fetch_json(url: str, timeout: float) -> dict:
    """GET JSON with a hard timeout. Runs in a worker thread so one device cannot block another."""

    def _get() -> dict:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"expected a JSON object from {url}")
        return body

    return await asyncio.to_thread(_get)


class Registry:
    def __init__(self, topology: dict, fetch=None):
        self._topology = topology
        self._fetch = fetch or fetch_json
        settings = topology.get("registry", {})
        self._interval = float(settings.get("poll_interval_s", 20))
        self._timeout = float(settings.get("poll_timeout_s", 3))
        self._offline_after = int(settings.get("offline_after_failures", 2))
        self._online_after = int(settings.get("online_after_successes", 1))
        names = list(topology.get("devices", {}))
        self._failures = {name: 0 for name in names}
        self._successes = {name: 0 for name in names}
        self._online = {name: False for name in names}
        self._pulled: dict[str, tuple[str, ...]] = {name: () for name in names}
        self._loaded: dict[str, tuple] = {name: () for name in names}
        self.snapshot = self._snapshot()

    def _snapshot(self) -> Snapshot:
        devices = {
            name: DeviceView(
                name=name,
                online=self._online[name],
                pulled=self._pulled[name],
                loaded=self._loaded[name],
                consecutive_failures=self._failures[name],
            )
            for name in self._topology.get("devices", {})
        }
        return Snapshot(MappingProxyType(devices))

    async def poll_once(self) -> None:
        devices = self._topology.get("devices", {})
        results = await asyncio.gather(*(self._poll(name, cfg) for name, cfg in devices.items()))
        for name, ok, tags, ps, err in results:
            self._apply(name, ok, tags, ps, err)
        self.snapshot = self._snapshot()

    async def _poll(self, name: str, cfg: dict):
        base = f"http://{cfg['host']}:{cfg.get('port', 11434)}"
        try:
            tags, ps = await asyncio.gather(
                self._fetch(f"{base}/api/tags", self._timeout),
                self._fetch(f"{base}/api/ps", self._timeout),
            )
        except Exception as exc:
            return name, False, None, None, str(exc)
        return name, True, tags, ps, ""

    def _apply(self, name: str, ok: bool, tags, ps, err: str) -> None:
        was_online = self._online[name]
        if ok:
            self._failures[name] = 0
            self._successes[name] += 1
            self._pulled[name] = _model_names(tags or {})
            self._loaded[name] = _loaded_models(ps or {})
            if not was_online and self._successes[name] >= self._online_after:
                self._online[name] = True
                log.info("device %s online", name)
            return
        self._successes[name] = 0
        self._failures[name] += 1
        if was_online and self._failures[name] >= self._offline_after:
            self._online[name] = False
            log.warning("device %s offline after %s failures: %s", name, self._failures[name], err)

    async def run(self) -> None:
        """Sleep first. The lifespan polls once before serving traffic."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.poll_once()
            except Exception:
                log.exception("registry poll failed")


_registry: Registry | None = None


def set_registry(registry: Registry | None) -> None:
    global _registry
    _registry = registry


def bind_snapshot(snapshot: Snapshot):
    """Pin the snapshot seen by current_snapshot() in this task and its worker threads."""
    return _bound.set(snapshot)


def reset_snapshot(token) -> None:
    _bound.reset(token)


def current_snapshot() -> Snapshot:
    bound = _bound.get()
    if bound is not None:
        return bound
    if _registry is None:
        return Snapshot(MappingProxyType({}))
    return _registry.snapshot


def mini_only(snapshot: Snapshot) -> Snapshot:
    """Mark every device except the mini offline. Used when the decomposer must stay on the mini."""
    devices = {}
    for name, view in snapshot.devices.items():
        devices[name] = view if name == "mini" else replace(view, online=False)
    return Snapshot(MappingProxyType(devices))
