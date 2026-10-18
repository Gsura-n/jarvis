"""Device registry: state changes, one slow device, and GET /devices."""

import asyncio
import logging
import time

from jarvis.registry import Registry


def test_registry_state_transitions(caplog):
    caplog.set_level(logging.INFO)
    mode = {"ok": False}

    async def fake(url, timeout):
        if not mode["ok"]:
            raise TimeoutError("asleep")
        if url.endswith("/api/ps"):
            return {"models": []}
        return {"models": [{"name": "llama3.1:8b"}]}

    registry = Registry(
        {"devices": {"mini": {"host": "mini.test", "port": 11434}}, "registry": {}}, fetch=fake
    )

    async def scenario():
        await registry.poll_once()
        assert registry.snapshot.devices["mini"].online is False
        mode["ok"] = True
        await registry.poll_once()
        assert registry.snapshot.devices["mini"].online is True
        assert "llama3.1:8b" in registry.snapshot.devices["mini"].pulled
        mode["ok"] = False
        await registry.poll_once()
        assert registry.snapshot.devices["mini"].online is True
        await registry.poll_once()
        assert registry.snapshot.devices["mini"].online is False

    asyncio.run(scenario())
    assert "device mini online" in caplog.text
    assert "device mini offline" in caplog.text


def test_timeout_on_one_device_does_not_block_the_other():
    started: list[tuple[str, float]] = []

    async def fake(url, timeout):
        started.append((url, time.monotonic()))
        if "slow.test" in url:
            await asyncio.sleep(0.4)
            raise TimeoutError("slow")
        if url.endswith("/api/ps"):
            return {"models": []}
        return {"models": [{"name": "llama3.1:8b"}]}

    registry = Registry(
        {
            "devices": {
                "slow": {"host": "slow.test", "port": 11434},
                "mini": {"host": "mini.test", "port": 11434},
            },
            "registry": {},
        },
        fetch=fake,
    )

    async def scenario():
        started_at = time.monotonic()
        await registry.poll_once()
        return started_at

    started_at = asyncio.run(scenario())
    assert min(stamp for url, stamp in started if "mini.test" in url) - started_at < 0.15
    assert min(stamp for url, stamp in started if "slow.test" in url) - started_at < 0.15
    assert registry.snapshot.devices["mini"].online is True
    assert registry.snapshot.devices["slow"].online is False


def test_devices_requires_the_api_key(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", "test-key")

    async def fake(url, timeout):
        if url.endswith("/api/ps"):
            return {"models": []}
        return {"models": [{"name": "llama3.1:8b"}]}

    monkeypatch.setattr("jarvis.registry.fetch_json", fake)
    from fastapi.testclient import TestClient

    from jarvis.gateway import app

    with TestClient(app) as client:
        assert client.get("/devices").status_code == 401
        assert client.get("/devices", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/devices", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["devices"]["mini"]["online"] is True
    assert "macbook-pro" in body["devices"]


def test_model_names_normalize_untagged_models():
    from jarvis.registry import _model_names

    names = _model_names({"models": [{"name": "nomic-embed-text"}, {"name": "llama3.1:8b"}]})
    assert names == ("nomic-embed-text:latest", "llama3.1:8b")
