import asyncio
import time

import httpx
import pytest
from httpx import ASGITransport

import jarvis.gateway as gateway
from jarvis.gateway import app


@pytest.mark.anyio
async def test_completions_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(gateway, "jarvis_api_key", lambda: "test-key")

    def fake_classify(*args, **kwargs):
        time.sleep(1.0)
        return {"intent": "general", "complexity": "simple", "needs_rag": False}, "mocked"

    monkeypatch.setattr(gateway, "classify", fake_classify)
    monkeypatch.setattr(gateway, "chat", lambda *args, **kwargs: "Mocked answer")
    monkeypatch.setattr(gateway, "get_embedding", lambda text: [])
    monkeypatch.setattr(gateway, "driver", None)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        chat_payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "auto"}
        headers = {"Authorization": "Bearer test-key"}

        t_start = time.perf_counter()
        chat_task = asyncio.create_task(
            client.post("/v1/chat/completions", json=chat_payload, headers=headers)
        )

        await asyncio.sleep(0.1)

        health_response = await client.get("/health")
        health_finish = time.perf_counter() - t_start

        chat_response = await chat_task
        chat_finish = time.perf_counter() - t_start

        assert health_response.status_code == 200
        assert chat_response.status_code == 200

        assert (
            health_finish < chat_finish - 0.5
        ), f"Health took {health_finish:.2f}s, chat took {chat_finish:.2f}s"
