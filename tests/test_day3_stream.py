"""Cancelling a stream closes the upstream connection before the device lock is released."""

import asyncio
import json
import logging
import time

import anyio
import httpx
import pytest

import jarvis.gateway as gateway
from jarvis.gateway import device_semaphore, stream_tokens


@pytest.fixture(autouse=True)
def _fresh_semaphores():
    """Each test gets a new semaphore. A semaphore binds to the first loop that uses it."""
    gateway._semaphores.clear()
    yield
    gateway._semaphores.clear()


class _OrderSemaphore(asyncio.Semaphore):
    def __init__(self, value: int, order: list[str]):
        super().__init__(value)
        self._order = order

    def release(self) -> None:
        self._order.append("unlocked")
        super().release()


class _ScriptedBody(httpx.AsyncByteStream):
    """SSE body that sleeps between lines and records when the client closes it."""

    def __init__(self, lines: list[bytes], order: list[str], delay: float = 0.02, pre_delay: float = 0.0):
        self._lines = lines
        self._order = order
        self._delay = delay
        self._pre_delay = pre_delay

    async def __aiter__(self):
        if self._pre_delay:
            await asyncio.sleep(self._pre_delay)
        for line in self._lines:
            await asyncio.sleep(self._delay)
            yield line

    async def aclose(self) -> None:
        self._order.append("closed")


def _sse(token: str) -> bytes:
    body = json.dumps({"choices": [{"delta": {"content": token}}]})
    return f"data: {body}\n".encode()


def _install(monkeypatch, lines: list[bytes], order: list[str], delay: float = 0.02, pre_delay: float = 0.0):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ScriptedBody(lines, order, delay=delay, pre_delay=pre_delay))

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gateway, "_stream_client", client, raising=False)


def _hold(device: str, order: list[str]) -> None:
    gateway._semaphores[device] = _OrderSemaphore(1, order)


def test_cancel_closes_the_upstream_before_the_lock(monkeypatch):
    """Fails on the thread worker: the lock is released while the model stream is still open."""
    order: list[str] = []
    tokens_out = [_sse(f"t{i}") for i in range(6)]
    _install(monkeypatch, tokens_out, order)
    _hold("macbook-pro", order)

    async def read_three():
        tokens = []
        with anyio.CancelScope() as scope:
            async for token in stream_tokens("reasoning-large-mbp", "macbook-pro", [], 32):
                tokens.append(token)
                if len(tokens) == 3:
                    scope.cancel()
        return tokens

    tokens = anyio.run(read_three)
    assert tokens == ["t0", "t1", "t2"]
    assert order.index("closed") < order.index("unlocked"), order
    assert device_semaphore("macbook-pro").locked() is False


def test_cancel_during_prompt_processing_closes_within_100ms(monkeypatch):
    order: list[str] = []
    closed_at: list[float] = []

    class _TimedBody(_ScriptedBody):
        async def aclose(self) -> None:
            closed_at.append(time.perf_counter())
            await super().aclose()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_TimedBody([_sse("late")], order, delay=0.01, pre_delay=2.0))

    monkeypatch.setattr(
        gateway,
        "_stream_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        raising=False,
    )
    _hold("mini", order)
    cancelled_at: list[float] = []

    async def cancel_before_the_first_token():
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tg:

                async def read():
                    async for _token in stream_tokens("general-small-mini", "mini", [], 16):
                        pass

                tg.start_soon(read)
                await anyio.sleep(0.05)
                cancelled_at.append(time.perf_counter())
                scope.cancel()

    anyio.run(cancel_before_the_first_token)
    assert closed_at, order
    assert closed_at[0] - cancelled_at[0] < 0.1
    assert order.index("closed") < order.index("unlocked"), order


def test_upstream_error_ends_the_stream_without_raising(monkeypatch, caplog):
    order: list[str] = []

    class _FailingBody(_ScriptedBody):
        async def __aiter__(self):
            yield _sse("one")
            yield _sse("two")
            raise RuntimeError("model failed")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FailingBody([], order))

    monkeypatch.setattr(
        gateway,
        "_stream_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        raising=False,
    )
    _hold("mini", order)

    async def read_all():
        tokens = [token async for token in stream_tokens("reasoning-large-mini", "mini", [], 32)]
        assert device_semaphore("mini").locked() is False
        return tokens

    with caplog.at_level(logging.ERROR):
        assert anyio.run(read_all) == ["one", "two"]
    assert "reasoning-large-mini" in caplog.text
    assert "unlocked" in order


def test_finite_stream_yields_every_token_in_order(monkeypatch):
    order: list[str] = []
    lines = [_sse(token) for token in ("t1", "t2", "t3", "t4", "t5")]
    lines.append(b"data: [DONE]\n")
    _install(monkeypatch, lines, order, delay=0.0)
    _hold("mini", order)

    async def read_all():
        return [token async for token in stream_tokens("code-small-mini", "mini", [], 8)]

    assert anyio.run(read_all) == ["t1", "t2", "t3", "t4", "t5"]
    assert device_semaphore("mini").locked() is False
