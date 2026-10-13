"""Cancelling a stream stops the model call and releases the device semaphore."""

import asyncio
import threading
import time

import pytest

import jarvis.gateway as gateway
from jarvis.gateway import device_semaphore, stream_tokens


@pytest.fixture(autouse=True)
def _fresh_semaphores():
    """Each test gets a new semaphore. A semaphore binds to the first loop that uses it."""
    gateway._semaphores.clear()
    yield
    gateway._semaphores.clear()


def test_cancel_closes_the_stream_and_releases_the_semaphore(monkeypatch):
    closed = threading.Event()

    def fake(*_args, **_kwargs):
        try:
            while True:
                time.sleep(0.02)
                yield "tok"
        finally:
            closed.set()

    monkeypatch.setattr(gateway, "chat_stream", fake)

    async def read_three():
        gen = stream_tokens("reasoning-large-mbp", "macbook-pro", [], 32)
        tokens = [await gen.__anext__() for _ in range(3)]
        await gen.aclose()
        assert device_semaphore("macbook-pro").locked() is False
        return tokens

    assert asyncio.run(read_three()) == ["tok", "tok", "tok"]
    assert closed.wait(1)


def test_worker_error_ends_the_stream_without_raising(monkeypatch):
    def fake(*_args, **_kwargs):
        yield "one"
        yield "two"
        raise RuntimeError("model failed")

    monkeypatch.setattr(gateway, "chat_stream", fake)

    async def read_all():
        tokens = [token async for token in stream_tokens("reasoning-large-mini", "mini", [], 32)]
        assert device_semaphore("mini").locked() is False
        return tokens

    assert asyncio.run(read_all()) == ["one", "two"]


def test_finite_stream_yields_every_token_in_order(monkeypatch):
    def fake(*_args, **_kwargs):
        yield from ["t1", "t2", "t3", "t4", "t5"]

    monkeypatch.setattr(gateway, "chat_stream", fake)

    async def read_all():
        return [token async for token in stream_tokens("code-small-mini", "mini", [], 8)]

    assert asyncio.run(read_all()) == ["t1", "t2", "t3", "t4", "t5"]
