"""
Model transport for the gateway: the only module that talks to model servers.

Today it calls the LiteLLM proxy over HTTP with an async client, so a slow
answer never blocks the event loop. ADR 0006 replaces the internals with an
in-process litellm.Router; the two public functions keep their signatures so
callers do not change.

    complete(alias, messages, **params) -> str
    stream(alias, messages, **params)   -> async iterator of (kind, text)

kind is "content" for answer tokens and "reasoning" for thinking tokens
(deepseek-r1 and other reasoning models send those separately).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from jarvis.config import settings

# Connect fast, read slow: local models can take minutes to finish.
TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=30.0)

_client: httpx.AsyncClient | None = None


class LLMError(RuntimeError):
    """A model call failed. The message is safe to show to the user."""


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.litellm_base,
            headers={"Authorization": f"Bearer {settings.litellm_key}"},
            timeout=TIMEOUT,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
        msg = (
            body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
        )
        return str(msg or body)[:300]
    except Exception:
        return r.text[:300]


async def complete(alias: str, messages: list[dict], retries: int = 1, **params) -> str:
    """One non streaming call. Retries once on connection errors and 5xx, never on 4xx."""
    payload = {"model": alias, "messages": messages, **params}
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = await _get_client().post("/chat/completions", json=payload)
        except httpx.TransportError as e:
            last = LLMError(f"{alias}: cannot reach model server ({type(e).__name__})")
        else:
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                return (msg.get("content") or "").strip()
            last = LLMError(f"{alias}: HTTP {r.status_code}: {_error_text(r)}")
            if r.status_code < 500:
                break
        if attempt < retries:
            await asyncio.sleep(2)
    assert last is not None
    raise last


async def stream(alias: str, messages: list[dict], **params) -> AsyncIterator[tuple[str, str]]:
    """Stream one call. Raises LLMError on a non 200 status instead of yielding nothing."""
    payload = {"model": alias, "messages": messages, "stream": True, **params}
    try:
        async with _get_client().stream("POST", "/chat/completions", json=payload) as r:
            if r.status_code != 200:
                await r.aread()
                raise LLMError(f"{alias}: HTTP {r.status_code}: {_error_text(r)}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError):
                    continue
                thinking = delta.get("reasoning_content") or delta.get("reasoning")
                if thinking:
                    yield "reasoning", thinking
                if delta.get("content"):
                    yield "content", delta["content"]
    except httpx.TransportError as e:
        raise LLMError(f"{alias}: connection lost ({type(e).__name__})") from e
