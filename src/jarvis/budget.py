"""Fit system, history, RAG and handoff into num_ctx minus max_tokens.

One token is about four characters. Drop oldest history first, then cut RAG
from the end, then handoff. The system prompt and current user message stay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("jarvis.budget")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class Fitted:
    system: str
    history: tuple[dict, ...]
    rag: str
    handoff: str
    user: str
    trimmed: tuple[str, ...]


def _cost(system: str, history: list[dict], rag: str, handoff: str, user: str) -> int:
    messages = sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in history)
    return (
        estimate_tokens(system)
        + estimate_tokens(rag)
        + estimate_tokens(handoff)
        + estimate_tokens(user)
        + messages
    )


def _shorten(text: str, cost_of, limit: int) -> str:
    while text and cost_of(text) > limit:
        extra = cost_of(text) - limit
        shortened = text[: max(0, len(text) - max(4, extra * 4))]
        text = shortened if shortened != text else text[:-4]
    return text


def fit_input(
    *,
    system: str,
    history: list[dict],
    rag: str,
    handoff: str,
    user: str,
    max_tokens: int,
    num_ctx: int,
) -> Fitted:
    limit = max(1, num_ctx - max_tokens)
    kept = [dict(item) for item in history]
    rag = rag or ""
    handoff = handoff or ""
    trimmed: list[str] = []

    while kept and _cost(system, kept, rag, handoff, user) > limit:
        kept.pop(0)
        if "history" not in trimmed:
            trimmed.append("history")

    if rag and _cost(system, kept, rag, handoff, user) > limit:
        rag = _shorten(rag, lambda text: _cost(system, kept, text, handoff, user), limit)
        trimmed.append("rag")
    if handoff and _cost(system, kept, rag, handoff, user) > limit:
        handoff = _shorten(handoff, lambda text: _cost(system, kept, rag, text, user), limit)
        trimmed.append("handoff")
    if _cost(system, kept, rag, handoff, user) > limit:
        trimmed.append("over_budget")
    return Fitted(system, tuple(kept), rag, handoff, user, tuple(trimmed))


def assemble_messages(
    *,
    system: str,
    history: list[dict],
    rag: str,
    handoff: str,
    user: str,
    max_tokens: int,
    num_ctx: int,
) -> list[dict]:
    fitted = fit_input(
        system=system,
        history=history,
        rag=rag,
        handoff=handoff,
        user=user,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
    )
    if fitted.trimmed:
        log.warning(
            "trimmed input to fit num_ctx=%s max_tokens=%s parts=%s",
            num_ctx,
            max_tokens,
            ",".join(fitted.trimmed),
        )
    parts = [fitted.system]
    if fitted.rag:
        parts.append(fitted.rag)
    if fitted.handoff:
        parts.append(fitted.handoff)
    messages = [{"role": "system", "content": "\n\n".join(part for part in parts if part)}]
    messages.extend({"role": item["role"], "content": item.get("content", "")} for item in fitted.history)
    messages.append({"role": "user", "content": fitted.user})
    return messages


def context_window(config: dict, deployment: str) -> int:
    for candidates in config.get("tiers", {}).values():
        for cand in candidates:
            if cand.get("litellm") == deployment and cand.get("num_ctx"):
                return int(cand["num_ctx"])
    return 8192


def answer_token_budget(deployment: str, config: dict) -> int:
    answer = config.get("budgets", {}).get("tokens", {}).get("answer", {})
    large = deployment in {"reasoning", "large-mbp"} or "-large" in deployment
    return int(answer.get("large" if large else "small", 6000 if large else 4000))
