"""
Intent classifier (ADR 0008): labels only, no prompt rewrite.

The small model sees the last few turns of the conversation and returns
{intent, complexity, needs_rag}. Output is constrained with a JSON schema
(LiteLLM maps it to Ollama's `format`) and validated here, so a bad reply
falls back to safe defaults instead of breaking the request. The user's own
messages go to the downstream model unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import requests

from jarvis.messages import text_of

INTENTS = ("general", "code", "reasoning", "code+reasoning")
COMPLEXITIES = ("simple", "medium", "complex")
DEFAULT = {"intent": "general", "complexity": "simple", "needs_rag": False}

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "complexity": {"type": "string", "enum": list(COMPLEXITIES)},
        "needs_rag": {"type": "boolean"},
    },
    "required": ["intent", "complexity", "needs_rag"],
    "additionalProperties": False,
}

SYSTEM = """You label requests for a router. You never answer them.

Label the LATEST user request, using the earlier turns only to resolve what it refers to.

intent:
- code: write, fix, review, explain or find specific code or files
- reasoning: design choices, architecture, debugging logic, "why" or "how should I", no code needed
- code+reasoning: design AND build something new, e.g. "design and implement", "build this feature"
- general: concepts, summaries, chat, anything else

"what does X do", "list my routes", "explain this file" are code, not code+reasoning.

complexity: simple (one focused answer), medium (a few parts), complex (many parts or files)

needs_rag: true when the request is about the user's own project, repo, files or components.

Reply with JSON only: {"intent": ..., "complexity": ..., "needs_rag": ...}"""

TURNS = 6  # messages of history shown to the classifier
TURN_CHARS = 400  # earlier turns are clipped to this; the latest request is clipped to 2000

CLASSIFY_TIMEOUT = 15  # seconds; a hung 3B model must not hold a request

Complete = Callable[..., str]


def build_messages(messages: list[dict]) -> list[dict]:
    """Show the conversation as a transcript, so the small model labels it instead of continuing it."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]
    *earlier, latest = turns[-TURNS:]
    lines = []
    for m in earlier:
        t = " ".join(text_of(m.get("content")).split())
        if len(t) > TURN_CHARS:
            t = t[:TURN_CHARS] + " ..."
        lines.append(f"{m['role'].capitalize()}: {t}")
    body = ""
    if lines:
        body = "Earlier conversation:\n" + "\n".join(lines) + "\n\n"
    body += "Latest user request:\n" + text_of(latest.get("content"))[:2000]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def parse_labels(raw: str) -> dict | None:
    """Validate the classifier's reply. Returns None if it is not exactly the labels we asked for."""
    try:
        data = json.loads(raw.strip())
    except (ValueError, AttributeError):
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end])
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    intent, complexity, rag = data.get("intent"), data.get("complexity"), data.get("needs_rag")
    if intent not in INTENTS or complexity not in COMPLEXITIES or not isinstance(rag, bool):
        return None
    return {"intent": intent, "complexity": complexity, "needs_rag": rag}


def classify(messages: list[dict], complete: Complete, alias: str = "classify-mini") -> tuple[dict, str]:
    """Returns (labels, source). source is "classifier" or "fallback: <reason>"."""
    try:
        raw = complete(
            alias,
            build_messages(messages),
            temperature=0,
            max_tokens=40,
            response_format={"type": "json_schema", "json_schema": {"name": "route", "schema": SCHEMA}},
            timeout=CLASSIFY_TIMEOUT,
            retry=False,
        )
    except requests.Timeout:
        return dict(DEFAULT), "fallback: classifier timeout"
    except Exception as e:
        return dict(DEFAULT), f"fallback: {e}"
    labels = parse_labels(raw)
    if labels is None:
        return dict(DEFAULT), f"fallback: unparseable reply {raw[:80]!r}"
    return labels, "classifier"
