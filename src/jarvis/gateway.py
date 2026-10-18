"""
Jarvis gateway, port 4001. OpenAI compatible.

Request path:
  1. API key check (every /v1 route except the two probe endpoints)
  2. Intent: taken from the `model` field when it names a tier, otherwise
     from the labels only classifier, which sees the last few turns (ADR 0008)
  3. RAG when the intent or classifier asks for it (runs in a worker thread)
  4. One model call with the real conversation, streamed when asked, or the
     task decomposer for design and build requests
  5. The routing trace is kept in memory, so "which model answered?" as a
     follow up returns the trace of the previous answer without a model call

Everything here is async. Blocking work (Neo4j, the decomposer) runs in
worker threads, so one long answer no longer freezes /health or other requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from neo4j import GraphDatabase

from jarvis import classify, llm
from jarvis.classify import text_of
from jarvis.config import settings
from jarvis.gateway_metrics import StepTimer
from jarvis.task_decomposer import decompose_and_execute

# ── Config: everything comes from jarvis.config (env file + jarvis.yaml) ─────
GATEWAY = settings.topology.get("gateway", {})
BUDGETS = settings.topology.get("budgets", {})
API_KEY = settings.api_key
HISTORY_CHARS = int(BUDGETS.get("tokens", {}).get("history", 2000)) * 4  # about 4 chars per token

_missing = settings.missing_required()
if _missing:
    print(f"⚠️  Missing required env vars: {', '.join(_missing)}")
    print(f"   Add them to {settings.env_file}")

# intent -> (LiteLLM alias, name shown in traces, extra system instruction).
# Day 3 replaces this table with policy.py reading the tiers in jarvis.yaml.
ROUTES = {
    "general": ("general", "general (llama3.1:8b)", ""),
    "code": ("coding", "coding (qwen2.5-coder:7b)", "Write clean, production-ready code with comments."),
    "reasoning": ("reasoning", "reasoning (deepseek-r1:8b)", "Think step by step. Be thorough."),
}

# `model` values that force an intent and skip the classifier.
EXPLICIT_MODELS = {
    "general": "general",
    "coding": "code",
    "reasoning": "reasoning",
    "jarvis-decompose": "code+reasoning",
}
# Anything else ("jarvis", "jarvis-trace", unknown names) is auto routed.
MODEL_IDS = ["jarvis", "jarvis-trace", "jarvis-decompose", "general", "coding", "reasoning"]

# Phrases that ask how the PREVIOUS answer was produced. Kept narrow on purpose:
# "why did you pick Postgres?" is a real question, not a trace request.
TRACE_FOLLOWUPS = (
    "show trace",
    "show the trace",
    "model trace",
    "which model answered",
    "what model answered",
    "which model did you use",
    "what model did you use",
    "which model was used",
    "who answered",
)

# Start RAG before classification finishes when the message obviously needs it.
RAG_TRIGGERS = (
    "my project",
    "my code",
    "my repo",
    "my codebase",
    "my file",
    "this project",
    "this repo",
    "this code",
    "this file",
    "in the project",
    "in the codebase",
    "in my app",
    "portfolio",
    "agent page",
    "dashboard",
    "component",
)

TRACE_MARK = "\n\n---\n**Model trace:**"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await llm.aclose()


app = FastAPI(title="Jarvis Gateway", lifespan=lifespan)

# Open WebUI and Continue call the gateway from their own processes, not from a
# browser page, so they need no CORS at all. Only list origins that really need it.
_origins = GATEWAY.get("cors_origins") or []
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


def require_api_key(request: Request) -> None:
    if not API_KEY:
        raise HTTPException(503, "JARVIS_API_KEY is not set on the gateway")
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    if not token or not hmac.compare_digest(token.encode(), API_KEY.encode()):
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})


# ── RAG (sync Neo4j driver, always called through asyncio.to_thread) ──────────
try:
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
except Exception:
    driver = None


def get_embedding(text: str) -> list:
    try:
        r = requests.post(
            f"{settings.ollama_base}/api/embed",
            json={"model": settings.embed_model, "input": text[:2000]},
            timeout=15,
        )
        return r.json().get("embeddings", [[]])[0]
    except Exception:
        return []


def rag_query(question: str, top_k: int = 4) -> str:
    if not driver:
        return ""
    try:
        embedding = get_embedding(question)
        if not embedding:
            return ""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (f:File)
                WHERE f.embedding IS NOT NULL
                WITH f, vector.similarity.cosine(f.embedding, $emb) AS score
                ORDER BY score DESC LIMIT $k
                RETURN f.path AS path, f.content_preview AS preview, score
                """,
                emb=embedding,
                k=top_k,
            )
            rows = [dict(r) for r in result]
        if not rows:
            return ""
        parts = ["### Relevant codebase context (via RAG):"]
        for r in rows:
            parts.append(f"\n**File:** `{r['path']}`\n```\n{r['preview']}\n```")
        return "\n".join(parts)
    except Exception as e:
        print(f"[RAG] error: {e}")
        return ""


# ── Conversation handling ─────────────────────────────────────────────────────
def clean_assistant(text: str) -> str:
    """Drop thinking blocks and our own trace appendix before a turn is resent or hashed."""
    text = THINK_RE.sub("", text)
    cut = text.find(TRACE_MARK)
    return (text[:cut] if cut >= 0 else text).strip()


def last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def build_conversation(messages: list[dict], system: str, budget_chars: int = HISTORY_CHARS) -> list[dict]:
    """
    System prompt, then as many recent turns as fit the history budget, ending
    with the latest user message (always kept whole). The client's own system
    messages are appended to ours instead of being dropped.
    """
    end = last_user_index(messages)
    client_system = "\n\n".join(
        text_of(m.get("content")) for m in messages if m.get("role") == "system" and m.get("content")
    )
    turns = []
    for m in messages[: end + 1]:
        role = m.get("role")
        if role == "user":
            turns.append({"role": "user", "content": text_of(m.get("content"))})
        elif role == "assistant":
            content = clean_assistant(text_of(m.get("content")))
            if content:
                turns.append({"role": "assistant", "content": content})

    kept = turns[-1:]
    used = 0
    for m in reversed(turns[:-1]):
        used += len(m["content"])
        if used > budget_chars:
            break
        kept.insert(0, m)
    while len(kept) > 1 and kept[0]["role"] == "assistant":
        kept.pop(0)

    full_system = system + (f"\n\n{client_system}" if client_system else "")
    return [{"role": "system", "content": full_system}] + kept


def system_prompt(intent: str, rag_context: str) -> str:
    parts = ["You are a helpful expert assistant."]
    extra = ROUTES.get(intent, ROUTES["general"])[2]
    if extra:
        parts.append(extra)
    if rag_context:
        parts.append(rag_context)
    return "\n\n".join(parts)


# ── Traces ────────────────────────────────────────────────────────────────────
class TraceStore:
    """Last N traces, keyed by a hash of the answer text the client will send back."""

    def __init__(self, size: int = 256):
        self.size = size
        self._data: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(answer: str) -> str:
        return hashlib.sha256(" ".join(clean_assistant(answer).split()).encode()).hexdigest()

    def put(self, answer: str, trace: str) -> None:
        k = self.key(answer)
        self._data[k] = trace
        self._data.move_to_end(k)
        while len(self._data) > self.size:
            self._data.popitem(last=False)

    def get(self, answer: str) -> str | None:
        return self._data.get(self.key(answer))


TRACES = TraceStore()


def trace_followup(messages: list[dict]) -> str | None:
    """If the latest message asks how the previous answer was made and we still have its trace, return it."""
    end = last_user_index(messages)
    if end < 0:
        return None
    ask = " ".join(text_of(messages[end].get("content")).lower().split())
    if not any(p in ask for p in TRACE_FOLLOWUPS):
        return None
    for m in reversed(messages[:end]):
        if m.get("role") == "assistant":
            return TRACES.get(text_of(m.get("content")))
    return None


def format_trace(requested, labels, source, rag_used, models_used, tracker: StepTimer) -> str:
    return (
        f"{TRACE_MARK}\n"
        f"- Requested model: `{requested}`\n"
        f"- Intent: `{labels['intent']}` ({source})\n"
        f"- Complexity: `{labels['complexity']}`\n"
        f"- RAG used: `{'yes' if rag_used else 'no'}`\n"
        f"- Models: {', '.join(f'`{m}`' for m in models_used)}\n\n" + tracker.summary()
    )


# ── Response helpers ──────────────────────────────────────────────────────────
def completion_body(text: str, model: str) -> dict:
    now = int(time.time())
    return {
        "id": f"jarvis-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse(msg_id: str, model: str, content: str | None = None, finish: str | None = None) -> str:
    delta = {"content": content} if content is not None else {}
    chunk = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def respond(text: str, stream: bool, model: str):
    """A finished answer, as JSON or as a one chunk stream, whichever the client asked for."""
    if not stream:
        return JSONResponse(completion_body(text, model))

    async def gen():
        msg_id = f"jarvis-{int(time.time())}"
        yield sse(msg_id, model, text)
        yield sse(msg_id, model, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def stream_answer(alias: str, convo: list[dict], model: str, on_done):
    """
    Relay tokens as they arrive. Thinking tokens are wrapped in <think> tags,
    which Open WebUI shows as a collapsible block, so a reasoning model is no
    longer silent while it thinks. A model error becomes visible text.
    """
    msg_id = f"jarvis-{int(time.time())}"
    answer: list[str] = []
    thinking = False
    try:
        async for kind, text in llm.stream(alias, convo):
            if kind == "reasoning":
                if not thinking:
                    thinking = True
                    text = "<think>\n" + text
                yield sse(msg_id, model, text)
                continue
            if thinking:
                thinking = False
                yield sse(msg_id, model, "\n</think>\n\n")
            answer.append(text)
            yield sse(msg_id, model, text)
        if thinking:
            yield sse(msg_id, model, "\n</think>\n\n")
    except llm.LLMError as e:
        note = f"\n\n[Jarvis] {e}"
        answer.append(note)
        yield sse(msg_id, model, note)
    tail = on_done("".join(answer))
    if tail:
        yield sse(msg_id, model, tail)
    yield sse(msg_id, model, finish="stop")
    yield "data: [DONE]\n\n"


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def completions(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Body must be JSON"}, status_code=400)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return JSONResponse({"error": "messages must be a list"}, status_code=400)
    stream = bool(body.get("stream"))
    requested = str(body.get("model") or "jarvis")

    end = last_user_index(messages)
    user_msg = text_of(messages[end].get("content")) if end >= 0 else ""
    if not user_msg.strip():
        return JSONResponse({"error": "No user message found"}, status_code=400)

    # "which model answered?" about the previous turn: answer from the trace store.
    previous_trace = trace_followup(messages)
    if previous_trace:
        return respond(previous_trace.strip().removeprefix("---").strip(), stream, requested)

    tracker = StepTimer()
    want_trace = requested == "jarvis-trace" or request.headers.get("x-jarvis-trace", "").lower() in (
        "1",
        "true",
        "yes",
    )

    rag_task = None
    if any(t in user_msg.lower() for t in RAG_TRIGGERS):
        rag_task = asyncio.create_task(asyncio.to_thread(rag_query, user_msg))

    forced = EXPLICIT_MODELS.get(requested)
    if forced:
        labels = {"intent": forced, "complexity": "n/a", "needs_rag": False}
        source = f"explicit model `{requested}`"
    else:
        tracker.start("classify")
        labels, source = await classify.classify(messages, llm.complete)
        tracker.end("classify")
    intent = labels["intent"]

    if rag_task is None and (labels["needs_rag"] or intent in ("code", "code+reasoning")):
        rag_task = asyncio.create_task(asyncio.to_thread(rag_query, user_msg))
    rag_context = ""
    if rag_task is not None:
        tracker.start("rag")
        rag_context = await rag_task
        tracker.end("rag")

    # Design and build: the decomposer (sync, so it runs in a worker thread).
    # It still sees only the latest message; Day 4 reworks it.
    if intent == "code+reasoning":
        tracker.start("task_decomposition")
        decomp = await asyncio.to_thread(decompose_and_execute, user_msg, rag_context)
        tracker.end("task_decomposition")
        answer = decomp["answer"] + f"\n\n*Decomposed into {len(decomp['subtasks'])} subtasks*"
        trace = format_trace(requested, labels, source, bool(rag_context), decomp["models_used"], tracker)
        TRACES.put(answer, trace)
        return respond(answer + (trace if want_trace else ""), stream, requested)

    alias, model_name, _ = ROUTES.get(intent, ROUTES["general"])
    convo = build_conversation(messages, system_prompt(intent, rag_context))
    step = f"llm_{alias}"

    def finish(answer: str) -> str:
        tracker.end(step)
        trace = format_trace(requested, labels, source, bool(rag_context), [model_name], tracker)
        TRACES.put(answer, trace)
        return trace if want_trace else ""

    tracker.start(step)
    if stream:
        return StreamingResponse(
            stream_answer(alias, convo, requested, finish), media_type="text/event-stream"
        )

    try:
        answer = await llm.complete(alias, convo)
    except llm.LLMError as e:
        tracker.end(step)
        return JSONResponse({"error": {"message": str(e), "type": "upstream_error"}}, status_code=502)
    return respond(answer + finish(answer), False, requested)


# ── Other endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "gateway": "jarvis", "port": int(GATEWAY.get("port", 4001))}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "jarvis"} for m in MODEL_IDS]}


# Probe endpoints some clients call during setup; they return nothing sensitive.
@app.get("/v1/openapi.json")
async def openapi_spec():
    return {"openapi": "3.0.0", "info": {"title": "Jarvis Gateway", "version": "2.0.0"}, "paths": {}}


@app.get("/v1/api/config")
async def api_config():
    return {"status": "ok", "name": "Jarvis Gateway", "version": "2.0.0"}


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit(
            f"JARVIS_API_KEY is not set in {settings.env_file}. The gateway will not start without it."
        )
    host = GATEWAY.get("host", "0.0.0.0")
    port = int(GATEWAY.get("port", 4001))
    print(f"🧠 Jarvis gateway on {host}:{port} (API key required)")
    uvicorn.run(app, host=host, port=port)
