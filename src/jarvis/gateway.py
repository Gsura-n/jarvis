"""
Smart Gateway — port 4001
OpenAI-compatible API that orchestrates:
  1. Intent classification (labels only, ADR 0008)
  2. RAG context (project-aware)
  3. Model routing + chaining
  4. Streaming responses
  5. Model trace on demand with timing + memory
"""

import asyncio
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from functools import partial

import httpx
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from neo4j import GraphDatabase
from starlette.middleware.cors import CORSMiddleware

from jarvis.budget import (
    REFERENCE_RULE,
    answer_token_budget,
    assemble_messages,
    context_window,
    history_trim_block,
    token_caps,
)
from jarvis.classify import classify
from jarvis.config import cors_origins, jarvis_api_key, settings
from jarvis.gateway_metrics import StepTimer
from jarvis.messages import TRACE_MARK, clean_assistant, system_text, text_of
from jarvis.policy import Decision, decide, prefer_decomposer, walk_tier
from jarvis.registry import Registry, bind_snapshot, current_snapshot, mini_only, reset_snapshot, set_registry
from jarvis.task_decomposer import decompose_and_execute
from jarvis.traces import TraceStore, trace_followup

log = logging.getLogger("jarvis.gateway")

# ── Config — everything comes from jarvis.config (env file + jarvis.yaml) ─────
LITELLM_BASE = settings.litellm_base
LITELLM_KEY = settings.litellm_key
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
OLLAMA_BASE = settings.ollama_base
EMBED_MODEL = settings.embed_model

# Validate required credentials at startup
_missing = settings.missing_required()
if _missing:
    print(f"⚠️  Missing required env vars: {', '.join(_missing)}")
    print(f"   Add them to {settings.env_file}")

_EXPLICIT_INTENT = {"general": "general", "coding": "code", "reasoning": "reasoning"}
_semaphores: dict[str, asyncio.Semaphore] = {}

# Traces of the last answers, so "which model answered?" can be served without a model call.
TRACES = TraceStore()

RAG_TRIGGERS = [
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
]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not jarvis_api_key():
        log.error("JARVIS_API_KEY is empty; every protected route will return 401")
    registry = Registry(settings.topology)
    set_registry(registry)
    await registry.poll_once()
    task = asyncio.create_task(registry.run())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        set_registry(None)


# ── Neo4j ─────────────────────────────────────────────────────────────────────
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
except Exception:
    driver = None


def explicit_model(name: str | None) -> str | None:
    return name if name in _EXPLICIT_INTENT else None


def device_semaphore(device: str) -> asyncio.Semaphore:
    """One heavy call at a time per machine. Classifier and embedder do not take this."""
    sem = _semaphores.get(device)
    if sem is None:
        raw = settings.topology.get("devices", {}).get(device, {}).get("heavy_concurrency", 1)
        sem = asyncio.Semaphore(max(1, int(raw)))
        _semaphores[device] = sem
    return sem


def classify_deployment(snapshot) -> str:
    return walk_tier("classify", snapshot, settings.topology, "classify").deployment or "classify-mini"


def get_embedding(text: str) -> list:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:2000]},
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


# ── LiteLLM helpers ───────────────────────────────────────────────────────────
def _retryable(e: Exception) -> bool:
    """Connection errors and 5xx may work next time. A 4xx will fail the same way."""
    if isinstance(e, requests.HTTPError):
        return e.response is not None and e.response.status_code >= 500
    return isinstance(e, requests.ConnectionError)


def chat(
    model_alias: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    retry: bool = True,
    timeout: float = 600,
    **params,
) -> str:
    """Non-streaming chat — returns full response string.

    Retries once on a connection error or 5xx, never on 4xx. Extra params (for example
    response_format) go to LiteLLM as given."""
    payload = {"model": model_alias, "messages": messages, "temperature": temperature, **params}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        r = requests.post(
            f"{LITELLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if retry and _retryable(e):
            print(f"[chat] {model_alias} failed ({e}) — retrying once...")
            time.sleep(2)
            return chat(
                model_alias, messages, temperature, max_tokens, retry=False, timeout=timeout, **params
            )
        raise


REASON_CHARS = 300


def short_reason(text: str) -> str:
    """One line, at most REASON_CHARS, with the LiteLLM key masked."""
    text = " ".join(str(text).split())
    if LITELLM_KEY:
        text = text.replace(LITELLM_KEY, "***")
    return text[:REASON_CHARS]


def _status_reason(status: int, body: str) -> str:
    """HTTP status plus the upstream error message. Never headers or the request."""
    try:
        err = json.loads(body).get("error")
        body = str(err.get("message") if isinstance(err, dict) else err or body)
    except Exception:
        pass
    return short_reason(f"HTTP {status}: {body}")


def describe_error(e: Exception) -> str:
    """A short reason for a failed upstream call."""
    response = getattr(e, "response", None)
    if isinstance(e, requests.HTTPError) and response is not None:
        return _status_reason(response.status_code, response.text)
    if isinstance(e, (requests.Timeout, httpx.TimeoutException)):
        return "timed out"
    if isinstance(e, (requests.ConnectionError, httpx.TransportError)):
        return "could not reach the model server"
    return short_reason(f"{type(e).__name__}")


def _stream_client() -> httpx.AsyncClient:
    """Async LiteLLM client. Tests swap this to inject a transport."""
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=5.0))


async def stream_tokens(
    deployment: str, device: str, messages: list, max_tokens: int, outcome: dict | None = None
):
    """Yield tokens for one heavy call.

    The read runs on the event loop. Cancelling it closes the HTTP connection
    before the device semaphore is released. CancelledError is not caught.

    Thinking (reasoning_content) is yielded inside a <think> block that closes before
    the first answer token. An upstream or transport error ends the stream with one
    "[Jarvis] <deployment>: <reason>" token and sets outcome["failed"] when a dict is given.
    """
    payload = {
        "model": deployment,
        "messages": messages,
        "temperature": 0.3,
        "stream": True,
        "max_tokens": max_tokens,
    }
    skipped = 0
    thinking = False
    failure = None
    try:
        async with device_semaphore(device):
            async with _stream_client() as client:
                async with client.stream(
                    "POST",
                    f"{LITELLM_BASE}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {LITELLM_KEY}"},
                ) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        failure = _status_reason(response.status_code, body)
                    else:
                        try:
                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                                    continue
                                try:
                                    delta = json.loads(line[6:])["choices"][0]["delta"]
                                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                                    text = delta.get("content") or ""
                                except Exception:
                                    skipped += 1
                                    continue
                                if thought:
                                    if not thinking:
                                        thinking = True
                                        yield "<think>\n"
                                    yield thought
                                if text:
                                    if thinking:
                                        thinking = False
                                        yield "\n</think>\n\n"
                                    yield text
                        finally:
                            if skipped:
                                log.warning(
                                    "skipped %s unreadable stream lines deployment=%s", skipped, deployment
                                )
    except httpx.HTTPError as e:
        failure = describe_error(e)
    except Exception:
        log.exception("stream worker failed deployment=%s", deployment)
    if thinking:
        yield "\n</think>\n\n"
    if failure:
        if outcome is not None:
            outcome["failed"] = True
        log.warning("upstream error deployment=%s reason=%s", deployment, failure)
        yield f"\n\n[Jarvis] {deployment}: {failure}"


# ── Step 1 — History ──────────────────────────────────────────────────────────
def prior_turns(messages: list, user_msg: str) -> list[dict]:
    """Turns before the latest copy of the current user message."""
    last = None
    for index, message in enumerate(messages):
        if message.get("role") == "user" and text_of(message.get("content")) == user_msg:
            last = index
    if last is None:
        return []
    turns = []
    for message in messages[:last]:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = text_of(message.get("content"))
        if role == "assistant":
            text = clean_assistant(text)
            if not text:
                continue
        turns.append({"role": role, "content": text})
    return turns


def budgeted_messages(
    alias: str,
    system: str,
    user: str,
    rag: str = "",
    history: list | None = None,
    handoff: str = "",
    client_system: str = "",
):
    """Our system prompt stays first and is never changed by the client's, which follows a blank line."""
    if client_system:
        system = f"{system}\n\n{client_system}"
    max_tokens = answer_token_budget(alias, settings.topology)
    messages = assemble_messages(
        system=system,
        history=history or [],
        rag=rag,
        handoff=handoff,
        user=user,
        max_tokens=max_tokens,
        num_ctx=context_window(settings.topology, alias),
        caps=token_caps(settings.topology),
        block=history_trim_block(settings.topology),
    )
    return messages, max_tokens


# ── Step 2 — Execute (non-streaming) ─────────────────────────────────────────
def execute(
    intent: str,
    user_msg: str,
    rag_context: str,
    tracker: StepTimer,
    history: list | None,
    decision: Decision,
    client_system: str = "",
) -> tuple[str, list[str]]:
    system_base = f"You are a helpful expert assistant. {REFERENCE_RULE}"
    history = history or []
    alias = decision.deployment
    label = f"{alias} ({decision.device})"
    build = partial(budgeted_messages, client_system=client_system)

    if intent == "general":
        tracker.start("llm_general")
        messages, max_tokens = build(alias, system_base, user_msg, rag_context, history)
        answer = chat(alias, messages, max_tokens=max_tokens)
        tracker.end("llm_general")
        return answer, [label]

    if intent == "code":
        tracker.start("llm_code")
        messages, max_tokens = build(
            alias,
            system_base + "\nWrite clean, production-ready code with comments.",
            user_msg,
            rag_context,
            history,
        )
        answer = chat(alias, messages, max_tokens=max_tokens)
        tracker.end("llm_code")
        return answer, [label]

    if intent == "reasoning" or intent == "code+reasoning":
        # code+reasoning is one call when the large tier is really available (ADR 0007).
        extra = "\nThink step by step. Be thorough."
        if intent == "code+reasoning":
            extra += "\nDesign the solution, then implement it in the same answer."
        tracker.start("llm_reasoning")
        messages, max_tokens = build(alias, system_base + extra, user_msg, rag_context, history)
        answer = chat(alias, messages, max_tokens=max_tokens)
        tracker.end("llm_reasoning")
        return answer, [label]

    tracker.start("llm_fallback")
    messages, max_tokens = build(alias, system_base, user_msg, rag_context, history)
    answer = chat(alias, messages, max_tokens=max_tokens)
    tracker.end("llm_fallback")
    return answer, [label]


def require_api_key(request: Request) -> None:
    """Shared key check. There are no roles yet: missing or wrong key is 401."""
    expected = jarvis_api_key()
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    ok = (
        scheme.lower() == "bearer"
        and bool(expected)
        and hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
    )
    if not ok:
        raise HTTPException(status_code=401, detail="missing or invalid api key")


# One dependency for every route except GET /health.
api = APIRouter(dependencies=[Depends(require_api_key)])


# ── Responses and traces ──────────────────────────────────────────────────────
def format_trace(
    intent: str,
    decision: Decision,
    rag_used: bool,
    models_used: list[str],
    label_source: str,
    tracker: StepTimer,
) -> str:
    """The appendix. It starts with TRACE_MARK, which clean_assistant cuts from replayed history."""
    return (
        f"{TRACE_MARK}\n"
        f"- Intent: `{intent}`\n"
        f"- Tier: `{decision.tier}`\n"
        f"- Device: `{decision.device}`\n"
        f"- Deployment: `{decision.deployment}`\n"
        f"- Degraded: `{str(decision.degraded).lower()}`\n"
        f"- Reason: {decision.reason}\n"
        f"- RAG used: `{'yes' if rag_used else 'no'}`\n"
        f"- Models: {', '.join(f'`{m}`' for m in models_used)}\n"
        f"- Labels from: `{label_source}`\n\n" + tracker.summary()
    )


def completion_json(tracker: StepTimer, answer: str) -> JSONResponse:
    return JSONResponse(
        {
            "id": f"jarvis-{int(tracker.wall_start)}",
            "object": "chat.completion",
            "created": int(tracker.wall_start),
            "model": "jarvis",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def sse_line(msg_id: str, content: str) -> str:
    chunk = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "jarvis",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def stored_trace_response(trace: str, stream: bool, tracker: StepTimer):
    """A trace follow-up: the stored text, as a stream or as JSON like the client asked."""
    text = trace.strip()
    if not stream:
        return completion_json(tracker, text)
    msg_id = f"jarvis-{int(tracker.wall_start)}"
    return StreamingResponse(
        iter([sse_line(msg_id, text), "data: [DONE]\n\n"]), media_type="text/event-stream"
    )


# ── Main endpoint ─────────────────────────────────────────────────────────────
@api.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    wants_stream = body.get("stream", False)

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = text_of(m.get("content"))
            break

    if not user_msg:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    requested = body.get("model")
    explicit = explicit_model(requested if isinstance(requested, str) else None)
    wants_trace = requested == "jarvis-trace" or request.headers.get("x-jarvis-trace") == "1"
    tracker = StepTimer()

    stored = trace_followup(messages, TRACES)
    if stored:
        return stored_trace_response(stored, bool(wants_stream), tracker)

    history = prior_turns(messages, user_msg)
    client_system = system_text(messages)
    snapshot = current_snapshot()

    # Step 1 — parallel classify (labels only) AND early RAG keyword check
    msg_lower = user_msg.lower()
    keyword_rag = any(t in msg_lower for t in RAG_TRIGGERS)

    tracker.start("classify")
    tracker.start("rag")

    rag_future = None
    classify_future = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        if explicit:
            labels = {"intent": _EXPLICIT_INTENT[explicit], "complexity": "simple", "needs_rag": False}
            label_source = "explicit model"
        else:
            classify_future = executor.submit(classify, messages, chat, classify_deployment(snapshot))

        # Keywords only start retrieval early. They never decide the intent.
        if keyword_rag:
            rag_future = executor.submit(rag_query, user_msg)

        if classify_future is not None:
            labels, label_source = classify_future.result()

        tracker.end("classify")

        intent = labels["intent"]
        needs_rag = labels["needs_rag"] or keyword_rag

        # If RAG wasn't started but the labels need it, start now
        rag_context = ""
        if rag_future is None and (needs_rag or intent in ("code", "code+reasoning")):
            rag_future = executor.submit(rag_query, user_msg)

        # Get RAG result if running
        if rag_future is not None:
            try:
                rag_context = rag_future.result()
            except Exception as e:
                print(f"[RAG] parallel error: {e}")
                rag_context = ""

        tracker.end("rag")

    decision = decide(intent, labels["complexity"], explicit, snapshot, settings.topology)
    # Degraded design-and-build, or an explicit jarvis-decompose, stays on the mini.
    if prefer_decomposer(intent, explicit, decision, requested == "jarvis-decompose"):
        print("[Decomposer] Routing to task decomposer on the mini...")
        tracker.start("task_decomposition")
        token = bind_snapshot(mini_only(snapshot))
        try:
            async with device_semaphore("mini"):
                decomp = await asyncio.to_thread(decompose_and_execute, user_msg, rag_context)
        finally:
            reset_snapshot(token)
        tracker.end("task_decomposition")
        decision = Decision(
            decision.tier, "decomposer", "mini", True, decision.reason + "; decomposer on mini"
        )

        answer = decomp["answer"]
        models_used = decomp["models_used"]
        answer += f"\n\n*Decomposed into {len(decomp['subtasks'])} subtasks*"
        trace = format_trace(intent, decision, bool(rag_context), models_used, label_source, tracker)
        TRACES.put(answer, trace)
        return completion_json(tracker, answer + trace if wants_trace else answer)

    # Step 3 — streaming path. Heavy calls hold that device's semaphore until the stream ends.
    if wants_stream and decision.deployment != "decomposer":
        system_msg = f"You are a helpful expert assistant. {REFERENCE_RULE}"
        if intent == "code":
            system_msg += "\nWrite clean, production-ready code with comments."
        elif intent in ("reasoning", "code+reasoning"):
            system_msg += "\nThink step by step. Be thorough."
        stream_messages, max_tokens = budgeted_messages(
            decision.deployment, system_msg, user_msg, rag_context, history, client_system=client_system
        )
        msg_id = f"jarvis-{int(time.time())}"

        async def stream_generator():
            outcome: dict = {}
            parts: list[str] = []
            tracker.start("llm_stream")
            async for token in stream_tokens(
                decision.deployment, decision.device, stream_messages, max_tokens, outcome
            ):
                parts.append(token)
                yield sse_line(msg_id, token)
            tracker.end("llm_stream")
            if not outcome.get("failed"):
                # Built after the stream ends, so the timings are real. A failed stream gets no trace.
                label = f"{decision.deployment} ({decision.device})"
                trace = format_trace(intent, decision, bool(rag_context), [label], label_source, tracker)
                TRACES.put("".join(parts), trace)
                if wants_trace:
                    yield sse_line(msg_id, trace)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    if decision.deployment != "decomposer":
        try:
            async with device_semaphore(decision.device):
                answer, models_used = await asyncio.to_thread(
                    execute, intent, user_msg, rag_context, tracker, history, decision, client_system
                )
        except Exception as e:
            reason = describe_error(e)
            log.warning("answer call failed deployment=%s reason=%s", decision.deployment, reason)
            return JSONResponse({"error": {"message": f"{decision.deployment}: {reason}"}}, status_code=502)

    trace = format_trace(intent, decision, bool(rag_context), models_used, label_source, tracker)
    TRACES.put(answer, trace)
    return completion_json(tracker, answer + trace if wants_trace else answer)


# ── Other endpoints ───────────────────────────────────────────────────────────
@api.get("/devices")
def list_devices():
    return current_snapshot().as_dict()


def health():
    return {"status": "ok", "gateway": "jarvis", "port": 4001}


@api.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "jarvis", "object": "model", "owned_by": "jarvis"},
            {"id": "coding", "object": "model", "owned_by": "jarvis"},
            {"id": "reasoning", "object": "model", "owned_by": "jarvis"},
            {"id": "general", "object": "model", "owned_by": "jarvis"},
            {"id": "jarvis-trace", "object": "model", "owned_by": "jarvis"},
            {"id": "jarvis-decompose", "object": "model", "owned_by": "jarvis"},
        ],
    }


@api.get("/v1/openapi.json")
def openapi_spec():
    return {"openapi": "3.0.0", "info": {"title": "Jarvis Gateway", "version": "1.0.0"}, "paths": {}}


@api.get("/v1/api/config")
def api_config():
    return {"status": "ok", "name": "Jarvis Gateway", "version": "1.0.0"}


def create_app() -> FastAPI:
    """Build the gateway. CORS origins are fixed at build time, so tests call this after setenv."""
    application = FastAPI(
        title="Jarvis Smart Gateway",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    application.include_router(api)
    application.add_api_route("/health", health, methods=["GET"])
    return application


app = create_app()


if __name__ == "__main__":
    print("🧠 Jarvis Smart Gateway starting on port 4001...")
    print("   Enhance → Classify → RAG → Route → Stream")
    print("   Point your tools to: http://localhost:4001/v1")
    uvicorn.run(app, host="0.0.0.0", port=4001)
