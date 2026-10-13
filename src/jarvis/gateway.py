"""
Smart Gateway — port 4001
OpenAI-compatible API that orchestrates:
  1. Prompt enhancement
  2. Intent classification
  3. RAG context (project-aware)
  4. Model routing + chaining
  5. Streaming responses
  6. Model trace on demand with timing + memory
"""

import asyncio
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress

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
from jarvis.config import cors_origins, jarvis_api_key, settings
from jarvis.gateway_metrics import StepTimer
from jarvis.policy import Decision, decide, prefer_decomposer, walk_tier
from jarvis.registry import Registry, bind_snapshot, current_snapshot, mini_only, reset_snapshot, set_registry
from jarvis.task_decomposer import decompose_and_execute

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

TRACE_TRIGGERS = [
    "which model",
    "what model",
    "why did you",
    "how did you",
    "explain your process",
    "show trace",
    "model trace",
    "who answered",
]

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
def chat(
    model_alias: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    _retry: bool = True,
) -> str:
    """Non-streaming chat — returns full response string. Retries once on failure."""
    payload = {"model": model_alias, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        r = requests.post(
            f"{LITELLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if _retry:
            print(f"[chat] {model_alias} failed ({e}) — retrying once...")
            time.sleep(2)
            return chat(model_alias, messages, temperature, max_tokens, _retry=False)
        raise


def _stream_client() -> httpx.AsyncClient:
    """Async LiteLLM client. Tests swap this to inject a transport."""
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=5.0))


async def stream_tokens(deployment: str, device: str, messages: list, max_tokens: int):
    """Yield tokens for one heavy call.

    The read runs on the event loop. Cancelling it closes the HTTP connection
    before the device semaphore is released. CancelledError is not caught.
    """
    payload = {
        "model": deployment,
        "messages": messages,
        "temperature": 0.3,
        "stream": True,
        "max_tokens": max_tokens,
    }
    skipped = 0
    try:
        async with device_semaphore(device):
            async with _stream_client() as client:
                async with client.stream(
                    "POST",
                    f"{LITELLM_BASE}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {LITELLM_KEY}"},
                ) as response:
                    response.raise_for_status()
                    try:
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            try:
                                chunk = json.loads(line[6:])
                                delta = chunk["choices"][0]["delta"].get("content", "")
                            except Exception:
                                skipped += 1
                                continue
                            if delta:
                                yield delta
                    finally:
                        if skipped:
                            log.warning(
                                "skipped %s unreadable stream lines deployment=%s", skipped, deployment
                            )
    except Exception:
        log.exception("stream worker failed deployment=%s", deployment)


# ── Step 1 — Enhance + Classify ───────────────────────────────────────────────
ENHANCE_SYSTEM = """You are a prompt engineer and intent classifier.

Given a user message (possibly casual or brief), do two things:
1. Rewrite it as a clear, detailed prompt that preserves the original intent but adds helpful context and specificity. Keep it concise — do not pad.
2. Classify the intent as ONE of: code | reasoning | code+reasoning | general

Rules for classification:
- code: writing, fixing, reviewing, or explaining specific code — use when asked to show, list, find, or describe code/files
- reasoning: design decisions, architecture, debugging logic, "why", "how should I" — pure thinking, no code needed
- code+reasoning: ONLY when user explicitly wants something DESIGNED AND BUILT from scratch — e.g. "design and implement", "create a new feature", "build this"
- general: concepts, summaries, chat, questions that don't need code or deep reasoning

IMPORTANT: "what does X do", "list my routes", "explain this file", "what are my components" = code (NOT code+reasoning)
ONLY use code+reasoning when the user clearly wants NEW code written after thinking through a solution.

needs_rag should be true when ANY of these apply:
- Query mentions "my project", "my code", "my repo", "my file", "this project", "this code"
- Query is about a specific file, component, function, or feature in a codebase
- Query mentions specific framework terms like "agent page", "dashboard", "component", "route", "API"
- Intent is "code" or "code+reasoning"

Respond ONLY with valid JSON, no markdown:
{
  "enhanced_prompt": "...",
  "intent": "code|reasoning|code+reasoning|general",
  "needs_rag": true|false,
  "rag_reason": "brief reason or empty string"
}
"""


def enhance_and_classify(user_message: str, deployment: str = "classify-mini") -> dict:
    msg_lower = user_message.lower()
    keyword_rag = any(t in msg_lower for t in RAG_TRIGGERS)

    try:
        result = chat(
            deployment,
            [
                {"role": "system", "content": ENHANCE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        clean = result.strip()
        for fence in ["```json", "```"]:
            clean = clean.replace(fence, "")
        clean = clean.strip()

        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]

        parsed = json.loads(clean)

    except Exception as e:
        print(f"[Classify] parse error: {e} — using fallback")
        intent = "general"
        if any(w in msg_lower for w in ["write", "build", "create", "implement", "fix", "code", "function"]):
            intent = "code"
        elif any(w in msg_lower for w in ["why", "how should", "design", "architect", "explain"]):
            intent = "reasoning"
        parsed = {
            "enhanced_prompt": user_message,
            "intent": intent,
            "needs_rag": keyword_rag,
            "rag_reason": "fallback classifier",
        }

    if keyword_rag:
        parsed["needs_rag"] = True
        if not parsed.get("rag_reason"):
            parsed["rag_reason"] = "keyword match"

    return parsed


def prior_turns(messages: list, user_msg: str) -> list[dict]:
    """Turns before the latest copy of the current user message."""
    last = None
    for index, message in enumerate(messages):
        if message.get("role") == "user" and message.get("content") == user_msg:
            last = index
    if last is None:
        return []
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages[:last]
        if message.get("role") in ("user", "assistant") and isinstance(message.get("content"), str)
    ]


def budgeted_messages(
    alias: str, system: str, user: str, rag: str = "", history: list | None = None, handoff: str = ""
):
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
    enhanced_prompt: str,
    rag_context: str,
    tracker: StepTimer,
    history: list | None,
    decision: Decision,
) -> tuple[str, list[str]]:
    system_base = f"You are a helpful expert assistant. {REFERENCE_RULE}"
    history = history or []
    alias = decision.deployment
    label = f"{alias} ({decision.device})"

    if intent == "general":
        tracker.start("llm_general")
        messages, max_tokens = budgeted_messages(alias, system_base, enhanced_prompt, rag_context, history)
        answer = chat(alias, messages, max_tokens=max_tokens)
        tracker.end("llm_general")
        return answer, [label]

    if intent == "code":
        tracker.start("llm_code")
        messages, max_tokens = budgeted_messages(
            alias,
            system_base + "\nWrite clean, production-ready code with comments.",
            enhanced_prompt,
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
        messages, max_tokens = budgeted_messages(
            alias, system_base + extra, enhanced_prompt, rag_context, history
        )
        answer = chat(alias, messages, max_tokens=max_tokens)
        tracker.end("llm_reasoning")
        return answer, [label]

    tracker.start("llm_fallback")
    messages, max_tokens = budgeted_messages(alias, system_base, enhanced_prompt, rag_context, history)
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


# ── Main endpoint ─────────────────────────────────────────────────────────────
@api.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    wants_stream = body.get("stream", False)

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    requested = body.get("model")
    explicit = explicit_model(requested if isinstance(requested, str) else None)
    wants_trace = requested == "jarvis-trace" or any(t in user_msg.lower() for t in TRACE_TRIGGERS)
    history = prior_turns(messages, user_msg)
    tracker = StepTimer()
    snapshot = current_snapshot()

    # Step 1 — parallel enhance+classify AND early RAG keyword check
    msg_lower = user_msg.lower()
    keyword_rag = any(t in msg_lower for t in RAG_TRIGGERS)

    tracker.start("enhance_classify")
    tracker.start("rag")

    rag_future = None
    classify_future = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        if explicit:
            meta = {
                "enhanced_prompt": user_msg,
                "intent": _EXPLICIT_INTENT[explicit],
                "needs_rag": keyword_rag,
                "rag_reason": "explicit model",
            }
            classify_future = None
        else:
            classify_future = executor.submit(enhance_and_classify, user_msg, classify_deployment(snapshot))

        # Start RAG early if keyword match detected
        if keyword_rag:
            rag_future = executor.submit(rag_query, user_msg)

        if classify_future is not None:
            try:
                meta = classify_future.result()
            except Exception as e:
                print(f"[Classify] error: {e}")
                meta = {
                    "enhanced_prompt": user_msg,
                    "intent": "general",
                    "needs_rag": False,
                    "rag_reason": "",
                }

        tracker.end("enhance_classify")

        enhanced_prompt = meta.get("enhanced_prompt", user_msg)
        intent = meta.get("intent", "general")
        needs_rag = meta.get("needs_rag", False)

        # If RAG wasn't started but intent needs it, start now
        rag_context = ""
        if rag_future is None and (needs_rag or intent in ("code", "code+reasoning")):
            rag_future = executor.submit(rag_query, enhanced_prompt)

        # Get RAG result if running
        if rag_future is not None:
            try:
                rag_context = rag_future.result()
            except Exception as e:
                print(f"[RAG] parallel error: {e}")
                rag_context = ""

        tracker.end("rag")

    decision = decide(intent, str(meta.get("complexity") or ""), explicit, snapshot, settings.topology)
    # Degraded design-and-build, or an explicit jarvis-decompose, stays on the mini.
    if prefer_decomposer(intent, explicit, decision, requested == "jarvis-decompose"):
        print("[Decomposer] Routing to task decomposer on the mini...")
        tracker.start("task_decomposition")
        token = bind_snapshot(mini_only(snapshot))
        try:
            async with device_semaphore("mini"):
                decomp = await asyncio.to_thread(decompose_and_execute, enhanced_prompt, rag_context)
        finally:
            reset_snapshot(token)
        tracker.end("task_decomposition")
        decision = Decision(
            decision.tier, "decomposer", "mini", True, decision.reason + "; decomposer on mini"
        )

        answer = decomp["answer"]
        models_used = decomp["models_used"]
        answer += f"\n\n*Decomposed into {len(decomp['subtasks'])} subtasks*"
        if wants_trace:
            answer += (
                f"\n\n---\n**Model trace:**\n- Intent: `{intent}`\n- Tier: `{decision.tier}`\n"
                f"- Device: `{decision.device}`\n- Deployment: `{decision.deployment}`\n"
                f"- Degraded: `{str(decision.degraded).lower()}`\n- Reason: {decision.reason}\n"
                f"- Models: {', '.join(f'`{m}`' for m in models_used)}\n\n" + tracker.summary()
            )

        return JSONResponse(
            {
                "id": f"jarvis-{int(tracker.wall_start)}",
                "object": "chat.completion",
                "created": int(tracker.wall_start),
                "model": "jarvis",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    # Step 3 — streaming path. Heavy calls hold that device's semaphore until the stream ends.
    if wants_stream and not wants_trace and decision.deployment != "decomposer":
        system_msg = f"You are a helpful expert assistant. {REFERENCE_RULE}"
        if intent == "code":
            system_msg += "\nWrite clean, production-ready code with comments."
        elif intent in ("reasoning", "code+reasoning"):
            system_msg += "\nThink step by step. Be thorough."
        stream_messages, max_tokens = budgeted_messages(
            decision.deployment, system_msg, enhanced_prompt, rag_context, history
        )
        msg_id = f"jarvis-{int(time.time())}"

        async def stream_generator():
            async for token in stream_tokens(
                decision.deployment, decision.device, stream_messages, max_tokens
            ):
                chunk = {
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "jarvis",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    if decision.deployment != "decomposer":
        async with device_semaphore(decision.device):
            answer, models_used = await asyncio.to_thread(
                execute, intent, enhanced_prompt, rag_context, tracker, history, decision
            )

    if wants_trace:
        trace = (
            f"\n\n---\n**Model trace:**\n"
            f"- Intent: `{intent}`\n"
            f"- Tier: `{decision.tier}`\n"
            f"- Device: `{decision.device}`\n"
            f"- Deployment: `{decision.deployment}`\n"
            f"- Degraded: `{str(decision.degraded).lower()}`\n"
            f"- Reason: {decision.reason}\n"
            f"- RAG used: `{'yes' if rag_context else 'no'}`\n"
            f"- Models: {', '.join(f'`{m}`' for m in models_used)}\n"
            f"- Enhanced prompt: _{enhanced_prompt}_\n\n" + tracker.summary()
        )
        answer += trace

    return JSONResponse(
        {
            "id": f"jarvis-{int(tracker.wall_start)}",
            "object": "chat.completion",
            "created": int(tracker.wall_start),
            "model": "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


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
